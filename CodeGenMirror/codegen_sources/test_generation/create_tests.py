# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import math
import os
import pdb
import subprocess
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path, PosixPath

import argparse
from submitit import AutoExecutor, LocalExecutor
from tqdm import tqdm
import pandas as pd
from utils import add_root_to_path

add_root_to_path()

from codegen_sources.preprocessing.lang_processors.tree_sitter_processor import (
    TREE_SITTER_ROOT,
)
from codegen_sources.model.src.utils import (
    get_java_bin_path,
    TREE_SITTER_ROOT,
    MAX_VIRTUAL_MEMORY_JAVA,
    limit_virtual_memory
)

from codegen_sources.preprocessing.lang_processors.lang_processor import LangProcessor
from codegen_sources.preprocessing.utils import bool_flag
import numpy as np
from codegen_sources.preprocessing.lang_processors.java_processor import JavaProcessor
from codegen_sources.test_generation.utils import chunks


EVOSUITE_JAR_PATH = Path(
    Path(__file__).absolute().parent.joinpath("evosuite-1.1.0.jar")
)
assert (
    EVOSUITE_JAR_PATH.is_file()
), "EvoSuite Jar is missing, run wget https://github.com/EvoSuite/evosuite/releases/download/v1.1.0/evosuite-1.1.0.jar"

MUTATION_SCORE_CUTOFF = 0.9

REPORT_FILE = "statistics.csv"

MAX_ARRAY_SIZE_CLUSTER=6
ARRAY_PARALLELISM=3
MAX_JAVA_MEM = 4096

MAX_CPUS=80

COLUMNS = ['TARGET_CLASS', 'Random_Seed', 'criterion', 'Size', 'HadUnstableTests',
       'Length', 'Total_Branches', 'Covered_Branches', 'BranchCoverage',
       'Lines', 'Coverage', 'Covered_Lines', 'LineCoverage', 'MethodCoverage',
       'Total_Goals', 'Covered_Goals', 'MutationScore', 'OutputCoverage',
       'folder', 'path_to_test']


def write_javacode_onefunctionperfile(
    codestring: str, line_number: int, folder: PosixPath, with_id: bool = False
):
    if "java.io.File(" in codestring.replace(
        " ", ""
    ) or "io.FileWriter" in codestring.replace(" ", ""):
        return
    functionname = codestring.split("(")[0].strip().split(" ")[-1]
    if with_id:
        assert " | " in codestring, f'missing " | " in input: {codestring}'
        id_string, codestring = codestring.split(" | ", 1)
        classname = f"CLASS_{id_string}"
    else:
        classname = "CLASS_" + functionname.upper() + f"_{line_number}"
    print(classname)
    filepath = folder.joinpath(classname + ".java")
    writefile = open(filepath, "w")
    writefile.write(
        """
import java.util.*;
import java.util.stream.*;
import java.lang.*;
class Pair {
public static <T, U> Map.Entry<T, U> of(T first, U second) {
    return new AbstractMap.SimpleEntry<>(first, second);
    }
}
"""
    )
    lang_processor = JavaProcessor(root_folder=TREE_SITTER_ROOT)
    writefile.write("public class " + classname + "{\n")
    code = codestring.replace("\r", "")
    writefile.write(lang_processor.detokenize_code(code))
    writefile.write("}\n")
    writefile.close()


def run_command_compile_java_file(folderpath):
    print(f"compiling files in {folderpath}")
    files = os.listdir(folderpath)
    executor = ThreadPoolExecutor(max_workers=MAX_CPUS)
    jobs = []
    for file in files:
        jobs.append(executor.submit(compile_file, file, folderpath))
    [j.result() for j in jobs]


def compile_file(file, folderpath):
    try:
        proc = subprocess.Popen(
            f"{limit_virtual_memory(MAX_VIRTUAL_MEMORY_JAVA)}; cd {folderpath} && {os.path.join(get_java_bin_path(), 'javac')} "
            + file,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            executable="/bin/bash",
        )
        out, err = proc.communicate(timeout=100)
    except subprocess.TimeoutExpired:
        return

    err = err.decode("utf-8").strip()
    if len(err) > 0:
        print(err)


def run_command_test_generation(folderpath):
    print(f"Generating tests in {folderpath}")
    executor = ThreadPoolExecutor(max_workers=MAX_CPUS)

    files = os.listdir(folderpath)
    jobs = []
    report_dirs = []
    for file in [f for f in files if f.endswith(".class")]:
        report_name = "es_report_" + file.replace(".class", "")
        report_dirs.append(report_name)
        jobs.append(executor.submit(create_tests, file, folderpath, report_name))

    job_res = [j.result() for j in jobs]
    if len(job_res) > 0:
        print(
            f"Percentage of timeouts: {len([j for j in job_res if j == 'timeout'])/len(job_res):.2%}"
        )
    # print(f"folderpath: {folderpath}")
    consolidated_report_path = get_consolidated_report_path(folderpath)
    # print(f"consolidated_report_path: {consolidated_report_path}")
    consolidated_report_path.mkdir(exist_ok=True)
    consolidated_report_path = consolidated_report_path.joinpath(REPORT_FILE)
    # print(f"consolidated_report_path: {consolidated_report_path}")
    # print("Writing report to: {} with report_dirs: {}".format(consolidated_report_path, report_dirs))
    consolidate_reports(consolidated_report_path, report_dirs, folderpath)


def get_consolidated_report_path(folderpath):
    return Path(folderpath).joinpath("es-consolidated-report")


def create_tests(file, folderpath, report_name):
    print(file)
    doutput = "-Doutput_variables=configuration_id,TARGET_CLASS,criterion,Size,Length,MutationScore"
    cmd = (
        f"{os.path.join(get_java_bin_path(), 'java')} -jar {EVOSUITE_JAR_PATH} -class "
        + file.replace(".class", "")
        + f" -projectCP . "
        f'-criterion "LINE:BRANCH:WEAKMUTATION:OUTPUT:METHOD:CBRANCH:STRONGMUTATION" '
        f"{doutput} "
        f" -Dshow_progress=false "
        f"-Dassertion_strategy=MUTATION "
        f"-Dminimize=true "
        f"-Dsearch_budget=20 "
        f"-Ddouble_precision=0.0001 "
        f"-Dmax_mutants_per_test 200 "
        f'-Danalysis_criteria="LINE,BRANCH,EXCEPTION,WEAKMUTATION,OUTPUT,METHOD,METHODNOEXCEPTION,CBRANCH,STRONGMUTATION" '
        f"-Doutput_variables=TARGET_CLASS,Random_Seed,criterion,Size,HadUnstableTests,Length,Total_Branches,Covered_Branches,BranchCoverage,Lines,Coverage,Covered_Lines,LineCoverage,MethodCoverage,Size,Length,Total_Goals,Covered_Goals,MutationScore,OutputCoverage "
        f"-Dmax_int {int(math.sqrt(2 ** 31 - 1))} "
        f"-Dreport_dir={report_name} "
        f"-mem={MAX_JAVA_MEM} "
        f"-Dextra_timeout=180 "
    )
    print(cmd)
    try:
        return subprocess.call(
            cmd, shell=True, timeout=1000, cwd=folderpath, executable="/bin/bash"
        )
    except subprocess.TimeoutExpired:
        return "timeout"


def consolidate_reports(consolidated_report_path, report_dirs, folderpath):
    print(f"Consolidating reports in {folderpath}")
    with open(consolidated_report_path, "w") as output_report:
        header_printed = False
        for report_dir in report_dirs:
            # print(f"consolidating report: {report_dir}")
            report_path = Path(folderpath).joinpath(report_dir).joinpath(REPORT_FILE)
            # print(f"report is file : {report_path.is_file()}")
            if report_path.is_file():
                with open(report_path, "r") as f:
                    report_lines = f.readlines()
                # print(f"report_lines written: {report_lines if not header_printed else report_lines[1:]}")
                output_report.writelines(
                    report_lines if not header_printed else report_lines[1:]
                )
                header_printed = True
                report_path.unlink()
                report_path.parent.rmdir()


def generate_javafiles_withclass(filepath: PosixPath, output_folder: PosixPath):
    print(f"creating files from {filepath} in {output_folder}")
    lines = open(filepath).readlines()
    for i, line in enumerate(lines):
        write_javacode_onefunctionperfile(line, i, output_folder, with_id=True)


def generate_tests_pipeline(in_file: PosixPath, out_path: PosixPath):
    print(f"Creating tests for {in_file}, outputting them in {out_path}")
    out_path.mkdir(exist_ok=True)
    generate_javafiles_withclass(in_file, out_path)
    run_command_compile_java_file(out_path)
    run_command_test_generation(out_path)


def output_selected_tests_summary(tests_path):
    subfolders = [p for p in list(tests_path.glob("*")) if not str(p).endswith("/log")]
    csv_dfs = []
    # pdb.set_trace()
    for folder in subfolders:
        csv_file = get_consolidated_report_path(folder).joinpath("statistics.csv")
        if csv_file.is_file() and len(open(csv_file).readlines()) > 1:
            csv = pd.read_csv(csv_file)
            csv["folder"] = folder
            csv_dfs.append(csv)
    # pdb.set_trace()
    concat_df = pd.concat(csv_dfs).reset_index(drop=True)
    concat_df = concat_df[concat_df["TARGET_CLASS"].apply(lambda x: not pd.isna(x))]
    concat_df["path_to_test"] = concat_df.apply(
        lambda row: row["folder"]
        .joinpath("evosuite-tests")
        .joinpath(row["TARGET_CLASS"] + "_ESTest.java"),
        axis="columns",
    )
    test_exists = concat_df["path_to_test"].apply(lambda x: x.is_file())
    print(
        f"{(~test_exists).sum() / len(test_exists):.2%} of the tests in the summary could not be found"
    )
    concat_df = concat_df[test_exists]

    concat_df.to_csv(tests_path.joinpath("tests_summary.csv"), index=False)
    test_string = []
    for p in concat_df.path_to_test:
        assert p.is_file(), f"test {p} does not exist"
        with open(p, "r", encoding="utf8") as input_file:
            test_string.append(input_file.read())

    concat_df["tests_strings"] = test_string

    selected_df = concat_df[(concat_df.MutationScore > MUTATION_SCORE_CUTOFF)]
    selected_df = selected_df[selected_df.path_to_test.apply(lambda x: x.is_file())]
    selected_df.to_csv(tests_path.joinpath("selected_tests_summary.csv"), index=False)

    selected_df.to_csv(tests_path.joinpath("selected_tests.csv"), index=False)


if __name__ == "__main__":
    print("#" * 10, "Creating Tests", "#" * 10)
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--input_path", help="path to the input files",
    )
    parser.add_argument(
        "--output_path", type=str, help="where the files should be outputed",
    )
    parser.add_argument(
        "--local",
        type=bool_flag,
        default=True,
        help="True if you want to run the processing pipeline locally, false if want to use submitit.",
    )
    parser.add_argument(
        "--rerun",
        type=bool_flag,
        default=False,
        help="True if you want to run the processing pipeline locally, false if want to use submitit.",
    )
    args = parser.parse_args()
    input_path = Path(args.input_path)
    assert input_path.exists(), f"{input_path} does not exist"
    output_path = Path(args.output_path)
    output_path.mkdir(exist_ok=True)
    if args.local is False:
        cluster = AutoExecutor(output_path.joinpath("log"))
        cluster.update_parameters(cpus_per_task=MAX_CPUS, mem_gb=1000)
        cluster.update_parameters(slurm_partition="sched_system_all_8",
                                  slurm_array_parallelism=ARRAY_PARALLELISM)
    else:
        cluster = LocalExecutor(output_path.joinpath("log"))
    cluster.update_parameters(timeout_min=1400)

    input_path = Path(args.input_path)
    input_path = input_path.joinpath("deduped")
    if input_path.is_file():
        infiles = [input_path]
    else:
        infiles = sorted(list(input_path.glob("java.000*.sa.tok")))

    out_folder = Path(args.output_path)
    sub_out_folders = [
        out_folder.joinpath(func_file.name.replace(".", "_")) for func_file in infiles
    ]
    if not args.rerun:
        indices_to_run = [
            i
            for i, p in enumerate(sub_out_folders)
            if not (
                get_consolidated_report_path(p).is_dir()
                and get_consolidated_report_path(p).joinpath(REPORT_FILE).is_file()
            )
        ]
        print(
            f"Running on the remaining {len(indices_to_run)} among {len(sub_out_folders)} files"
        )
        infiles = np.array(infiles)[indices_to_run]
        sub_out_folders = np.array(sub_out_folders)[indices_to_run]
    ## todo: also should do some chunking here not to overrun the cluster limit
    infiles_chunks = list(chunks(list(infiles), MAX_ARRAY_SIZE_CLUSTER))
    sub_out_folders_chunks = list(chunks(list(sub_out_folders), MAX_ARRAY_SIZE_CLUSTER))
    outer_pbar = tqdm(total=len(infiles), desc="Processing files")
    for i, (infiles_chunk, sub_out_folders_chunk) in enumerate(zip(infiles_chunks, sub_out_folders_chunks)):
        jobs = cluster.map_array(
            generate_tests_pipeline,
            infiles_chunk,
            sub_out_folders_chunk,
        )
        for j in jobs:
            j.result()
            outer_pbar.update(1)
            outer_pbar.set_description(f"Processing files, on chunk {i} of {len(infiles_chunks)}")
    # jobs = cluster.map_array(generate_tests_pipeline, infiles, sub_out_folders)
    # for j in tqdm(jobs):
    #     j.result()
    output_selected_tests_summary(out_folder)
    print("\n" * 2)