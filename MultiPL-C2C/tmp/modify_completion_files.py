import os
import shutil
from pathlib import Path
import itertools
from string import ascii_lowercase

from tqdm import tqdm
import gzip
import json

from dataset_builder.utils import find_all, get_source_code_from_prompt, SHORT2CANONICAL
from inference.chatgpt_utils import truncate_after_additional_stops, cleanup_completion_simple
from src.single_experiment_error_types import *
from codegen_sources.model.src.utils import TREE_SITTER_ROOT
from codegen_sources.preprocessing.lang_processors.lang_processor import LangProcessor


def remove_lang_from_explanation(exp: str, lang_short: str):
    lang = SHORT2CANONICAL[lang_short]
    if lang in exp.lower():
        start = exp.lower().find(lang)
        while exp[start] != ".":
            if start < 0:
                x=1
            start -= 1
        exp = exp[:start+1]+"\n\n"

    if any([w in exp.lower() for w in ["rewrite"]]):
        x=1
    return exp



def open_json(fpath: Path, mode: str):
    return gzip.open(fpath, mode + "t") if fpath.suffix == ".gz" else open(fpath, mode)


def modify_files(data_dir):
    with open(data_dir.replace("dump", "translation_prompts") + ".json") as f:
        og_data = json.load(f)
        og_data = {d["name"]: d for d in og_data}
    translator = __import__(f"humaneval_to_{list(og_data.values())[0]['language']}").Translator()
    cnt = 0
    for path in tqdm(itertools.chain(Path(data_dir).glob("*.json"), Path(data_dir).glob("*.results.json.gz"))):
        if ".results.json" in str(path):
            continue

        with open_json(path, "r") as f:
            data = json.load(f)

        new_completions = [cleanup_completion_simple(c, data["translation_prompt"]) for c in data["completions"]]
        if any([new_c != c for new_c, c in zip(new_completions, data["completions"])]):
            data["completions"] = new_completions

        # if og_data[data["name"]]["tests"] != [data["tests"]]:
        #     data["tests"] = og_data[data["name"]]["tests"]
        # exp = get_source_code_from_prompt(data["translation_prompt"])
        # new_exp = remove_lang_from_explanation(exp, data["language"])
        # if exp!= new_exp:
        #     data["translation_prompt"] = data["translation_prompt"].replace(exp, new_exp)
        #     data["completions"] = []
            cnt += 1
            # if os.path.exists(str(path).replace(".json", ".results.json")):
            #     os.remove(str(path).replace(".json",  ".results.json"))
        # with open(path, "w") as f:
        #     json.dump(data, f, indent=2)
        # with open(Path(str(path).replace(".json",".results.json")), "w") as f:
        #     json.dump(eval_data, f, indent=2)
    print(f"total modified count: {cnt}")


def remove_completion_ending_stop_token(data):
    new_completions = []
    for completion in data["completions"]:
        for stop in data["stop_tokens"]:
            if completion.endswith(stop):
                completion = completion[:-len(stop)]
                break
        new_completions.append(completion)
    return new_completions

def modify_codegen2_16B_dump(dump_dir):
    for lang_pair in tqdm(os.listdir(dump_dir)):
        if "-" not in lang_pair:
            continue
        for exp in os.listdir(f"{dump_dir}/{lang_pair}"):
            exp_dir = f"{dump_dir}/{lang_pair}/{exp}"
            modify_files_remove_stop_tokens(exp_dir)


def modify_files_remove_stop_tokens(data_dir):
    """
    BAM IBM endpoint returns stop token w/ completion if model is stopped by stop token
    Here we will remove them from completion, if the last few tokens is a stop sequence
    """

    cnt = 0
    for path in tqdm(itertools.chain(Path(data_dir).glob("*.json"), Path(data_dir).glob("*.results.json.gz"))):
        if ".results.json" in str(path):
            continue

        with open_json(path, "r") as f:
            data = json.load(f)

        new_completions = remove_completion_ending_stop_token(data)
        if any([new_c != c for new_c, c in zip(new_completions, data["completions"])]):
            data["completions"] = new_completions
            cnt += 1
            if os.path.exists(str(path).replace(".json", ".results.json")):
                os.remove(str(path).replace(".json",  ".results.json"))
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    print(f"In {data_dir}, total modified count: {cnt}")


def delete_completion_files_conditional(data_dir):
    cnt = 0
    for path in tqdm(itertools.chain(Path(data_dir).glob("*.json"), Path(data_dir).glob("*.results.json.gz"))):
        if ".results.json" in str(path):
            continue
        with open_json(path, "r") as f:
            data = json.load(f)
        eval_path = str(path).replace(".json", ".results.json")
        try:
            with open_json(Path(eval_path), "r") as f:
                eval_data = json.load(f)
        except:
            eval_data = {}
        # if "1. Define the function signature\n2.".lower() not in data["translation_prompt"].lower():
        eval_exp = get_source_code_from_prompt(data["translation_prompt"])
        new_eval_exp = remove_lang_from_explanation(eval_exp, data["language"])
        if len(data["completions"]) != 20 or eval_exp != new_eval_exp:
            os.remove(path)
            if os.path.exists(eval_path):
                os.remove(eval_path)
            print(data["name"])
            cnt += 1
    print(f"total of {cnt} programs deleted")


def remove_target_info_from_completion_files(data_dir):
    cnt = 0
    for path in tqdm(itertools.chain(Path(data_dir).glob("*.json"), Path(data_dir).glob("*.results.json.gz"))):
        if ".results.json" in str(path):
            continue
        with open_json(path, "r") as f:
            data = json.load(f)
        eval_path = str(path).replace(".json", ".results.json")
        try:
            with open_json(Path(eval_path), "r") as f:
                eval_data = json.load(f)
        except:
            eval_data = {}
        eval_exp = get_source_code_from_prompt(data["translation_prompt"])
        new_eval_exp = remove_lang_from_explanation(eval_exp, data["language"])
        if eval_exp != new_eval_exp:
            # remove explanation from completion file prompt, delete existing completions
            data["completions"] = []
            data["translation_prompt"] = data["translation_prompt"].replace(eval_exp, new_eval_exp)
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            # remove existing evaluation files if any
            if os.path.exists(eval_path):
                os.remove(eval_path)
            print(data["name"])
            cnt += 1
    print(f"total of {cnt} programs deleted")


def delete_transcoder_dump_not_in_intersection(tests_dir, data_dir):
    common_ids = get_common_ids(tests_dir)
    cnt = 0
    for f in os.listdir(data_dir):
        if f.split(".")[0] not in common_ids:
            print(f'deleting {f}')
            os.remove(f"{data_dir}/{f}")
            cnt += 1
    print(f"total of {cnt} items deleted")


def get_common_ids(tests_dir):
    cpp_ids = [f.split(".")[0] for f in os.listdir(f"{tests_dir}/cpp")]
    py_ids = [f.split(".")[0] for f in os.listdir(f"{tests_dir}/python")]
    common_ids = set(py_ids).intersection(cpp_ids)
    return common_ids


def get_cpp(prompt: str):
    start_str = "### C++ version"
    start = prompt.find(start_str)
    end = prompt.find("###", start+1)
    return prompt[start+len(start_str):end].strip()


def delete_transcoder_changed_source(tests_dir, tok_dir, data_dir):
    common_ids = get_common_ids(tests_dir)
    cpp_golds = {}
    for split in ["valid", "test"]:
        cpp_golds.update({l.split(" | ")[0]:" | ".join(l.split(" | ")[1:])
                          for l in open(f"{tok_dir}/transcoder_{split}.cpp.tok")
                          if l.split(" | ")[0] in common_ids})
    cpp_processor = LangProcessor.processors["cpp"](root_folder=TREE_SITTER_ROOT)
    cpp_golds = {k:cpp_processor.detokenize_code(v) for k, v in cpp_golds.items()}

    cnt = 0
    for f in os.listdir(data_dir):
        if f.endswith("results.json") or f.endswith(".py"):
            continue
        data = json.load(open(f"{data_dir}/{f}"))
        cpp_code = get_cpp(data["translation_prompt"])
        if cpp_code.strip() != cpp_golds[data["name"]].strip():
            os.remove(f"{data_dir}/{f}")
            print(f"removed {data['name']}")
            cnt += 1

    print(f"total {cnt} files removed")


def restore_to_unique_only_prompt_and_completion(input_dir):
    for f in os.listdir(input_dir):
        if f.endswith("results.json"):
            pass
        f_path = f"{input_dir}/{f}"
        data = json.load(open(f_path))
        unique_prompts = set([p for p in data["translation_prompt"] if "TOFILL" not in p])
        unique_prompts_cnt = len(unique_prompts)
        default_prompt = data["translation_prompt"][-1]
        for i, prompt in enumerate(data["translation_prompt"]):
            if prompt in unique_prompts:
                unique_prompts.remove(prompt)
            else:
                data["translation_prompt"][i] = default_prompt
        unique_completions = set(data["completions"])
        data["completions"] = data["completions"][:min(unique_prompts_cnt, len(unique_completions))]
        json.dump(data, open(f_path, "w"), indent=2)


def update_codegen_dir(old_dir: str, new_dir: str):
    # if there is result file, copy result file along with completion file over,
    # otherwise delete them from old dir
    delete_cnt, move_cnt = 0,0
    for lang_pair in tqdm(os.listdir(old_dir)):
        if lang_pair.startswith("."):
            continue
        for exp in os.listdir(f"{old_dir}/{lang_pair}"):
            if exp.startswith("."):
                continue
            for f in os.listdir(f"{old_dir}/{lang_pair}/{exp}"):
                if f.endswith(".results.json") or f.startswith("."):
                    continue
                old_path = f"{old_dir}/{lang_pair}/{exp}/{f}"
                old_eval_path = old_path.replace(".json", ".results.json")
                if not os.path.exists(f"{new_dir}/{lang_pair}/{exp}"):
                    os.mkdir(f"{new_dir}/{lang_pair}/{exp}")
                    print(f"new experiment directory created: {new_dir}/{lang_pair}/{exp}")
                new_path = f"{new_dir}/{lang_pair}/{exp}/{f}"
                new_eval_path = new_path.replace(".json", ".results.json")
                try:
                    with open(old_path) as f_in:
                        data = json.load(f_in)
                except:
                    print(f"error parsing {old_path}")
                    data = {"completions": []}
                # if completion is empty, delete old files
                if not os.path.isfile(old_eval_path) or len(data["completions"]) < 20:
                    os.remove(old_path)
                    delete_cnt += 1
                else:
                    shutil.copy(old_path, new_path)
                    shutil.copy(old_eval_path, new_eval_path)
                    move_cnt += 1
    print(f"deleted {delete_cnt} old empty completion files")
    print(f"moved {move_cnt} to new directory")

    # modify new dir (remove extra stop tokens things)
    modify_codegen2_16B_dump(new_dir)


if __name__ == "__main__":
    # modify_files(f"../dump_codegen216b/py-js/humaneval-py-js-PTremove-completion")
    # modify_codegen2_16B_dump(f"../dump_llamacode2instruct")
    update_codegen_dir(old_dir=f"../dump_codegen216b", new_dir="../dump")
    # delete_completion_files_conditional("../dump/py-jl/humaneval-py-jl-PTremove-MTexplain-completion")
    # delete_transcoder_dump_not_in_intersection(
    #     tests_dir="../../CodeGenMirror/data/transcoder_evaluation_gfg_fixed",
    #     data_dir="../dump/cpp-py/transcoder_fixed_eval-cpp-py-TSkeep-completion"
    # )
    # delete_transcoder_changed_source(
    #     tests_dir="../../CodeGenMirror/data/transcoder_evaluation_gfg",
    #     tok_dir="../../decompose-and-translate/data/transcoder_test_set",
    #     data_dir="../dump/cpp-py/transcoder_eval-cpp-py-TStyped-MTexplain-completion"
    # )
    # restore_to_unique_only_prompt_and_completion("../dump/py-java/humaneval-py-java-PTremove-MTexplain20-completion")