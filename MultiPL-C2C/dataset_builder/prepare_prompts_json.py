"""
This script prepares all prompts for a particular language as YAML files (one
per benchmark). The scripts completions_*.py will then update each file with
completions from an LLM.

To run this script:

1. mkdir ../datasets/LANGUAGE-keep-MODEL

  where MODEL is either davinci or incoder.

2. python3 prepare_prompts_yaml.py --lang LANGUAGE --target-dir ../datasets/LANGUAGE-keep-MODEL --doctests keep

  This will create lots of YAML files in TARGET-DIR. You should commit these files to the repository.

3. Now run either completions_codex.py or completions_incoder.py.


Estimate of how big each YAML file gets:

- length of prompt + completion = 2048 tokens
- Each token is ~4 characters
- 200 samples per prompt
- (2048 * 4 * 200) / 1024 / 1024 = 1.5 MB of data.

This ignores the tests cases, but it should be compact enough.

"""

import argparse
import os.path
import pdb
import sys
from pprint import pprint

import pandas as pd
from dataset_builder.utils import bool_flag, get_multi_vew_acronym, \
    get_gold_src_programs, cap, SHORT2CANONICAL, FEW_SHOT_EXAMPLES, ablate_few_shot
from dataset_builder.generic_translator import modify_translation_prompt
from generic_translator import list_originals, translate_prompt_and_tests, get_stop_from_translator
from pathlib import Path
import json

def get_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--lang", type=str, required=False, default="humaneval_to_java.py", help="Language to translate to"
    )
    args.add_argument(
        "--src-lang", type=str, required=False, default="py", help="Language to translate from"
    )
    args.add_argument(
        "--output", type=str, required=False, default="../translation_prompts", help="Target directory containing output JSON file"
    )
    args.add_argument(
        "--few-shot-file", type=str, required=False, default=None, help="few shot JSON file"
    )
    args.add_argument(
        "--shots", type=int, required=False, default=1,
        help="how many shots to use"
    )
    args.add_argument(
        "--ablate_few_shot", type=int, required=False, default=0,
        help="how many few-shot examples to swap (with a randomly sampled wrong program)"
    )
    args.add_argument(
        "--ablate_few_shot_false_only", type=bool_flag, default=True,
        help="if sampling few-shot ablation program from false programs or all unverified programs"
    )


    args.add_argument(
        "--doctests",
        type=str,
        default="keep",
        help="What to do with doctests: keep, remove, or transform",
    )

    args.add_argument(
        "--prompt-terminology",
        type=str,
        default="verbatim",
        help="How to translate terminology in prompts: verbatim, reworded, or removed"
    )

    args.add_argument(
        "--source-program",
        type=str,
        default="remove",
        help="What to do with source program: remove (NL-code task), keep (code-code task)"
    )

    args.add_argument(
        "--source-prompt",
        type=bool_flag,
        default=True,
        help="Whether to keep the prompt in the source program. Treatment with prompt wording and doctest will "
             "be the same as prompt-terminology and doctests"
    )

    args.add_argument(
        "--target-signature",
        type=str,
        default="keep",
        help="What to do with target program signature: remove (harder, not fully defined), keep (easier)"
    )
    args.add_argument(
        "--multiturn-prompt",
        type=str,
        default="single",
        help="what type of multiturn prompt to generate: single (single turn), steps (generate NL steps, then "
             "translate), steps-cot (generate NL steps, translate each step w/ chain of thoughts provided w/ signature,"
             "combine w/ signature."
    )
    args.add_argument(
        "--multiturn-template-count",
        type=str,
        default=1,
        help="If prompt is multiturn, the intermediate steps are also generated. This parameter controls how many of "
             "those intermediate steps are to be generated, and stored as prompt templates. During inference step, if "
             "multiple prompts are presented, the generation will take turn taking the mod of template count when "
             "selecting the prompt"
    )
    args.add_argument(
        "--prompt-type",
        type=str,
        default="completion",
        help="completion or chat (completion)"
    )
    args.add_argument(
        "--multi-view-dirs",
        type=str,
        default=None,
        help="comma separated list of directories of existing output to obatin multi-views from"
    )
    args.add_argument("--originals", type=str, default="../datasets/originals-with-cleaned-doctests")

    args = args.parse_args()
    args.multi_view_dirs = args.multi_view_dirs.split(",") if args.multi_view_dirs is not None else args.multi_view_dirs
    return args

def main(args):
    translator = __import__(args.lang[:-3]).Translator()
    src_translator = __import__(f"humaneval_to_{args.src_lang}").Translator()

    if args.prompt_terminology not in ["verbatim", "reworded", "remove"]:
        print(f"Invalid prompt-terminology option: {args.prompt_terminology}")
        sys.exit(1)

    if args.doctests not in ["keep", "remove", "transform"]:
        print(f"Unknown doctests option: {args.doctests}")
        sys.exit(1)

    if args.source_program not in ["remove", "keep"]:
        print(f"Unknown source program option: {args.source_program}")
        sys.exit(1)

    if args.target_signature not in ["keep", "remove"]:
        print(f"Unknown target signature option: {args.target_signature}")
        sys.exit(1)

    if args.few_shot_file is not None:
        if args.few_shot_file.endswith(".csv"):
            few_shot_data = pd.read_csv(args.few_shot_file).set_index("TARGET_CLASS")
        elif args.few_shot_file.endswith(".jsonl"):
            few_shot_data = [json.loads(l) for l in open(args.few_shot_file)]
        else:
            assert args.few_shot_file.endswith(".txt") and args.prompt_type == "completion"
            few_shot_data = open(args.few_shot_file).read()

    else:
        few_shot_data, args.few_shot_file = [], ""
    src_gold_programs = get_gold_src_programs(args.src_lang, args.originals, src_translator) if args.src_lang != "py" else {}

    os.makedirs(Path(args.output).parent, exist_ok=True)

    results = [ ]
    for original in list_originals(args.originals).values():
        original_name = original.name.split(".")[0]
        print(f"Processing {original_name}...")

        few_shot = few_shot_data if isinstance(few_shot_data, list) or isinstance(few_shot_data, str) else \
            few_shot_data.loc[original_name] if original_name in few_shot_data.index else []
        if args.multi_view_dirs is not None:
            if any([not os.path.isdir(d) for d in args.multi_view_dirs]):
                raise FileNotFoundError(f"Multiview directory not found: {pprint(args.multi_view_dirs)}")
            multi_view_files = [f"{d}/{original_name}.results.json" for d in args.multi_view_dirs if os.path.isfile(f"{d}/{original_name}.json")]
        else:
            multi_view_files = None
        result = translate_prompt_and_tests(
            original, translator, args.doctests, args.prompt_terminology, args.source_program, args.target_signature,
            args.source_prompt,
            [] if "MT" in args.few_shot_file else few_shot,
            args.shots,
            args.prompt_type,
            args.src_lang,
            src_gold_programs.get(original_name),
            obfuscate=args.multiturn_prompt.endswith("-obf")
        )
        if result is None:
            print(f"Skipping {original_name} because problem didn't transpile")
            continue

        (prompt, tests) = result
        problem = {
            "name": original_name,
            "language": translator.file_ext(),
            "prompt": prompt,
            "doctests": args.doctests,
            "original": str(original.absolute()),
            "prompt_terminology": args.prompt_terminology,
            "source_program": args.source_program,
            "target_signature": args.target_signature,
            "tests": tests,
            "stop_tokens": get_stop_from_translator(translator),
        }
        problem = modify_translation_prompt(problem, args.multiturn_prompt.replace("-obf", ""),
                                            few_shot if "MT" in args.few_shot_file else [], args.shots,
                                            src_lang=cap(SHORT2CANONICAL[args.src_lang]),
                                            multi_view_files=multi_view_files,
                                            multiturn_template_count=args.multiturn_template_count)

        if problem is None or (problem["name"] in FEW_SHOT_EXAMPLES[:args.shots] and args.few_shot_file.endswith(".txt")):
            print(f"skipping {original_name} because of translation prompt modification")
            continue

        results.append(problem)

    if args.ablate_few_shot:
        assert args.ablate_few_shot <= args.shots
        results = ablate_few_shot(translator, results, args.ablate_few_shot,
                                  is_retrieval=args.few_shot_file.endswith(".csv"),
                                  false_only=args.ablate_few_shot_false_only)

    print(f"Total of {len(results)} problems generated")
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)


def modify_output_path(args):
    lang = args.lang.split("_")[-1].replace(".py", "")
    ds_name = args.originals.split('/')[-1].replace('originals', 'humaneval')
    if args.prompt_terminology != "remove":
        args.output = f"{args.output}/{args.src_lang}-{lang}/{ds_name}-{args.src_lang}-{lang}-PT{args.prompt_terminology}-DT{args.doctests}-SP{args.source_prompt}"
    else:
        args.output = f"{args.output}/{args.src_lang}-{lang}/{ds_name}-{args.src_lang}-{lang}-PT{args.prompt_terminology}"
    if args.multiturn_prompt != "single":
        args.output += f"-MT{args.multiturn_prompt}"
        if args.multiturn_prompt == "multi-view":
            args.output += "_" + get_multi_vew_acronym(args.multi_view_dirs)
        if args.multiturn_template_count > 1:
            args.output += f"{args.multiturn_template_count}RR"
    if args.few_shot_file is not None and args.shots > 0:
        args.output += f"-{args.shots}shot"
        if args.few_shot_file.endswith(".csv"):
            args.output += "(retrieval)"
    if args.prompt_type == "completion":
        args.output += f"-completion"
    if args.ablate_few_shot:
        args.output += f"-ALT{'wrong' if args.ablate_few_shot_false_only else 'unverified'}-fs-{args.ablate_few_shot}"
    args.output += ".json"


if __name__ == "__main__":
    # commandline usage
    args = get_args()
    modify_output_path(args)
    main(args)

    # debug usage
    # for lang in ["d"]:#"js cpp ts php rb cs go pl r rs scala swift sh lua rkt jl d".split():  #"js cpp ts php rb cs go pl r rs scala swift sh lua rkt jl d"
    #     args = get_args()
    #     args.src_lang = "py"
    #     # lang = "jl"
    #     args.lang = f"humaneval_to_{lang}.py"
    #     args.prompt_terminology = "remove"
    #     args.doctests = "keep"
    #     args.source_prompt = True  # TODO
    #     args.source_program = "keep"
    #     args.target_signature = "keep"
    #     args.prompt_type = "completion"
    #     args.originals = "../datasets/originals"
        #
        # args.shots = 0
        # args.multiturn_prompt = "single"
        # few_shot_mt_str = '_MT' + args.multiturn_prompt if args.shots > 0 and args.multiturn_prompt != "single" else ""
        # args.few_shot_file = f"../few_shot_prompts/{lang}/{args.src_lang}-{lang}_translate{few_shot_mt_str}.txt"
        # modify_output_path(args)
        # main(args)

        # args.shots = 4
        # args.multiturn_prompt = "single"
        # few_shot_mt_str = '_MT' + args.multiturn_prompt if args.shots > 0 and args.multiturn_prompt != "single" else ""
        # args.few_shot_file = f"../few_shot_prompts/{lang}/{args.src_lang}-{lang}_translate{few_shot_mt_str}.txt"
        # modify_output_path(args)
        # main(args)
        #
        # args.shots = 0
        # args.multiturn_prompt = "explain"
        # few_shot_mt_str = '_MT' + args.multiturn_prompt if args.shots > 0 and args.multiturn_prompt != "single" else ""
        # args.few_shot_file = f"../few_shot_prompts/{lang}/{args.src_lang}-{lang}_translate{few_shot_mt_str}.txt"
        # args.multi_view_dirs = [
        #     f"../dump/{args.src_lang}-java/humaneval-{args.src_lang}-java-PTremove-MT{args.multiturn_prompt}-completion-manually-remove-java",
        #     # f"../dump/{args.src_lang}-java/humaneval-{args.src_lang}-rkt-PTremove-MT{args.multiturn_prompt}-completion-manually-remove-java"
        # ]
        # modify_output_path(args)
        # main(args)
        # #
        # args.shots = 0
        # args.multiturn_prompt = "explain-lbl"
        # few_shot_mt_str = '_MT' + args.multiturn_prompt if args.shots > 0 and args.multiturn_prompt != "single" else ""
        # args.few_shot_file = f"../few_shot_prompts/{lang}/{args.src_lang}-{lang}_translate{few_shot_mt_str}.txt"
        # args.multi_view_dirs = [
        #     f"../dump/{args.src_lang}-java/humaneval-{args.src_lang}-java-PTremove-MT{args.multiturn_prompt}-completion-manually-remove-java"
        # ]
        # modify_output_path(args)
        # main(args)
        # # #
        # args.shots = 0
        # args.multiturn_prompt = "explain-lbl-simp"
        # few_shot_mt_str = '_MT' + args.multiturn_prompt if args.shots > 0 and args.multiturn_prompt != "single" else ""
        # args.few_shot_file = f"../few_shot_prompts/{lang}/{args.src_lang}-{lang}_translate{few_shot_mt_str}.txt"
        # # args.multiturn_template_count = 20
        # args.multi_view_dirs = [
        #     f"../dump/{args.src_lang}-java/humaneval-{args.src_lang}-java-PTremove-MT{args.multiturn_prompt}-completion-manually-remove-java"
        # ]
        # modify_output_path(args)
        # main(args)

        #
        # 0 shot explain, append to 4 shot translation (w/ explain)
        # args.shots = 4
        # args.multiturn_prompt = "explain"
        # few_shot_mt_str = '_MT' + args.multiturn_prompt if args.shots > 0 and args.multiturn_prompt != "single" else ""
        # args.few_shot_file = f"../few_shot_prompts/{lang}/{args.src_lang}-{lang}_translate{few_shot_mt_str}.txt"
        # args.multi_view_dirs = [
        #    # f"../dump/{args.src_lang}-java/humaneval-{args.src_lang}-java-PTremove-MT{args.multiturn_prompt}-completion-manually-remove-java"
        #     f"../dump/{args.src_lang}-{lang}/humaneval-{args.src_lang}-{lang}-PTremove-MT{args.multiturn_prompt}-completion"
        # ]
        # modify_output_path(args)
        # main(args)
        #
        # 4 shot explain
        # args.shots = 4
        # args.multiturn_prompt = "explain"
        # few_shot_mt_str = '_MT' + args.multiturn_prompt if args.shots > 0 and args.multiturn_prompt != "single" else ""
        # args.few_shot_file = f"../few_shot_prompts/{lang}/{args.src_lang}-{lang}_translate{few_shot_mt_str}.txt"
        # args.multi_view_dirs = None
        # args.multi_view_dirs = [
        #     # f"../dump/{args.src_lang}-java/humaneval-{args.src_lang}-java-PTremove-MT{args.multiturn_prompt}-4shot-completion"
        #     f"../dump/{args.src_lang}-java/humaneval-py-java-PTremove-MTexplain20RR-completion-agg20_RRheuristic_coder_reviewer-shots-1_codegen2-16B_alpha0.7",
        # ]
        # modify_output_path(args)
        # main(args)
        # #
        # args.shots = 4
        # args.multiturn_prompt = "explain-lbl"
        # few_shot_mt_str = '_MT' + args.multiturn_prompt if args.shots > 0 and args.multiturn_prompt != "single" else ""
        # args.few_shot_file = f"../few_shot_prompts/{lang}/{args.src_lang}-{lang}_translate{few_shot_mt_str}.txt"
        # # args.multiturn_template_count = 20
        # args.multi_view_dirs = [
        #     f"../dump/{args.src_lang}-js/humaneval-{args.src_lang}-js-PTremove-MT{args.multiturn_prompt}-4shot-completion"
        # ]
        # # args.multi_view_dirs=None
        # modify_output_path(args)
        # main(args)
        # #
        # args.shots = 4
        # args.multiturn_prompt = "explain-lbl-simp"
        # few_shot_mt_str = '_MT' + args.multiturn_prompt if args.shots > 0 and args.multiturn_prompt != "single" else ""
        # args.few_shot_file = f"../few_shot_prompts/{lang}/{args.src_lang}-{lang}_translate{few_shot_mt_str}.txt"
        # # args.multiturn_template_count = 20
        # args.multi_view_dirs = [
        #     # f"../dump/{args.src_lang}-java/humaneval-{args.src_lang}-java-PTremove-MT{args.multiturn_prompt}20RR-4shot-completion-agg20_RRheuristic_code_count_fragments",
        #     f"../dump/{args.src_lang}-java/humaneval-{args.src_lang}-java-PTremove-MT{args.multiturn_prompt}-4shot-completion",
        #     # f"../dump/{args.src_lang}-js/humaneval-{args.src_lang}-js-PTremove-MT{args.multiturn_prompt}-4shot-completion"
        # ]
        # # args.multi_view_dirs = None
        # modify_output_path(args)
        # main(args)


        ########### ablation trials (obfusecate code) #############
        # args.shots = 0
        # for exp in ["single", "explain", "explain-lbl", "explain-lbl-simp"]:
        #     args.multiturn_prompt = f"{exp}-obf"
        #     few_shot_mt_str = '_MT' + args.multiturn_prompt if args.shots > 0 and args.multiturn_prompt != "single" else ""
        #     args.few_shot_file = f"../few_shot_prompts/{lang}/{args.src_lang}-{lang}_translate{few_shot_mt_str}.txt"
        #     ### for non-java, re-use java exp
        #     args.multi_view_dirs = [
        #         f"../dump/{args.src_lang}-java/humaneval-{args.src_lang}-java-PTremove-MT{args.multiturn_prompt}-completion{'-manually-remove-java' if 'explain' in args.multiturn_prompt else ''}",
        #     ]
        #     modify_output_path(args)
        #     main(args)

        ############# pivot ablation trials ##############
        # for pivot_lang in ["java", "php", "rkt"]:
        #     if pivot_lang == lang:
        #         continue
        #     for correctness in ["gold", "wrong"]:
        #         print(f"==== pivot experiment {args.src_lang}-{pivot_lang}-{lang}, correctness={correctness} ====")
        #         args.shots = 0
        #         args.multiturn_prompt = f"pivot-{correctness}-{pivot_lang}"
        #         few_shot_mt_str = '_MT' + args.multiturn_prompt if args.shots > 0 and args.multiturn_prompt != "single" else ""
        #         args.few_shot_file = f"../few_shot_prompts/{lang}/{args.src_lang}-{lang}_translate{few_shot_mt_str}.txt"
        #         modify_output_path(args)
        #         main(args)

        ############# few-shot ablation trials ##############
        # print(f"==== fixed 1-shot ablation experiment {args.src_lang}--{lang}, correctness=unverified ====")
        # args.shots = 1
        # args.ablate_few_shot = 1
        # args.ablate_few_shot_false_only = False
        # few_shot_mt_str = '_MT' + args.multiturn_prompt if args.shots > 0 and args.multiturn_prompt != "single" else ""
        # args.few_shot_file = f"../few_shot_prompts/{lang}/{args.src_lang}-{lang}_translate{few_shot_mt_str}.txt"
        # modify_output_path(args)
        # main(args)
        #
        # print(f"==== fixed 1-shot ablation experiment {args.src_lang}--{lang}, correctness=wrong ====")
        # args.shots = 1
        # args.ablate_few_shot = 1
        # args.ablate_few_shot_false_only = True
        # args.multiturn_prompt = "single"
        # few_shot_mt_str = '_MT' + args.multiturn_prompt if args.shots > 0 and args.multiturn_prompt != "single" else ""
        # args.few_shot_file = f"../few_shot_prompts/{lang}/{args.src_lang}-{lang}_translate{few_shot_mt_str}.txt"
        # modify_output_path(args)
        # main(args)
        #
        # print(f"==== retrieval 1-shot ablation experiment {args.src_lang}--{lang}, correctness=gold ====")
        # args.shots = 1
        # args.ablate_few_shot = 0
        # args.few_shot_file = f"../few_shot_prompts/{lang}/{args.src_lang}-{lang}_gold_bm25_top4.csv"
        # modify_output_path(args)
        # main(args)
        #
        # print(f"==== retrieval 1-shot ablation experiment {args.src_lang}--{lang}, correctness=unverified ====")
        # args.shots = 1
        # args.ablate_few_shot = 1
        # args.ablate_few_shot_false_only = False
        # args.few_shot_file = f"../few_shot_prompts/{lang}/{args.src_lang}-{lang}_gold_bm25_top4.csv"
        # modify_output_path(args)
        # main(args)
        #
        # print(f"==== retrieval 1-shot ablation experiment {args.src_lang}--{lang}, correctness=wrong ====")
        # args.shots = 1
        # args.ablate_few_shot = 1
        # args.ablate_few_shot_false_only = True
        # args.few_shot_file = f"../few_shot_prompts/{lang}/{args.src_lang}-{lang}_gold_bm25_top4.csv"
        # modify_output_path(args)
        # main(args)
