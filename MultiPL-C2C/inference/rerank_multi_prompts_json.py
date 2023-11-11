"""
This file takes in a prompt.json file, ranks intermediate steps with different methods, then output
in dump dir as individual completion / prediction files, this file also contains
- function that gathers dump file into reranked prompt.json file
- function that estimate performance of reranked prompt.json file (only if success_rates were given in
    original prompt.json file
"""
import argparse
import collections
import json
import os
import pdb
import string
import asyncio
import regex as re
from copy import deepcopy
from pathlib import Path

import numpy as np
from typing import List, Dict

from tqdm import tqdm

# from codegen_sources.model.src.utils import TREE_SITTER_ROOT
# from codegen_sources.preprocessing.lang_processors.lang_processor import LangProcessor
from dataset_builder.utils import get_source_lang_from_prompt, get_source_code_from_prompt, get_target_lang_from_prompt
from inference.gather_completions import main
from inference.rerank_utils import get_coder_reviewer_score, Scorer, get_coder_score, get_reviewer_score

MAX_TO_GENERATE=512 # limitation on how long the rerank/explanation can be
alphabet = string.ascii_uppercase
ONE_SHOT_CHOICE = """You are an helpful AI assistant who understands all programming languages and can translate between them at ease. You are given a Python program and several potential explanations of the program, but only one of them is correct. You have to pick which one is correct.

### Python version
 
from typing import List

def words_string(s: str) -> List[str]:
    if not s:
        return []

    s_list = []

    for letter in s:
        if letter == ',':
            s_list.append(' ')
        else:
            s_list.append(letter)

    s_list = "".join(s_list)
    return s_list.split()

### Potential Explanations

A. This function takes a string as input and returns a list of words. It does so by first creating an empty list called `s_list`. Then it iterates over each character in the input string. If the character is a comma, it appends a space to `s_list`. Otherwise, it appends the character itself. Finally, it joins all the characters in `s_list` into a single string, and splits that string into a list of words using whitespace as the delimiter. 

B. This program takes a string as input and returns a list of the words in the string. It does this by first converting all commas in the string to spaces, and then splitting the string into a list of words.

C. This program takes a string as input and returns a list of words. The input string is converted to a list of characters. If a comma is found, it is replaced with a space. The list of characters is then joined back into a string and split into a list of words. If the input string is empty, an empty list is returned.

D. This program takes a string as input and returns a list of words in the string. The function `words_string` first checks if the input string is empty, if it is, it returns an empty list. If it is not empty, the function creates an empty list `s_list`. The function then loops through each character in the input string. If the character is a comma, it is replaced with a space and added to the `s_list`. If the character is not a comma, it is added to the `s_list` as is. The function then joins all the characters in the `s_list` to create a string and splits the string into a list of words using the `split()` method. The list of words is then returned.

### Correct Explanation

The correct explanation is B. This program takes a string as input and returns a list of the words in the string. It does this by first converting all commas in the string to spaces, and then splitting the string into a list of words.

"""
# scorer = Scorer("distilgpt2")
# scorer = Scorer("Salesforce/codegen2-1B")
# scorer = Scorer("Salesforce/codegen2-16B")

def get_args():
    args = argparse.ArgumentParser()
    args.add_argument("--prompts-file", type=str, required=True, help="File of prompts")
    args.add_argument(
        "--target-dir",
        type=str,
        required=True,
        help="Directory to write completions to",
    )
    args.add_argument("--temperature", type=float, default=0.2)
    args.add_argument("--max-samples", type=int, default=0.1)
    args.add_argument("--model", type=str, required=True)
    args.add_argument("--limit-completions", type=int, default=200)
    args.add_argument("--completion_type", type=str, default="choice_argmax",
                      help="options are: choice_argmax(_0shot), choice_sample, generation, score")
    args.add_argument("--log-file", type=str, default=None)
    args.add_argument("--log-level", type=str, default="INFO")
    args.add_argument(
        "--local-model",
        action="store_true",
        help="If set, --model is the name of a model file to load",
    )
    args = args.parse_args()
    args.process_function = completion_rerank
    return args


def format_rerank_prompt(problem, args):
    inter_steps = [get_source_code_from_prompt(p).strip() for p in problem["translation_prompt"]]
    steps_str = "\n\n".join([f"{alphabet[i]}. {s}" for i, s in enumerate(inter_steps)]) + "\n\n"
    src_lang = get_source_lang_from_prompt(problem["translation_prompt"][0])
    src_code = get_source_code_from_prompt(problem["translation_prompt"][0], -3)
    instruction = "You are an helpful AI assistant who understands all programming languages and can translate between" \
             f" them at ease. You are given a Python program and several potential explanations of the program, but" \
             f" only one of them is correct. You have to pick which one is correct.\n\n"
    prompt = f"### {src_lang} version\n\n" \
             f"{src_code}" \
             f"### Potential Explanations\n\n" \
             f"{steps_str}" \
             f"### Correct Explanation\n\n" \
             f""
    if args.completion_type.startswith("choice"):
        prompt += f"The correct explanation is"
        stops = [".", "because", "since", "the"]
        if "1shot" in args.completion_type:
            prompt = ONE_SHOT_CHOICE + prompt
        else:
            prompt = instruction + prompt
    elif args.completion_type == "generation":
        instruction = "You are an helpful AI assistant who understands all programming languages and can translate between" \
                      f" them at ease. You are given a Python program and several potential explanations of the program and " \
                      f"you need to pick the best (or generate a new) explanation that would allow a reader to rewrite the " \
                      f"Python program in Java.\n\n"
        stops = ["\n#"]
        prompt = instruction + prompt
    else:
        raise NotImplementedError

    return prompt, stops


def format_score_prompt(problem, translation_prompt, args):
    src_lang = get_source_lang_from_prompt(problem["translation_prompt"][0])
    tgt_lang = get_target_lang_from_prompt(problem["translation_prompt"][0])
    src_code = get_source_code_from_prompt(problem["translation_prompt"][0], -3)
    stops = ["\n#", "."]
    if "da" in args.completion_type:
        prompt = f"""You are an helpful AI assistant who understands all programming languages and can translate between them at ease. Score the following code explanation given the corresponding {src_lang} code with respect to accuracy, completeness, and helpfulness on a continuous scale from 0 to 100, where a score of zero means "translated ({tgt_lang}) program would not do anything close to source ({src_lang}) program" and score of one hundred means "translated ({tgt_lang}) program would do exactly what the source ({tgt_lang}) program do and can pass the same unit tests". Note that accuracy measures the quality of individual statements the explanation, do they explain the {src_lang} statements correctly, completeness measures how much of the {src_lang} program is explained in the explanation, and helpfulness measures how likely the explanation can be used to rewrite the {src_lang} program in {tgt_lang}.

### {src_lang} version

{src_code.strip()}

### Explanation

{get_source_code_from_prompt(translation_prompt).strip()}

### Score

total averaged score: """
    else:
        raise NotImplementedError
    return prompt, stops


def clean_up_completion(completion: str) -> str:
    """ remove any sentence that mentions choosing any letter options. Output sentences
     should only be the explanation themselves and not about the meta-review of the explanation """
    sents = completion.split(". ")
    filt_sents = []
    for sent in sents:
        if any((f" {letter} " in sent) or (sent.endswith(f" {letter}") or sent == letter or (f" {letter}," in sent))  for letter in alphabet):
            continue
        filt_sents.append(sent)
    clean_completion = ". ".join(filt_sents).replace("<|im_end|>","")
    return clean_completion


def simulate_success_rate(problems: List[Dict]) -> float:
    success_cnt = 0
    for problem in problems:
        if np.random.rand() <= problem["success_rate"]:
            success_cnt += 1
    return success_cnt / len(problems)


def simulate_random_success_rate(problems: List[Dict]) -> float:
    success_cnt = 0
    for problem in problems:
        if np.random.rand() <= np.random.choice(problem["success_rates"]):
            success_cnt += 1
    return success_cnt / len(problems)


def upperbound_success_rate(problems: List[Dict]) -> float:
    success_cnt = 0
    for problem in problems:
        success_rate = np.max(problem["success_rates"])
        if success_rate>0:
            success_cnt += 1
    return success_cnt / len(problems)


async def completion_rerank(completion, problem, args, max_to_generate):
    target_dir_path = Path(args.target_dir)
    completions_path = target_dir_path / (problem["name"] + ".json")

    target_dir_path.mkdir(exist_ok=True)

    if completions_path.exists():
        with completions_path.open() as f:
            completion_results = json.load(f)
    else:
        # Copy problem to completion_results
        completion_results = problem.copy()
        completion_results["completions"] = []

    # completion_results has the same keys as problem, and one extra key "completions".
    if "score" in args.completion_type:
        num_completions_required = len(problem["translation_prompt"]) - len(completion_results["completions"])
    else:
        num_completions_required = args.limit_completions - len(completion_results["completions"])

    if num_completions_required < 1:
        return

    while num_completions_required > 0:
        if "score" not in args.completion_type:
            num_samples = min(num_completions_required, args.max_samples)
            prompt, stops = format_rerank_prompt(problem, args)
            _, completions = await completion(
                model=args.model,
                prompt=prompt,
                max_tokens=max_to_generate,
                temperature=args.temperature,
                n=num_samples,
                top_p=0.95,
                stop=stops,
            )
        else:
            num_samples = 1
            translation_prompt = problem["translation_prompt"][len(completion_results["completions"])]
            prompt, stops = format_score_prompt(problem, translation_prompt, args)
            _, single_score_completions = await completion(
                model=args.model,
                prompt=prompt,
                max_tokens=max_to_generate,
                temperature=args.temperature,
                n=args.max_samples,
                top_p=0.95,
                stop=stops,
            )
            try:
                completions = [np.mean([float(c.strip().split()[0].replace("**","")) for c in single_score_completions])]
            except:
                pdb.set_trace()
                completions = []
                num_samples = 0

        completion_results["completions"].extend(completions)
        with completions_path.open("w") as f:
            f.write(json.dumps(completion_results, indent=2))
        num_completions_required -= num_samples


def gather_dumps(args):
    print(f"gathering dumps from {args.target_dir}")
    problems = []
    for f in os.listdir(args.target_dir):
        res = json.load(open(f"{args.target_dir}/{f}"))
        assert isinstance(res["translation_prompt"], list), \
            "reranking completion results require translation prompt to be of type list"
        problem = deepcopy(res)
        del problem["completions"]
        if args.completion_type.startswith("choice") or args.completion_type.startswith("score"):
            if args.completion_type.startswith("choice"):
                options = [alphabet.index(c.strip().split()[0].upper().replace("**","")) for c in res["completions"]]
                option_counts = collections.Counter(options)
            else:
                option_counts = collections.Counter({k: v for k, v in enumerate(res["completions"])})
            if "argmax" in args.completion_type:
                best_index = option_counts.most_common(1)[0][0]
            elif "sample" in args.completion_type:
                p = np.array([*option_counts.values()])
                p = p / p.sum()
                best_index = np.random.choice(list(option_counts.keys()), 1, p=p)[0]
            else:
                raise NotImplementedError
            translation_prompt = problem["translation_prompt"][best_index]
            problem["success_rate"] = problem["success_rates"][best_index]
        elif args.completion_type.startswith("generation"):
            if len(res["completions"]) > 1:
                print("Found multiple completion results for generation reranking. Only taking first generation!")
            translation_prompt = clean_up_completion(res["completions"][0])
        else:
            raise NotImplementedError
        problem["translation_prompt"] = translation_prompt
        problems.append(problem)

    # calculate success rate if success rate of each intermediate prompt is given
    if args.completion_type.startswith("choice") or args.completion_type.startswith("score"):
        non_trivial_problems = set([p["name"] for p in problems if sum(p["success_rates"])/len(p["success_rates"]) not in [0.0, 1.0]])
        if "1shot" in args.completion_type:
            non_trivial_problems.remove("HumanEval_101_words_string")
        # calculate average success rate of prompts amongst non-trivial selections
        num_trials = 20
        random_success_rates = [simulate_random_success_rate(problems) for _ in range(num_trials)]
        mean_success_rate = np.mean([p["success_rate"] for p in problems if p["name"] in non_trivial_problems])
        random_mean_success_rates = [np.mean([np.random.choice(p["success_rates"]) for p in problems
                                            if p["name"] in non_trivial_problems]) for _ in range(num_trials)]

        # calculate average rank amongst non-trivial selections
        mean_rank = np.mean([sorted(p["success_rates"])[::-1].index(p["success_rate"]) for p in problems
                             if p["name"] in non_trivial_problems])
        random_mean_ranks = [np.mean([sorted(p["success_rates"])[::-1].index(np.random.choice(p["success_rates"])) for p in problems
                             if p["name"] in non_trivial_problems]) for _ in range(num_trials)]

        print(f"upperbound success rate = {upperbound_success_rate(problems):.3f}")
        print(f"Simulated success rate = {simulate_success_rate(problems):.3f} (random = {np.mean(random_success_rates)} +- {np.std(random_success_rates)}")
        print(f"Total of {len(non_trivial_problems)} non-trivial problems:")
        print(f"mean_success_rate in non-trivial problems = {mean_success_rate:.3f} "
              f"(random = {np.mean(random_mean_success_rates):.3f} +- {np.std(random_mean_success_rates):.3f})")
        print(f"mean_rank in non-trivial problems = {mean_rank:.3f} "
              f"(random = {np.mean(random_mean_ranks):.3f} +- {np.std(random_mean_ranks):.3f})")

    # out_path = args.prompts_file.replace(".json", f"_RR{args.completion_type}.json")
    # json.dump(problems, open(out_path, "w"), indent=2)


def process_heuristic_total_length(problem: Dict):
    exps = [get_source_code_from_prompt(p) for p in problem["translation_prompt"]]
    best_idx = np.argmax([len(p) for p in exps])
    problem["translation_prompt"] = problem["translation_prompt"][best_idx]
    problem["success_rate"] = problem["success_rates"][best_idx]


def process_heuristic_total_length_no_code(problem: Dict):
    exps = [get_source_code_from_prompt(p) for p in problem["translation_prompt"]]
    exps = [re.sub(r"```[^`]*```", "", p) for p in exps]
    best_idx = np.argmax([len(p) for p in exps])
    problem["translation_prompt"] = problem["translation_prompt"][best_idx]
    problem["success_rate"] = problem["success_rates"][best_idx]


def process_heuristic_lines_code_explained(problem: Dict):
    src_code = get_source_code_from_prompt(problem["translation_prompt"][0], -3)
    src_code_lines = [l.strip() for l in src_code.split("\n")]
    exps = [get_source_code_from_prompt(p) for p in problem["translation_prompt"]]
    best_idx = np.argmax([len([l for l in p.split("\n") if any([l in cl for cl in src_code_lines])]) for p in exps])
    problem["translation_prompt"] = problem["translation_prompt"][best_idx]
    problem["success_rate"] = problem["success_rates"][best_idx]


def process_heuristic_code_count_fragments(problem: Dict):
    def sample_with_temperature(weights, t=1):
        weights = np.array(weights) / t
        weights = np.exp(weights)/sum(np.exp(weights))
        return np.random.choice(len(weights), p=weights)

    exps = [get_source_code_from_prompt(p) for p in problem["translation_prompt"]]
    num_frags = [len(re.findall(r"`[^`]*`", p)) for p in exps]
    best_idx = np.argmax(num_frags)
    # best_idx=sample_with_temperature(num_frags, 1)  # does not work as well
    problem["translation_prompt"] = problem["translation_prompt"][best_idx]
    problem["success_rate"] = problem["success_rates"][best_idx]


def process_heuristic_code_count_fragments_plus(problem: Dict):
    COMMON_TOKENS={"if", "else","then", "end","for","while",".",",",":",";"}
    exps = [get_source_code_from_prompt(p) for p in problem["translation_prompt"]]
    src_code = get_source_code_from_prompt(problem["translation_prompt"][0], -3)
    # src_code_toks = set(src_code.split()).difference(COMMON_TOKENS)
    src_code_toks = set(LangProcessor.processors["python"](TREE_SITTER_ROOT).tokenize_code(src_code)).difference(COMMON_TOKENS)
    num_frags = []
    for exp in exps:
        num_frag = len(re.findall(r"`[^`]*`", exp))
        exp = re.sub(r"`[^`]*`", "", exp)
        # num_frag += len([t for t in exp.split() if t.lower() in src_code.lower() and t.lower() not in COMMON_TOKENS])
        # no_quote_frags = len(src_code_toks.intersection(exp.split()))
        no_quote_frags = sum([exp.count(f" {t} ") for t in src_code_toks])
        # print(f"quote_frags: {num_frag}, not_quoted_frags: {no_quote_frags}")
        num_frag += no_quote_frags
        num_frags.append(num_frag)
    best_idx = np.argmax(num_frags)
    problem["translation_prompt"] = problem["translation_prompt"][best_idx]
    problem["success_rate"] = problem["success_rates"][best_idx]


def process_heuristic_num_lines(problem: Dict):
    exps = [get_source_code_from_prompt(p) for p in problem["translation_prompt"]]
    num_lines = [p.count("\n") for p in exps]
    best_idx = np.argmax(num_lines)
    problem["translation_prompt"] = problem["translation_prompt"][best_idx]
    problem["success_rate"] = problem["success_rates"][best_idx]


def process_heuristic_coder_reviewer(problem: Dict, shots: int=0, alpha=0.5):
    # cache_score_key = f"coder_reviewer_{scorer.model_name.split('/')[-1]}_{shots}shot"
    cache_score_key = f"coder_reviewer_codegen2-16B_{shots}shot"
    if cache_score_key in problem:
        # scores = problem[cache_score_key]
        scores = [c*alpha + r*(1-alpha) for c,r in zip(problem[f"coder_codegen2-16B_{shots}shot"], problem[f"reviewer_codegen2-16B_{shots}shot"])]
    else:
        exps = [get_source_code_from_prompt(p) for p in problem["translation_prompt"]]
        src_code = get_source_code_from_prompt(problem["translation_prompt"][0], -3)
        # DEFAULT_FEW_SHOT = Path(__file__).absolute().parents[1].joinpath("few_shot_prompts","java","py-java_translate_MTexplain_coder_reviewer.txt")
        DEFAULT_FEW_SHOT = Path(__file__).absolute().parents[1].joinpath("few_shot_prompts", "java",
                                                                         "py-java_translate_MTexplain-lbl-simp_coder_reviewer.txt")
        coder_score = get_coder_score(src_code, exps, scorer, few_shot_file=str(DEFAULT_FEW_SHOT), shots=shots)
        reviewer_score = get_reviewer_score(src_code, exps, scorer, few_shot_file=str(DEFAULT_FEW_SHOT), shots=shots)
        scores = [c + r for c,r in zip(coder_score, reviewer_score)]
        problem[cache_score_key.replace("coder_reviewer", "coder")] = coder_score
        problem[cache_score_key.replace("coder_reviewer", "reviewer")] = reviewer_score
        problem[cache_score_key] = scores
    best_idx = np.argmax(scores)
    # problem["translation_prompt"] = problem["translation_prompt"][best_idx]
    problem["success_rate"] = problem["success_rates"][best_idx]


def heuristics_select(args):
    print(f"reading directly from {args.prompts_file}")
    with open(args.prompts_file) as f:
        problems = json.load(f)
    # alpha=0.5
    for alpha in np.linspace(0, 1, 11):
        # process each problem, select with heuristics function
        for problem in tqdm(problems, desc="processing problems"):
            type_args = args.completion_type.split('-')
            process_func = globals().get(f"process_{type_args[0]}")
            if process_func is None:
                raise NotImplementedError
            kwargs = {type_args[i]: int(type_args[i+1]) if type_args[i+1].isdigit() else type_args[i+1] for i in range(1,len(type_args)-1,2)}
            # process_func(problem, **kwargs)
            process_func(problem, **kwargs, alpha=alpha)

        # calculate success rate if success rate of each intermediate prompt is given
        non_trivial_problems = set([p["name"] for p in problems if sum(p["success_rates"])/len(p["success_rates"]) not in [0.0, 1.0]])

        # calculate average success rate of prompts amongst non-trivial selections
        num_trials = 100
        random_success_rates = [simulate_random_success_rate(problems) for _ in range(num_trials)]
        success_rates = [simulate_success_rate(problems) for _ in range(num_trials)]

        mean_success_rate = np.mean([p["success_rate"] for p in problems if p["name"] in non_trivial_problems])
        random_mean_success_rates = [np.mean([np.random.choice(p["success_rates"]) for p in problems
                                            if p["name"] in non_trivial_problems]) for _ in range(num_trials)]

        # calculate average rank amongst non-trivial selections
        mean_rank = np.mean([sorted(p["success_rates"])[::-1].index(p["success_rate"]) for p in problems
                             if p["name"] in non_trivial_problems])
        random_mean_ranks = [np.mean([sorted(p["success_rates"])[::-1].index(np.random.choice(p["success_rates"])) for p in problems
                             if p["name"] in non_trivial_problems]) for _ in range(num_trials)]

        print(f"========== heuristic method: {args.completion_type} alpha={alpha} ==========")
        print(f"upperbound success rate = {upperbound_success_rate(problems):.3f}")
        print(f"Simulated success rate = {np.mean(success_rates):.3f} +- {np.std(success_rates):.3f} (random = {np.mean(random_success_rates):.3f} +- {np.std(random_success_rates):.3f})")
        print(f"Total of {len(non_trivial_problems)} non-trivial problems:")
        print(f"mean_success_rate in non-trivial problems = {mean_success_rate:.3f} "
              f"(random = {np.mean(random_mean_success_rates):.3f} +- {np.std(random_mean_success_rates):.3f})")
        print(f"mean_rank in non-trivial problems = {mean_rank:.3f} "
              f"(random = {np.mean(random_mean_ranks):.3f} +- {np.std(random_mean_ranks):.3f})")

    scorer_name = "codegen2-16B"  # scorer.model_name.split('/')[-1]
    # out_path = args.prompts_file.replace(".json", f"_RR{args.completion_type}_{scorer_name}.json")
    # json.dump(problems, open(out_path, "w"), indent=2)


def reselect_coder_reviewer_with_alpha(src_path, tgt_path, alpha=0.5, shots=2):
    """when doing alpha sweep, coder reviewer dump file does not contain all translation prompts
    here we restore the prompts from the source file, and reselect based on alpha value"""
    src_problems = {p["name"]:p for p in json.load(open(src_path))}
    tgt_problems = {p["name"]:p for p in json.load(open(tgt_path))}
    for name, problem in tgt_problems.items():
        best_idx = np.argmax([c*alpha + r*(1-alpha) for c,r in zip(problem[f"coder_codegen2-16B_{shots}shot"], problem[f"reviewer_codegen2-16B_{shots}shot"])])
        problem["translation_prompt"] = src_problems[name]["translation_prompt"][best_idx]
        problem["success_rate"] = src_problems[name]["success_rates"][best_idx]

    num_trials = 100
    random_success_rates = [simulate_random_success_rate(tgt_problems.values()) for _ in range(num_trials)]
    success_rates = [simulate_success_rate(tgt_problems.values()) for _ in range(num_trials)]
    print(
        f"Simulated success rate = {np.mean(success_rates):.3f} +- {np.std(success_rates):.3f} (random = {np.mean(random_success_rates):.3f} +- {np.std(random_success_rates):.3f})")
    out_path = tgt_path.replace(".json", f"_alpha{alpha}.json")
    json.dump(list(tgt_problems.values()), open(out_path, "w"), indent=2)


if __name__ == "__main__":
    args = get_args()
    if "heuristic" not in args.completion_type:
        asyncio.run(main(args))
        gather_dumps(args)
    else:
        heuristics_select(args)

    # LOCAL DEBUG ONLY
    # reselect_coder_reviewer_with_alpha(
    #     "../translation_prompts/py-java/humaneval-py-java-PTremove-MTexplain-lbl-simp20RR-4shot-completion-agg20.json",
    #     '../translation_prompts/py-java/humaneval-py-java-PTremove-MTexplain-lbl-simp20RR-4shot-completion-agg20_RRheuristic_coder_reviewer-shots-0_codegen2-16B.json',
    #     alpha=0.7,
    #     shots=0
    # )

