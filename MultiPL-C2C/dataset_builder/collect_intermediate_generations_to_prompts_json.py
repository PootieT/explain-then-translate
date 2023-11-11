
import os
import json
from typing import *

from analysis.analyze_result_folder import read_folder, get_problem_success_rate
from dataset_builder.utils import get_source_code_from_prompt


def clean_prompt(translation_prompt: str):
    return translation_prompt.replace("This  ", "This ")


def combine_multiple_explanations(translation_prompts: List[str]) -> str:
    prompt = translation_prompts[-1]
    prompt = prompt.replace("Can you explain", "Can you explain in a few different ways")\
             .replace("The goal with the explanation", "The goal with the explanations")\
             .replace("### Explanation", "### Explanations")
    explanations = [get_source_code_from_prompt(p).strip() for p in translation_prompts]
    prompt = prompt.replace(explanations[-1], "\n\n".join(explanations)).replace("\n\n\n", "\n\n")
    prompt = clean_prompt(prompt)
    return prompt


def collect(dump_dirs: List[str], output_path: str, combine_to_one: bool=False):
    """
    This file gathers lists of dump folders (with intermediate steps) and generate
    a single prompt.json file such that it can then be re-ranked or just used to
    in round-robin fashion to generate with multiple intermediate states.
    """
    dfs = [read_folder(d) for d in dump_dirs]
    max_beam = min([len(df.results[0]) for df in dfs])
    copy_fields = ["language", "prompt", "doctests", "original", "prompt_terminology", "source_program","target_signature","tests","stop_tokens"]
    problems = []
    for name in dfs[0].index:
        problem = {"name": name}
        problem.update({k: dfs[0].loc[name][k] for k in copy_fields})
        translation_prompts, success_rates = [], []
        if combine_to_one:
            translation_prompts = combine_multiple_explanations([df.loc[name].translation_prompt for df in dfs])
        else:
            for df in dfs:
                if isinstance(df.loc[name].translation_prompt, str):
                    translation_prompts.append(clean_prompt(df.loc[name].translation_prompt))
                    success_rates.append(get_problem_success_rate(df, name, max_beam))
                elif isinstance(df.loc[name].translation_prompt, list):
                    for i, prompt in enumerate(df.loc[name].translation_prompt):
                        translation_prompts.append(prompt)
                        completion_indices = list(range(i, len(df.loc[name].results), len(df.loc[name].translation_prompt)))
                        success_rates.append(get_problem_success_rate(df, name, completion_indices))
        problem.update({
            "translation_prompt": translation_prompts,
            "success_rates": success_rates
        })
        problems.append(problem)

    print(f"aggregated {len(problems)} problems, {len(dump_dirs)} input files with {max_beam} generations each")
    json.dump(problems, open(output_path, "w"), indent=2)


if __name__ == "__main__":
    collect(
        dump_dirs=[
            "../dump/py-rkt/humaneval-py-rkt-PTremove-MTexplain20RR-completion",
            # "../dump/py-java/humaneval-py-java-PTremove-MTexplain-lbl-4shot-completion_trial2",
            # "../dump/py-java/humaneval-py-java-PTremove-MTexplain-lbl-4shot-completion_trial3",
            # "../dump/py-java/humaneval-py-java-PTremove-MTexplain-lbl-4shot-completion_trial4",
        ],
        output_path="../translation_prompts/py-rkt/humaneval-py-rkt-PTremove-MTexplain20RR-completion-agg20.json",
        combine_to_one=False
    )
