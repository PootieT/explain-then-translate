import os
import json
from tqdm import tqdm
EXPS=[
    "humaneval-{pair}-PTremove-completion.json",
    "humaneval-{pair}-PTremove-MTexplain-completion.json",
    "humaneval-{pair}-PTremove-MTexplain-lbl-completion.json",
    "humaneval-{pair}-PTremove-MTexplain-lbl-simp-completion.json"
]
PROMPT_ROOT="../translation_prompts/translation_prompts_chatgpt"
TGT_ROOT="../translation_prompts/translation_prompts_chatgpt_with_explanation"
DUMP_ROOT="../../../MultiPL-EX/dump_chatgpt"


def main():
    for pair in tqdm(os.listdir("../translation_prompts/translation_prompts_chatgpt")):
        for exp in EXPS:
            exp = exp.format(pair=pair)
            prompt_path = f"{PROMPT_ROOT}/{pair}/{exp}"
            prompts = json.load(open(prompt_path))
            prompts_dict = {p["name"]: p for p in prompts}
            dump_dir = f"{DUMP_ROOT}/{pair}/{exp}".replace(".json", "")
            for prob in os.listdir(dump_dir):
                problem = json.load(open(f"{dump_dir}/{prob}"))
                if problem["name"] in prompts_dict:
                    prompts_dict[problem["name"]]["translation_prompt"] = problem["translation_prompt"]
                else:
                    prompts_dict[problem["name"]] = problem
                    del prompts_dict[problem["name"]]["completions"]
            prompts_modified = list(prompts_dict.values())
            tgt_path = f"{TGT_ROOT}/{pair}/{exp}"
            os.makedirs(f"{TGT_ROOT}/{pair}", exist_ok=True)
            with open(tgt_path, "w") as f:
                json.dump(prompts_modified, f, indent=2)


if __name__ == "__main__":
    main()