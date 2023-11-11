import os


def refactor_files(input_dir):
    for f in os.listdir(input_dir):
        cur_path = f"{input_dir}/{f}"
        if "python" in f:
            new_path = cur_path.replace("python", "py")
            os.rename(cur_path, new_path)
            print(f"renamed {cur_path} to {new_path}")


def refactor_exp_name(src_name, tgt_name):
    cnt = 0
    for lang in "js cpp ts php rb cs go pl r rs scala swift sh lua rkt jl d".split():
        # refactor translation prompt json names
        # src_path = f"../translation_prompts/py-{lang}/humaneval-py-{lang}-{src_name}.json"
        # tgt_path = f"../translation_prompts/py-{lang}/humaneval-py-{lang}-{tgt_name}.json"
        # os.rename(src_path, tgt_path)

        # refactor dump directory names

        src_path = f"../dump/py-{lang}/humaneval-py-{lang}-{src_name}"
        tgt_path = f"../dump/py-{lang}/humaneval-py-{lang}-{tgt_name}"
        if os.path.isdir(src_path):
            os.rename(src_path, tgt_path)
            cnt += 1

    print(f"modified {cnt} dirs to {tgt_name}")


if __name__ == "__main__":
    # refactor_files("../dump/py-java")
    refactor_exp_name(
        "PTremove-MTexplain-4shot-completion-coder-reviewer-manually-remove-lua",
        "PTremove-MTexplain-4shot-completion-coder-reviewer"
    )
    refactor_exp_name(
        "PTremove-MTexplain-lbl-simp-4shot-completion-java-no-heuristic-manually-remove-lua",
        "PTremove-MTexplain-lbl-simp-4shot-completion-java-no-heuristic"
    )