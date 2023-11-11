import itertools
import json
import os.path
from pathlib import Path
from typing import *

import scipy
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from dataset_builder.utils import SHORT2CANONICAL, cap, CANONICAL2SHORT
from src.single_experiment_error_types import understand_errors, get_all_data, get_result_breakdown_by_len, ERROR_TYPES, \
    get_src_program_len
from src.single_experiment_pass_k import for_file

RESULT_PATH = "../translation_results/Code Translation Prompting - MultiPLE (completion).csv"
PYTHON_X_EXPS = {
    "baseline (0 shot)": "direct",
    "explain -> translation (java)": "exp",
    "explain-lbl -> translation": "exp-lbl",
    "explain-lbl-simp -> translation": "exp-lbl-d",
    "baseline (4 shot)": "direct",
    # "explain -> translation (4 shot) (java)": "exp",  # python-X experiments
    "explain (0 shot) -> translation (4 shot) (coder-reviewer)": "exp",  # python-X experiments (heuristics)
    "explain-lbl -> translation (4 shot)": "exp-lbl",
    # "explain-lbl-simp -> translation (4 shot) (no heuristic)": "exp-lbl-d",  # python-X experiments
    "explain-lbl-simp -> translation (4 shot) (code frag)": "exp-lbl-d",  # python-X experiments (heuristics)
}

X_X_EXPS = {
    "baseline (0 shot)": "direct",
    "explain -> translation": "exp",
    "explain-lbl -> translation": "exp-lbl",
    "explain-lbl-simp -> translation": "exp-lbl-d",
    "baseline (4 shot)": "direct",
    "explain (0 shot) -> translation (4 shot)": "exp",
    "explain-lbl -> translation (4 shot)": "exp-lbl",
    "explain-lbl-simp -> translation (4 shot) (code frag)": "exp-lbl-d",  # for python-x direction (for some reason no heuristic not found)
    "explain-lbl-simp -> translation (4 shot)": "exp-lbl-d",
}

exp2path={
    "direct": "",
    "exp": "-MTexplain",
    "exp-lbl": "-MTexplain-lbl",
    "exp-lbl-d": "-MTexplain-lbl-simp"
}
res2lang={
    "high": ["JavaScript", "C++", "Java", "TypeScript"],
    "medium": ["PHP", "Ruby", "C#", "Go"],
    "low": ["Perl", "R", "Rust", "Scala", "Swift"],
    "extremely low": ["Bash", "Lua", "Racket", "Julia", "D"]
}
xxres2lang={
    "High-High": [["Python","JavaScript"],["JavaScript","Java"],["C++","Python"],["Java","C++"]],
    "High-Extremely Low": [["JavaScript","Racket"],["Python","D"],["C++","Lua"],["Java","Julia"]],
    "Extremely Low-High": [["Lua","Python"],["Racket","Java"],["Julia","JavaScript"],["D","C++"]],
    "Extremely Low-Extremely Low": [["Lua","Racket"],["Racket","Julia"],["D","Lua"],["Julia","D"]]
}
xxres2langstr = {res: [f"{p[0]}-{p[1]}" for p in pairs] for res, pairs in xxres2lang.items()}

res2bar_loc={
    "high": 1.5,
    "medium": 1.5,
    "low": 1.5,
    "extremely low": 2.5
}
short2print = {k:cap(v) for k,v in SHORT2CANONICAL.items()}
PALETTE = "vlag"


def snake_to_cap(s):
    return " ".join([cap(sub) for sub in s.split("_")])

def r2(x, y):
    return scipy.stats.pearsonr(x, y)[0] ** 2

def get_dump_dir(row: pd.Series) -> str:
    shot_str = "-4shot" if row.shots == 4 else ""
    src = CANONICAL2SHORT[row.src_lang.lower()]
    tgt = CANONICAL2SHORT[row.tgt_lang.lower()]
    dump_path = f"../dump_chatgpt/{src}-{tgt}/humaneval-{src}-{tgt}-PTremove" \
                f"{exp2path[row.exp]}{shot_str}-completion"
    # TODO, still there are two trials py-X in X-X where we should use java-no-heuristic tag
    if row.exp == "exp-lbl-d" and row.shots == 4:
        dump_path += "-frag" if src == "py" else "-java-no-heuristic"
        if src=="py" and tgt=="java":
            dump_path = "/".join(dump_path.split("/")[:-1] + ["humaneval-py-java-PTremove-MTexplain-lbl-simp20RR-4shot-completion-agg20_RRheuristic_code_count_fragments"])
    return dump_path


def cleanup_table(pass_fillin=True, fill_incomplete=True, experiment_subsets: Union[str, Dict[str,str]]="python_x") -> pd.DataFrame:
    df = pd.read_csv(RESULT_PATH, skiprows=1)
    # discard irrelevant roles for better visuals in google sheet
    df.columns = ["exp", *df.columns[1:]]
    df = df[~df.src_lang.isna()]
    df['Pass'].fillna(0, inplace=True)
    df.exp = df.exp.str.strip()
    df.src_lang.replace(short2print, inplace=True)
    df.tgt_lang.replace(short2print, inplace=True)
    for err in ERROR_TYPES:
        df[err] = df[err].fillna(0)

    if experiment_subsets == "python_x":
        df = df[df.src_lang == "Python"]
    experiment_subsets = globals()[f"{experiment_subsets.upper()}_EXPS"] if isinstance(experiment_subsets, str) else experiment_subsets
    df = df[df.exp.isin(experiment_subsets)]
    df.exp.replace(experiment_subsets, inplace=True)

    if pass_fillin:
        df.loc[df['Pass@1(n=20)'].isna(), 'Pass@1(n=20)'] = df[df['Pass@1(n=20)'].isna()].Pass
    if fill_incomplete:
        df['Pass@1(n=20)'] = df['Pass@1(n=20)'].fillna(0)
    else:
        df = df[~df['Pass@1(n=20)'].isna()]
    return df


def update_error_breakdown_and_length_breakdown(df, buckets=4, len_type="char"):
    for i, row in tqdm(df.iterrows(), total=len(df)):
        dump_path = get_dump_dir(row)
        if not os.path.exists(dump_path):
            print(f"dump path does not exist: {dump_path}")
            continue
        try:
            result_array = np.array([understand_errors(p) for p in get_all_data(dump_path)]).squeeze()
            result_by_len = get_result_breakdown_by_len(result_array, get_all_data(dump_path), buckets=buckets, len_type=len_type)
            errors = result_array.mean(axis=0)

            df.loc[i, ERROR_TYPES] = errors
            len_bins = get_bucket_col_names(buckets)
            df.loc[i, len_bins] = result_by_len
        except Exception as e:
            print(f"exception parsing dump: {dump_path}")
            continue
    return df


def combine_error_arrays(err: np.array):
    semantic_err = err[:, [3,5]].sum(-1)
    syntactic_err = err[:, [0,1,2,4]].sum(-1)
    combined_err = np.vstack([semantic_err, syntactic_err]).T
    return add_pass_rate(combined_err)


def add_pass_rate(err: np.array):
    return np.concatenate([np.expand_dims(1-err.sum(-1), 1), err], 1)


def get_error_conversion_trial(exp_err: np.array, base_err: np.array) -> np.array:
    error_types = exp_err.shape[1]
    res = np.zeros([error_types, error_types])
    for i in range(error_types):  # source error type
        for j in range(error_types):  # target error type
            res[i, j] = (base_err[:, i] * exp_err[:, j]).mean()
    return res


def get_error_conversion(df: pd.DataFrame, combine_errors: False) -> Tuple[pd.DataFrame, np.array]:
    df = df[df.exp!="direct"]
    error_conversions = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        dump_path = get_dump_dir(row)
        baseline_row = row.copy()
        baseline_row["exp"] = "direct"
        baseline_path = get_dump_dir(baseline_row)
        for p in [dump_path, baseline_path]:
            if not os.path.exists(p):
                print(f"dump path does not exist: {p}")
                exit()
        exp_res = np.array([understand_errors(p) for p in get_all_data(dump_path)]).squeeze()
        base_res = np.array([understand_errors(p) for p in get_all_data(baseline_path)]).squeeze()
        if combine_errors:
            exp_res, base_res = combine_error_arrays(exp_res), combine_error_arrays(base_res)
        else:
            exp_res, base_res = add_pass_rate(exp_res), add_pass_rate(base_res)
        try:
            error_conversions.append(get_error_conversion_trial(exp_res, base_res)*100)
        except Exception as e:
            print(f"exception while getting error conversion: {e}")
            print(f"exp_dir: {dump_path}")
            continue
    return df, np.array(error_conversions)


def get_bucket_col_names(buckets):
    return [str(int(p * 100)) for p in np.linspace(0, 1, buckets + 1)[1:]]


def plot_python_x(df):
    df = df.copy()
    fig, axes = plt.subplots(2, 4, sharex=False, sharey=False, figsize=(14,7))
    # plt.setp(axes, ylim=(0,1))
    # fig.suptitle('Pass@1 for Python-X translation (Top: 0-shot, Bottom: 4-shot)')
    # fig.text(0.4, 0.04, "Target Language", ha="center")
    axes[0,0].set_title('High')
    axes[0,1].set_title('Medium')
    axes[0,2].set_title('Low')
    axes[0,3].set_title('Extremely Low')
    df = df[df.src_lang == "Python"]

    for row, shot in enumerate([0, 4]):
        for col, res in enumerate(["high", "medium", "low", "extremely low"]):
            sub_df = df[(df.tgt_lang.isin(res2lang[res])) & (df.shots==shot)]
            g = sns.barplot(
                data=sub_df, #kind="bar",
                x="tgt_lang", y='Pass@1(n=20)', hue="exp",
                errorbar="sd", palette=PALETTE, #alpha=.6, #height=6,
                ax=axes[row, col], order=res2lang[res], hue_order=["direct", "exp", "exp-lbl", "exp-lbl-d"]
            )
            # g.despine(left=True)
            g.set_xlabel("")
            if row == 0:
                g.set_xticks([])
            ylabel = f'{shot}-shot pass@1(n=20)' if col==0 else ""
            g.set_ylabel(ylabel)
            margin = 0.05
            non_zero_min = sub_df[sub_df['Pass@1(n=20)']>0]['Pass@1(n=20)'].min()-margin
            _max = sub_df['Pass@1(n=20)'].max()+margin
            g.set_ylim((non_zero_min,_max))
            g.get_legend().remove()
            g.axvline(x=res2bar_loc[res],
                      ymin=0, ymax=1, c="red", ls='--', linewidth=0.8, zorder=0, clip_on=False)
    # handles, labels = axes.get_legend_handles_labels()
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc=2, bbox_to_anchor=(1.02, 1), borderaxespad=0., frameon=False)
    fig.tight_layout()
    # plt.show()
    plt.savefig("../analysis/translation_python_x.png", bbox_inches="tight")
    plt.close()


def plot_x_x(df):
    df = df.copy()
    fig, axes = plt.subplots(2, 4, sharex=False, sharey=False, figsize=(14,7))
    # plt.setp(axes, ylim=(0,1))
    # fig.suptitle('Pass@1 for Python-X translation (Top: 0-shot, Bottom: 4-shot)')
    # fig.text(0.4, 0.04, "Target Language", ha="center")
    axes[0,0].set_title('High-High')
    axes[0,1].set_title('High-Extremely Low')
    axes[0,2].set_title('Extremely Low-High')
    axes[0,3].set_title('Extremely Low-Extremely Low')
    df["src_tgt"] = df.apply(lambda row: f"{row.src_lang}-{row.tgt_lang}", axis=1)
    for row, shot in enumerate([0, 4]):
        for col, res in enumerate(["High-High", "High-Extremely Low", "Extremely Low-High", "Extremely Low-Extremely Low"]):
            sub_df = df[(df.src_tgt.isin(xxres2langstr[res])) & (df.shots==shot)]
            g = sns.barplot(
                data=sub_df, #kind="bar",
                x="src_tgt", y='Pass@1(n=20)', hue="exp",
                errorbar="sd", palette=PALETTE, #alpha=.6, #height=6,
                ax=axes[row, col], order=xxres2langstr[res]
            )
            # g.despine(left=True)
            g.set_xlabel("")
            if row == 0:
                g.set_xticks([])
            axes[row, col].tick_params('x', labelrotation=30)
            ylabel = f'{shot}-shot pass@1(n=20)' if col==0 else ""
            g.set_ylabel(ylabel)
            margin = 0.05
            non_zero_min = sub_df[sub_df['Pass@1(n=20)']>0]['Pass@1(n=20)'].min()-margin
            _max = sub_df['Pass@1(n=20)'].max()+margin
            g.set_ylim((non_zero_min,_max))
            g.get_legend().remove()
    # handles, labels = axes.get_legend_handles_labels()
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc=2, bbox_to_anchor=(1.02, 1), borderaxespad=0., frameon=False)
    fig.tight_layout()
    # plt.show()
    plt.savefig("../analysis/translation_x_x.png", bbox_inches="tight")
    plt.close()


def plot_error_type_vs_explanation_type(df, combine_errors=True, pairs: Optional[List[str]]=None, shots=0):
    df = df.copy()
    if combine_errors:
        df["Semantic Error"] = df["assertion_error"] + df["unhelpful"]
        df["Syntactic Error"] = df["type_error"] + df["undeclared_error"] + df["other_syntax_error"]
        error_cols = ["Semantic Error", "Syntactic Error"]
    else:
        error_cols = [snake_to_cap(e) for e in ERROR_TYPES]
        df.rename(columns={e: snake_to_cap(e) for e in ERROR_TYPES}, inplace=True)
    if pairs is not None:
        df = df[df[["src_lang", "tgt_lang"]].isin(pairs)]

    df = df[df.shots==shots]

    melt_df = pd.melt(df, id_vars=['exp', "src_lang", "tgt_lang"], value_vars=error_cols,
                      var_name='Error Type', value_name='Error@1')

    sns.set_theme(style="ticks")
    plt.figure(figsize=(6, 6))
    g = sns.boxplot(x="Error Type", y="Error@1", hue="exp", data=melt_df,
                whis=[0, 100], width=.6, palette=PALETTE, hue_order=["direct", "exp", "exp-lbl", "exp-lbl-d"])
    g.get_legend().set_title("")
    if not combine_errors:
        plt.xticks(rotation=30)
    plt.tight_layout()

    plt.savefig(f"../analysis/err_type_vs_exp_type{'_detail' if not combine_errors else ''}_{shots}shot.png")
    plt.close()


def plot_error_type_vs_explanation_type_by_res(df, combine_errors=True, pairs: Optional[List[str]]=None):
    if combine_errors:
        df["Semantic Error"] = df["assertion_error"] + df["unhelpful"]
        df["Syntactic Error"] = df["type_error"] + df["undeclared_error"] + df["other_syntax_error"]
        error_cols = ["Semantic Error", "Syntactic Error"]
    else:
        error_cols = [snake_to_cap(e) for e in ERROR_TYPES]
        df.rename(columns={e: snake_to_cap(e) for e in ERROR_TYPES}, inplace=True)
    if pairs is not None:
        df = df[df[["src_lang", "tgt_lang"]].isin(pairs)]

    melt_df = pd.melt(df, id_vars=['exp', "src_lang", "tgt_lang", "shots"], value_vars=error_cols,
                      var_name='Error Type', value_name='Error@1')

    sns.set_theme(style="ticks")

    fig, axes = plt.subplots(2, 4, sharex=False, sharey=False, figsize=(12, 6) if combine_errors else (15, 8))
    # plt.setp(axes, ylim=(0,1))
    fig.suptitle('Different types of Error@1 in Python-X (Top: 0-shot, Bottom: 4-shot)')
    # fig.text(0.4, 0.04, "Target Language", ha="center")
    axes[0, 0].set_title('High')
    axes[0, 1].set_title('Medium')
    axes[0, 2].set_title('Low')
    axes[0, 3].set_title('Extremely Low')
    for row, shot in enumerate([0, 4]):
        for col, res in enumerate(["high", "medium", "low", "extremely low"]):
            sub_df = melt_df[(melt_df.tgt_lang.isin(res2lang[res])) & (melt_df.shots==shot)]
            g = sns.boxplot(x="Error Type", y="Error@1", hue="exp", data=sub_df,
                            whis=[0, 100], width=.6, palette=PALETTE,ax=axes[row, col],
                            hue_order=["direct", "exp", "exp-lbl", "exp-lbl-d"])

            # g.despine(left=True)
            g.set_xlabel("")
            ylabel = f'{shot}-shot error@1(n=20)' if col == 0 else ""
            g.set_ylabel(ylabel)
            margin = 0.05
            non_zero_min = sub_df[sub_df['Error@1'] > 0]['Error@1'].min() - margin
            _max = sub_df['Error@1'].max() + margin
            if not combine_errors:
                axes[row, col].tick_params('x',labelrotation=30)
            g.set_ylim((non_zero_min, _max))
            g.get_legend().remove()

    # handles, labels = axes.get_legend_handles_labels()
    handles, labels = plt.gca().get_legend_handles_labels()

    fig.legend(handles, labels, loc=2, bbox_to_anchor=(1.02, 1), borderaxespad=0., frameon=False)
    fig.tight_layout()
    plt.savefig(f"../analysis/err_type_vs_exp_type{'_detail' if not combine_errors else ''}_breakdown.png", bbox_inches="tight")
    plt.close()


def plot_len_vs_explanation_type_scatter(df, pairs: Optional[List[str]]=None, shots=0, len_type="char"):
    """
    result is sub-optimal, dots are very grid-y, probably better as binned barplot
    """
    if pairs is not None:
        df = df[(df.src_lang.str + df.tgt_lang.str).isin(pairs)]
    df = df[df.shots==shots]

    data = [[],[],[]]
    for i, row in tqdm(df.iterrows(), total=len(df)):
        dump_path = get_dump_dir(row)
        if not os.path.exists(dump_path):
            print(f"dump path does not exist: {dump_path}")
            continue
        else:
            try:
                success = np.array([for_file(p) for p in itertools.chain(Path(dump_path).glob("*.results.json"))])[:, 0]
                lens = [get_src_program_len(d, type="token") for d in get_all_data(dump_path)]
                data[0].extend(success)
                data[1].extend(lens)
                data[2].extend([row.exp]*len(success))
            except Exception as e:
                print(f"exception parsing dump: {dump_path}")
                continue

    len_df = pd.DataFrame({"Program Length": data[1], "Pass@1": data[0], "exp": data[2]})
    sns.set_theme(style="ticks")

    # Initialize the figure with a logarithmic x axis
    f, ax = plt.subplots(figsize=(6, 6))

    # Plot the orbital period with horizontal boxes
    g = sns.lmplot(
        data=len_df,
        x="Program Length", y="Pass@1", hue="exp",
        height=5
    )

    # Use more informative axis labels than are provided by default
    g.set_axis_labels(f"Program Length (# {len_type}s)", "Pass@1")
    # g.get_legend().set_title("")

    # Add in points to show each observation (too cluttered, doens't work well with hue)
    # sns.stripplot(x="Error Type", y="Error@1", hue="exp", data=melt_df,
    #               size=4, linewidth=0)  # color=".3"

    # Tweak the visual presentation
    plt.tight_layout()
    # ax.xaxis.grid(True)
    # ax.set(ylabel="")
    # sns.despine(trim=True, left=True)
    plt.savefig(f"../analysis/len_vs_exp_type_{shots}shot_{len_type}.png")




def plot_baseline_vs_explanation_pass_rate_scatter(df, pairs: Optional[List[str]]=None, shots=0, improve=False):
    """
    result is sub-optimal, dots are very grid-y, probably better as binned barplot
    """
    if pairs is not None:
        df = df[(df.src_lang.str + df.tgt_lang.str).isin(pairs)]
    df = df[df.shots==shots]

    data = {} # key: pair+name
    for i, row in tqdm(df.iterrows(), total=len(df)):
        dump_path = get_dump_dir(row)
        if not os.path.exists(dump_path):
            print(f"dump path does not exist: {dump_path}")
            continue
        else:
            try:
                success = np.array([for_file(p) for p in itertools.chain(Path(dump_path).glob("*.results.json"))])[:, 0]
                names = [json.load(open(p))["name"] for p in itertools.chain(Path(dump_path).glob("*.results.json"))]
                for i, (score, name) in enumerate(zip(success, names)):
                    key = f"{row.src_lang}-{row.tgt_lang}-{name}"
                    if key not in data:
                        data[key] = {row.exp: score}
                    else:
                        data[key][row.exp] = score
            except Exception as e:
                print(f"exception parsing dump: {dump_path}")
                continue

    res_df = []
    for k, v in data.items():
        for exp, score in v.items():
            if exp != "direct":
                res_df.append({"key": k, "exp": exp, "Exp Pass@1": score,
                               "Direct Pass@1": data[k]["direct"]})

    res_df = pd.DataFrame(res_df)
    res_df = res_df[res_df["Exp Pass@1"] != res_df["Direct Pass@1"]]
    res_df = res_df[res_df["exp"]=="exp"]
    sns.set_theme(style="ticks")

    # Initialize the figure with a logarithmic x axis
    f, ax = plt.subplots(figsize=(6, 6))
    if improve:
        # Plot the orbital period with horizontal boxes
        res_df["Exp Pass@1 > Direct Pass@1"] = res_df["Exp Pass@1"] > res_df["Direct Pass@1"]
        g = sns.histplot(
            data=res_df,
            x="Direct Pass@1", hue="Exp Pass@1 > Direct Pass@1"
        )
    else:
        g = sns.jointplot(data=res_df, x="Direct Pass@1", y="Exp Pass@1", kind="hist")# kind="kde" hue="exp",
        # Use more informative axis labels than are provided by default
        g.set_axis_labels(f"Direct Pass@1", "Exp Pass@1")
        # g.get_legend().set_title("")

    # Add in points to show each observation (too cluttered, doens't work well with hue)
    # sns.stripplot(x="Error Type", y="Error@1", hue="exp", data=melt_df,
    #               size=4, linewidth=0)  # color=".3"

    # Tweak the visual presentation
    plt.tight_layout()
    # ax.xaxis.grid(True)
    # ax.set(ylabel="")
    # sns.despine(trim=True, left=True)
    plt.savefig(f"../analysis/direct_vs_exp_scatter{'_improve' if improve else ''}_{shots}shot.png")


def plot_len_vs_explanation_type_bar(df, pairs: Optional[List[str]]=None, shots=0,len_type="char", buckets=4, by_res=False):
    df = df.copy()
    if pairs is not None:
        df = df[(df.src_lang.str+df.tgt_lang.str).isin(pairs)]
    if not by_res:
        df = df[df.shots==shots]
    df = update_error_breakdown_and_length_breakdown(df, buckets=buckets, len_type=len_type)

    melt_df = pd.melt(df, id_vars=['exp', "src_lang", "tgt_lang", "shots"], value_vars=get_bucket_col_names(buckets),
                      var_name='Program Length Percentile', value_name='Pass@1')

    sns.set_theme(style="ticks")

    if not by_res:
        f, ax = plt.subplots(figsize=(6, 6))
        g = sns.boxplot(x="Program Length Percentile", y="Pass@1", hue="exp", data=melt_df,
                        whis=[0, 100], width=.6, palette=PALETTE, hue_order=["direct", "exp", "exp-lbl", "exp-lbl-d"])
        g.get_legend().set_title("")
        plt.tight_layout()
        plt.savefig(f"../analysis/len_vs_exp_type_{shots}shot_{buckets}buckets_{len_type}.png")
    else:
        fig, axes = plt.subplots(2, 4, sharex=False, sharey=False, figsize=(12, 6))
        # plt.setp(axes, ylim=(0,1))
        fig.suptitle('Pass@1 over program of different lengths (Top: 0-shot, Bottom: 4-shot)')
        # fig.text(0.4, 0.04, "Target Language", ha="center")
        axes[0, 0].set_title('High')
        axes[0, 1].set_title('Medium')
        axes[0, 2].set_title('Low')
        axes[0, 3].set_title('Extremely Low')
        for row, shot in enumerate([0, 4]):
            for col, res in enumerate(["high", "medium", "low", "extremely low"]):
                sub_df = melt_df[(melt_df.tgt_lang.isin(res2lang[res])) & (melt_df.shots==shot)]
                g = sns.boxplot(x="Program Length Percentile", y="Pass@1", hue="exp", data=sub_df,
                                whis=[0, 100], width=.6, palette=PALETTE,ax=axes[row, col],
                                hue_order=["direct", "exp", "exp-lbl", "exp-lbl-d"])

                # g.despine(left=True)
                g.set_xlabel("")
                ylabel = f'{shot}-shot pass@1(n=20)' if col == 0 else ""
                g.set_ylabel(ylabel)
                margin = 0.05
                non_zero_min = sub_df[sub_df['Pass@1'] > 0]['Pass@1'].min() - margin
                _max = sub_df['Pass@1'].max() + margin
                g.set_ylim((non_zero_min, _max))
                g.get_legend().remove()

        # handles, labels = axes.get_legend_handles_labels()
        handles, labels = plt.gca().get_legend_handles_labels()
        fig.legend(handles, labels, loc=2, bbox_to_anchor=(1.02, 1), borderaxespad=0., frameon=False)
        fig.tight_layout()
        plt.savefig(f"../analysis/len_vs_exp_type_{buckets}buckets_{len_type}_breakdown.png", bbox_inches="tight")
    plt.close()


def plot_regression_4shot_vs_0shot_exp(df: pd.DataFrame):
    df_pivot = df.pivot(index=['src_lang', 'tgt_lang'], columns=['exp','shots'], values='Pass@1(n=20)')
    df_pivot.columns = [f"{exp} ({int(shot)} shot) Pass@1" for exp, shot in df_pivot.columns]
    df_pivot["exp (0 shot) - direct (0 shot) Pass@1"] = df_pivot["exp (0 shot) Pass@1"] - df_pivot["direct (0 shot) Pass@1"]
    df_pivot["direct (4 shot) - direct (0 shot) Pass@1"] = df_pivot["direct (4 shot) Pass@1"] - df_pivot["direct (0 shot) Pass@1"]
    sns.set_theme(style="ticks")

    plt.figure(figsize=(6, 6))
    ax = sns.regplot(x="exp (0 shot) - direct (0 shot) Pass@1", y="direct (4 shot) - direct (0 shot) Pass@1", data=df_pivot)
    # ax = sns.jointplot(x="exp (0 shot) Pass@1", y="direct (4 shot) Pass@1", data=df_pivot, kind="reg", stat_func=r2)
    for i, row in df_pivot.iterrows():
        ax.text(row["exp (0 shot) - direct (0 shot) Pass@1"], row["direct (4 shot) - direct (0 shot) Pass@1"]+ .005, i[1], fontdict={"size":10})
    r2_val = r2(df_pivot["exp (0 shot) - direct (0 shot) Pass@1"],df_pivot["direct (4 shot) - direct (0 shot) Pass@1"])
    df_pivot_no_outlier = df_pivot.iloc[[1,2,3,4,5,6,8,9,10,12,13,14,15,16,17]]
    r2_val_no_outlier = r2(df_pivot_no_outlier["exp (0 shot) - direct (0 shot) Pass@1"],df_pivot_no_outlier["direct (4 shot) - direct (0 shot) Pass@1"])
    # plt.title(f"4-shot vs. exp pass@1 improvement (r^2={r2_val:.3f})")
    plt.tight_layout()
    plt.savefig(f"../analysis/0shot_exp_vs_4shot_direct_regression.png")
    plt.close()


def plot_error_conversion(df: pd.DataFrame, combine_errors=True, split_exp=True):
    # for each pair of (baseline-exp) trials:
    # count where does each source error category go to second category
    # heatmap
    # aggregate over different directions
    # could breakdown over res and shots
    if combine_errors:
        error_cols = ["Pass", "Semantic Error", "Syntactic Error"]
    else:
        error_cols = ["Pass"] + [snake_to_cap(e) for e in ERROR_TYPES]
    new_df, error_conversions = get_error_conversion(df, combine_errors)
    if split_exp:
        fig, axes = plt.subplots(2, 3, sharex=False, sharey=False, figsize=(12, 6) if combine_errors else (16, 8))
        fig.suptitle('Python-X program execution status conversion from baseline to explanation (Top: 0-shot, Bottom: 4-shot)')
        axes[0, 0].set_title('direct -> exp')
        axes[0, 1].set_title('direct -> exp-lbl')
        axes[0, 2].set_title('direct -> exp-lbl-d')
        for row, shot in enumerate([0, 4]):
            for col, exp in enumerate(["exp", "exp-lbl", "exp-lbl-d"]):
                mask = (new_df.exp==exp) & (new_df.shots == shot)
                conversion = error_conversions[mask].mean(0)
                conversion_df = pd.DataFrame(conversion.T, columns=error_cols, index=error_cols)
                second_max, margin = np.sort(conversion.reshape(-1))[-2], 5
                g = sns.heatmap(conversion_df, annot=True, ax=axes[row, col], vmin=0, vmax=second_max+margin)

                axes[row, col].tick_params('x', labelrotation=30)
                xlabel = "status@1 direct" if row == 1 else ""
                g.set_xlabel(xlabel)
                ylabel = f'{shot}-shot status@1 w/ explanation' if col == 0 else ""
                g.set_ylabel(ylabel)
    else:
        fig, axes = plt.subplots(2, 4, sharex=False, sharey=False, figsize=(12, 6) if combine_errors else (16, 8))
        fig.suptitle(
            'Python-X program execution status conversion from baseline to explanation (Top: 0-shot, Bottom: 4-shot)')
        axes[0, 0].set_title('High')
        axes[0, 1].set_title('Medium')
        axes[0, 2].set_title('Low')
        axes[0, 3].set_title('Extremely-Low')
        for row, shot in enumerate([0, 4]):
            for col, res in enumerate(["high", "medium", "low", "extremely low"]):
                mask = (new_df.tgt_lang.isin(res2lang[res])) & (new_df.shots == shot)
                conversion = error_conversions[mask].mean(0)
                conversion_df = pd.DataFrame(conversion.T, columns=error_cols, index=error_cols)
                second_max, margin = np.sort(conversion.reshape(-1))[-2], 5
                g = sns.heatmap(conversion_df, annot=True, ax=axes[row, col], vmin=0, vmax=second_max + margin)

                axes[row, col].tick_params('x', labelrotation=30)
                xlabel = "status@1 direct" if row == 1 else ""
                g.set_xlabel(xlabel)
                ylabel = f'{shot}-shot status@1 w/ explanation' if col == 0 else ""
                g.set_ylabel(ylabel)
        fig.tight_layout()
        plt.savefig(f"../analysis/error_conversion{'_detail' if not combine_errors else ''}{'combine_exp' if not split_exp else ''}_breakdown.png", bbox_inches="tight")
    plt.close()

def plot_coder_reviewer_alpha():
    df = pd.read_csv("../translation_results/coder_reviewer_alpha.csv")
    plt.figure(figsize=(6,6))
    g = sns.lineplot(data=df, x="alpha", y="estimated_pass@1", hue="exp") #  palette=PALETTE
    plt.ylabel("Estimated Pass@1")
    plt.xlabel("Alpha")
    g.axhline(0.83,  c="black", ls='--', linewidth=0.8, zorder=0, clip_on=False)
    g.axhline(0.851, c="black", ls='--', linewidth=0.8, zorder=0, clip_on=False)
    g.axhline(0.861, c="black", ls='--', linewidth=0.8, zorder=0, clip_on=False)
    g.axhline(0.411, c="black", ls='--', linewidth=0.8, zorder=0, clip_on=False)

    g.get_legend().set_title("")
    plt.savefig(f"../analysis/coder_reviewer_alpha.png")
    plt.close()


def plot_nl_code_vs_code_code(df: pd.DataFrame, shots: int=0):
    langs = [k for k in SHORT2CANONICAL.keys() if k not in ["py", "go_test.go"]]
    langs_print = [cap(SHORT2CANONICAL[l]) for l in langs]
    # Python - X, correlated with NL- X
    nl_x_df = pd.read_csv("../analysis/passk.csv", names=["lang", "model","temp","prompt","k","pass"], header=None)
    nl_x_df = nl_x_df[(nl_x_df.model =="davinci")  & (nl_x_df.temp==0.2) & (nl_x_df.k==1) & (nl_x_df.prompt=="reworded")]
    nl_x_df = nl_x_df.set_index("lang").loc[langs]
    # against python-x and against improvements of exp python-x
    df = df[df.shots==shots]
    py_x_direct = df[df.exp=="direct"].set_index("tgt_lang").loc[langs_print]
    py_x_exp = []
    for lang in langs_print:
        sub_df = df[(df.tgt_lang==lang) & (df.exp!="direct")]
        best_exp = sub_df.iloc[sub_df["Pass@1(n=20)"].argmax()]
        py_x_exp.append(best_exp)
    py_x_exp = pd.DataFrame(py_x_exp)
    py_x_improve = py_x_exp["Pass@1(n=20)"].to_numpy() - py_x_direct["Pass@1(n=20)"].to_numpy()
    py_x_rel_improve = py_x_improve/py_x_direct["Pass@1(n=20)"].to_numpy()

    sns.set_theme(style="ticks")
    plt.figure(figsize=(6, 6))
    ax = sns.regplot(x=nl_x_df["pass"], y=py_x_direct["Pass@1(n=20)"])
    ax = sns.regplot(x=nl_x_df["pass"], y=py_x_improve)
    ax = sns.regplot(x=nl_x_df["pass"], y=py_x_rel_improve)
    # for i, row in df_pivot.iterrows():
    #     ax.text(row["exp (0 shot) - direct (0 shot) Pass@1"], row["direct (4 shot) - direct (0 shot) Pass@1"] + .005,
    #             i[1], fontdict={"size": 10})
    r2_direct = r2(nl_x_df["pass"], py_x_direct["Pass@1(n=20)"])
    r2_exp = r2(nl_x_df["pass"], py_x_exp["Pass@1(n=20)"])
    r2_improve = r2(nl_x_df["pass"], py_x_improve)
    r2_rel_improve = r2(nl_x_df["pass"], py_x_rel_improve)
    ax.text(0.35, 0.5, f"r^2(NL-code, direct)={r2_direct:.2f}", fontdict={"size": 10})
    ax.text(0.2, -0.02, f"r^2(NL-code, exp improve)={r2_improve:.2f}", fontdict={"size": 10})
    ax.text(0.25, 0.22, f"r^2(NL-Code, exp relative improve)={r2_rel_improve:.2f}", fontdict={"size": 10})
    plt.ylabel("ChatGPT(gpt3.5-turbo-0301) Python-X Translation pass@1(n=20)")
    plt.xlabel("Codex(davinci) NL-Code pass@1(n=20)")
    plt.title(f"NL-Code vs. translation/exp improvement")
    plt.savefig(f"../analysis/nl-code_vs_translation_regression.png")
    plt.close()

def plot_cross_model_x_x():
    file_path = "../translation_results/cross_model_x_x.csv"
    df = pd.read_csv(file_path)
    df = df[pd.to_numeric(df['Absolute Improvement'], errors='coerce').notnull()]
    df["Absolute Improvement"] = pd.to_numeric(df["Absolute Improvement"])
    df["Relative Improvement"] = pd.to_numeric(df["Relative Improvement"])
    fig, axes = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(8, 4))
    for i, field in enumerate(['Absolute Improvement', 'Relative Improvement']):
        axes[i].set_title(field)
        axes[i].tick_params('x', labelrotation=30)
        g = sns.barplot(data=df, x="Resource", y=field, hue="Model", palette=PALETTE,
                    hue_order=["ChatGPT", "Llama2CodeInstruct-34B", "CodeGen2-16B", "CodeGen2-1B"],
                    ax=axes[i], order=["High-to-High", "ExtLow-to-High", "High-to-ExtLow", "ExtLow-to-ExtLow"])
        g.set_xlabel("")
        y_label = "Best Explanation Improvement Over Baseline" if i == 0 else ""
        g.set_ylabel(y_label)
        if i == 0:
            g.get_legend().remove()
    plt.tight_layout()
    plt.savefig(f"../analysis/cross_model_x_x.png")
    plt.close()

def plot_cross_model_py_x():
    lmap = {"js": "High", "cpp": "High", "jv": "High", "ts": "High",
            "php": "Medium","rb": "Medium","cs": "Medium","go": "Medium",
            "pl": "Low","r": "Low","rs": "Low","sc": "Low","sw": "Low",
            "sh": "Extremely Low","lua": "Extremely Low","rkt": "Extremely Low","jl": "Extremely Low","d": "Extremely Low",
            }
    file_path = "../translation_results/cross_model_py_x.csv"
    df = pd.read_csv(file_path)
    df = pd.wide_to_long(df, "lang", i=["Model", "Metric"], j="value", sep="_", suffix="\D+")
    df = df.reset_index()
    df = df.rename(columns={"lang": "value", "value": "Target Resource"})
    df = df.pivot(index=["Model", "Target Resource"], columns='Metric', values='value').reset_index()
    df["Target Resource"] = df["Target Resource"].map(lmap)

    df = df[pd.to_numeric(df['Absolute Improvement'], errors='coerce').notnull()]
    df = df[pd.to_numeric(df['Relative Improvement'], errors='coerce').notnull()]
    df["Absolute Improvement"] = pd.to_numeric(df["Absolute Improvement"])
    df["Relative Improvement"] = pd.to_numeric(df["Relative Improvement"])
    fig, axes = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(8, 4))
    for i, field in enumerate(['Absolute Improvement', 'Relative Improvement']):
        axes[i].set_title(field)
        # axes[i].tick_params('x', labelrotation=10)
        g = sns.barplot(data=df, x="Target Resource", y=field, hue="Model", palette=PALETTE,
                        hue_order=["ChatGPT", "Llama2CodeInstruct-34B", "CodeGen2-16B", "CodeGen2-1B with ChatGPT", "CodeGen2-1B"],
                        ax=axes[i], order=["High", "Medium", "Low", "Extremely Low"])
        g.set_xlabel("")
        y_label = "Best Explanation Improvement Over Baseline" if i == 0 else ""
        g.set_ylabel(y_label)
        if i == 0:
            g.get_legend().remove()
    plt.tight_layout()
    plt.savefig(f"../analysis/cross_model_py_x.png")
    plt.close()



if __name__=="__main__":
    # df = cleanup_table(pass_fillin=True, experiment_subsets="x_x")
    # plot_x_x(df)
    # df = cleanup_table(pass_fillin=True, experiment_subsets="python_x")
    # df = df.append(cleanup_table(pass_fillin=True, experiment_subsets="python_x"))
    # plot_python_x(df)


    # df = update_error_breakdown_and_length_breakdown(df)

    # plot_error_type_vs_explanation_type(df, combine_errors=True, shots=0)
    # plot_error_type_vs_explanation_type(df, combine_errors=False, shots=0)
    # plot_error_type_vs_explanation_type(df, combine_errors=True, shots=4)
    # plot_error_type_vs_explanation_type(df, combine_errors=False, shots=4)
    # plot_error_type_vs_explanation_type_by_res(df, combine_errors=True)
    # plot_error_type_vs_explanation_type_by_res(df, combine_errors=False)

    # plot_error_conversion(df, combine_errors=True)
    # plot_error_conversion(df, combine_errors=False)
    # plot_error_conversion(df, combine_errors=True, split_exp=False)
    # plot_error_conversion(df, combine_errors=False, split_exp=False)

    # plot_len_vs_explanation_type_scatter(df, shots=0)  # doesn't look great
    # plot_len_vs_explanation_type_scatter(df, shots=4)  # doesn't look great
    # plot_len_vs_explanation_type_bar(df, shots=0, len_type="openai_token", buckets=10)
    # plot_len_vs_explanation_type_bar(df, shots=4, len_type="openai_token", buckets=10)
    # plot_len_vs_explanation_type_bar(df, len_type="openai_token", buckets=4, by_res=True)

    # plot_baseline_vs_explanation_pass_rate_scatter(df, shots=0, improve=True)

    # plot_regression_4shot_vs_0shot_exp(df)
    # plot_coder_reviewer_alpha()
    # plot_nl_code_vs_code_code(df, shots=0)
    plot_cross_model_x_x()
    # plot_cross_model_py_x()
