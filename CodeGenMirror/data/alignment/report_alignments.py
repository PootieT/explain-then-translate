import argparse
import itertools
import pdb
import random
import traceback
import textwrap
import torch
from data.alignment.Aligner import Aligner, FastAlignError
from data.ast.Program import Program
from tqdm import tqdm
import os
import subprocess
from typing import List, Set
from copy import deepcopy
import json
import pandas as pd
from codegen_sources.model.preprocess import XLM_preprocess
from multiprocessing import Pool, cpu_count
from itertools import repeat

PATH_TO_ANSIFILTER="/home/shypula/ansifilter/src"
PATH_TO_FASTALIGN = "/home/shypula/fast_align/build/fast_align"

VOCAB_PATH = str(Program.default_bpe_vocab)

twenty_to_index = [19874,
16202,
550,
27581,
12394,
33022,
4358,
24289,
11546,
33334,
28330,
16148,
34892,
20574,
604,
32200,
1448,
15667,
29434,
10616]

color2number={'black': 0,
                 'red': 1,
                 'green': 2,
                 'yellow': 3,
                 'blue': 4,
                 'magenta': 5,
                 'cyan': 6,
                 'white': 7,
                 'default': 9}


def get_parser():
    parser = argparse.ArgumentParser(description='Report alignments')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input file')
    # parser.add_argument('--alignment', '-a', type=str, required=True,
    #                     help='Alignment file')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output file')
    parser.add_argument('--src_lang', type=str, required=True,
                        help='Source language')
    parser.add_argument('--tgt_lang', type=str, required=True,
                        help='Target language')
    parser.add_argument('--alignent_report_type', type=str, required=False,
                        default="fast_align")
    parser.add_argument('--alignment_direction', type=str, required=False,
                        default="forward")
    parser.add_argument('--token_type', type=str, required=False,
                        default="lexer", help="'lexer' or 'bpe'")
    parser.add_argument('--alignment_strategy', type=str, required=True,
                        help='Alignment strategy. Options: simple_min_tree and majority_vote_tree')
    parser.add_argument('--reorder', type=bool, required=False,
                        default=False, help='Reorder the source tokens to the target by order')
    parser.add_argument('--add_indices', type=bool, required=False, default=False, help='Add indices to the output')
    parser.add_argument('--debug', type=bool, required=False, default=False, help='Debug mode')


    return parser.parse_args()


def color_substr_ansi(string, backgr_color: str = "red", text_color: str = "black"):
    backgr_code = color2number[backgr_color]
    text_color = color2number[text_color]
    # \33[  starts the coloring and \33[ ends
    formatted_substr = f"\033[4{backgr_code};3{text_color}m{string}\033[m"
    return formatted_substr


def color_token_list_indices_with_span(token_list: List[str], token_span: Set[int],
                                  backgr_color: str = "red", text_color: str = "black"):
    assert backgr_color in color2number
    assert text_color in color2number
    for i, token in enumerate(token_list):
        if i in token_span:
            token_list[i] = color_substr_ansi(token, backgr_color, text_color)
    return " ".join(token_list)


def format_src_tgt_string(colored_src, colored_src_tree, colored_tgt, tgt_idx, src_idx_list, src_words, tgt_word):
    src_idx_list = [str(i) for i in src_idx_list]
    s = f"{'-'*20} Alignment at tgt_idx {tgt_idx} and src_indices {' '.join(src_idx_list)}{'-'*20}\n"
    s += f"{'-'*20} With tgt word {tgt_word} and src words {' '.join(src_words)} {'-'*20}\n\n"
    s += textwrap.fill("Target: " + colored_tgt, width=140) + "\n\n"
    s += textwrap.fill("Source: " + colored_src, width=140 ) + "\n\n"
    # s += textwrap.fill("Source: " + colored_src_tree ) + "\n\n"
    s += f"{'-'*20} End of alignment at idx {tgt_idx} {'-'*20}\n\n"
    return s


def get_first_token_before_first_parenthesis(code):
    assert isinstance(code, str) or isinstance(
        code, list
    ), f"function is not the right type, should be str or list : {code}"
    if isinstance(code, str):
        code = code.split()
    return code[code.index("(") - 1]


def txt_to_html(path_to_txt: str):
    base_file, _ = os.path.splitext(path_to_txt)
    path_to_html = base_file + ".html"
    subprocess.run([f"{PATH_TO_ANSIFILTER}/ansifilter",  "-i", path_to_txt, "-o", path_to_html, "-H"], check=True)
    return


def annotate_pair_with_alignment(pair, alignment_string, src_lang, tgt_lang, alignment_strategy = "simple_min_tree",
                                 alignment_type = "fastAlign", alignment_direction = "forward", token_type = "lexer",
                                 pair_separator=" ||| ", seq_separator = " ", src_color="cyan", tgt_color="red",
                                 src_tree_color = "yellow", text_color="black", annotate = True, reorder = False):
    """
    Annotate a pair with the alignment
    :param pair:
    :param alignment_dict:
    :return:
    """
    assert alignment_strategy in ("simple_min_tree", "majority_vote_tree", "word", "bpe")
    src_str, tgt_str = pair.split(pair_separator)
    src_tokens = src_str.split(seq_separator)
    tgt_tokens = tgt_str.split(seq_separator)
    if annotate:
        src_tokens = [f"{i}:{token}" for i, token in enumerate(src_tokens)]
        tgt_tokens = [f"{i}:{token}" for i, token in enumerate(tgt_tokens)]
    aligner = Aligner(src_lang, tgt_lang, src_str, tgt_str, alignment_string, alignment_type, alignment_direction,
                      token_type, reorder)
    # print(f"src_str: {src_str} and tgt_str: {tgt_str}")
    if alignment_strategy not in ("word", "bpe"):
        aligner.build_src_tgt_token2spans()
    # hackey way to pass in 'bpe' alignment strategy but use tree alignments
    aligner.set_tree_alignment_dict(alignment_strategy if alignment_strategy != "bpe" else "word")
    bpe_alignment_list = aligner.get_bpe_alignments_list(debug=False)
    lexer_alignment_list = aligner.get_lexer_tree_alignments_list(debug=False)
    src_tgt_bpe_tuple = aligner.get_src_tgt_prog(tokenized_style=True, apply_bpe=True)
    # hackey way to reset the alignment dict so we can inspect bpe alignments using the following code blocks
    if token_type == "lexer" and alignment_strategy == "bpe":
        src_tokens, tgt_tokens = aligner.get_src_tgt_annotator_bpe_toks()
        src_tokens = [f"{i}:{token}" for i, token in enumerate(src_tokens)]
        tgt_tokens = [f"{i}:{token}" for i, token in enumerate(tgt_tokens)]

        aligner.reset_alignment_dict({k:v for k, v in bpe_alignment_list})
        aligner.reset_tree_alignment_dict({k:v for k, v in bpe_alignment_list})

    result_str = ""
    for tgt_idx, tgt_token in enumerate(tgt_tokens):
        # breakpoint()
        src_idx_list = aligner.get_tgt_idx_aligned_toks(tgt_idx)
        src_tree_list = aligner.get_tgt_idx_aligned_subtree(tgt_idx)
        colored_tgt = color_token_list_indices_with_span(deepcopy(tgt_tokens), [tgt_idx], backgr_color=tgt_color, text_color=text_color)
        colored_src = color_token_list_indices_with_span(deepcopy(src_tokens), src_idx_list, backgr_color=src_color, text_color=text_color)
        colored_src_tree = color_token_list_indices_with_span(deepcopy(src_tokens), src_tree_list, backgr_color=src_tree_color,
                                                         text_color=text_color)
        tgt_word = tgt_tokens[tgt_idx]
        src_words = [src_tokens[i] for i in src_idx_list]
        src_target_alignment_str = format_src_tgt_string(colored_src, colored_src_tree, colored_tgt,
                                                         tgt_idx, src_idx_list, src_words, tgt_word)
        # print(src_target_alignment_str)
        result_str += src_target_alignment_str
    bpe_alignment_list = aligner.get_bpe_alignments_list(debug=False)
    lexer_alignment_list = aligner.get_lexer_tree_alignments_list(debug=False)
    src_tgt_bpe_tuple = aligner.get_src_tgt_prog(tokenized_style=True, apply_bpe=True)
    return result_str, bpe_alignment_list, lexer_alignment_list, src_tgt_bpe_tuple


def write_alignment_report(alignment_report, output_path, src_tgt_pair, src_lang, tgt_lang):
    function_name = get_first_token_before_first_parenthesis(src_tgt_pair)
    output_file = os.path.join(output_path, function_name + f"_{src_lang}_{tgt_lang}" + ".txt")
    with open(output_file, "w") as f:
        f.write(alignment_report)
    txt_to_html(output_file)
    # os.remove(output_file)


lang2lang_endings = {
    "python": "python_sa",
    "java": "java_sa",
}

def get_file_paths(input_path, src_lang, tgt_lang):
    src_lang_sa = lang2lang_endings[src_lang]
    tgt_lang_sa = lang2lang_endings[tgt_lang]

    src_tgt_str = src_lang_sa + "-" + tgt_lang_sa if src_lang_sa < tgt_lang_sa else tgt_lang_sa + "-" + src_lang_sa

    src_path = os.path.join(input_path, f"train.{src_tgt_str}.{src_lang_sa}.bpe")
    tgt_path = os.path.join(input_path, f"train.{src_tgt_str}.{tgt_lang_sa}.bpe")

    return src_path, tgt_path


class FastAligner(object):
    def __init__(self, input_path, output_path, src_lang, tgt_lang, token_type = "lexer", fast_align_iters = 10):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.input_path = input_path
        self.output_path = output_path
        self.fast_align_iters = fast_align_iters
        assert token_type in ("lexer", "bpe"), f"token_type {token_type} not supported, must be lexer or bpe"
        self.token_type = token_type


    def bpe_to_joined_line(self, line):
        return Program.bpe_string_to_joined(line)

    def run_fast_align(self):
        src_corpus, tgt_corpus = self.read_corpus()
        self.output_corpus_path = os.path.join(self.output_path, f"{self.src_lang}_{self.tgt_lang}.corpus.txt")
        with open(self.output_corpus_path, "w") as f:
            for src_line, tgt_line in zip(src_corpus, tgt_corpus):
                paired_line = " ||| ".join([src_line, tgt_line])
                f.write(paired_line + "\n")
        self.output_alignments_path = os.path.join(self.output_path, f"{self.src_lang}_{self.tgt_lang}.alignments.txt")
        # p = None
        with open(self.output_alignments_path, "w") as f:
            p = subprocess.run([PATH_TO_FASTALIGN, "-i", self.output_corpus_path,
                                "-d", "-o", "-I", str(self.fast_align_iters)], stdout=f)
        return p

    def read_corpus(self):
        self.src_path, self.tgt_path = get_file_paths(self.input_path, self.src_lang, self.tgt_lang)
        with open(self.src_path, "r") as f:
            src_corpus = f.readlines()
        with open(self.tgt_path, "r") as f:
            tgt_corpus = f.readlines()
        if self.token_type == "lexer":
            src_corpus = [self.bpe_to_joined_line(line.strip()) for line in src_corpus]
            tgt_corpus = [self.bpe_to_joined_line(line.strip()) for line in tgt_corpus]
        else:
            src_corpus = [line.strip() for line in src_corpus]
            tgt_corpus = [line.strip() for line in tgt_corpus]
        return src_corpus, tgt_corpus

    def get_alignments_path(self):
        if not hasattr(self, "output_alignments_path"):
            self.run_fast_align()
        return self.output_alignments_path

    def get_joined_corpus_path(self):
        if not hasattr(self, "output_corpus_path"):
            self.run_fast_align()
        return self.output_corpus_path


def main():
    ## TODO REFACTOR THIS TO SOMEHOW TAKE CARE OF EDGE CASES WHERE THE JAVA PROGRAM CANNOT BE PARSED
    ## MAYBE USE UNIFORM ATTENTION ????
    ## MAYBE USE A DIFFERENT ALIGNMENT STRATEGY FOR THE JAVA PROGRAM ????

    ## TODO: we also need 2-way attention matrices java-python, python-java


    parser = get_parser()
    input_path = parser.input
    # alignment_file = parser.alignment
    output_path = parser.output
    src_lang = parser.src_lang
    tgt_lang = parser.tgt_lang
    alignment_report_type = parser.alignent_report_type
    alignment_strategy = parser.alignment_strategy
    alignment_direction = parser.alignment_direction
    token_type = parser.token_type
    add_indices = parser.add_indices
    reorder = parser.reorder
    debug = parser.debug
    # pdb.set_trace()

    make_src_tgt_alignments_and_corpus(alignment_direction, token_type, alignment_report_type, alignment_strategy,
                                       input_path, output_path, src_lang, tgt_lang, add_indices, reorder, debug)


    return 0


def write_annotation_df(attention_list, src_tgt_pair, output_path, src_lang, tgt_lang, src_bpe_toks,
                        tgt_bpe_toks, sep = " ||| "):
    src_bpe_toks = ["</s>"] + src_bpe_toks + ["</s>"]
    src_prog, tgt_prog = src_tgt_pair.split(sep)
    attn_dict = {tgt_idx: src_idxs for tgt_idx, src_idxs in attention_list}
    tgt_prog_len = max(len(tgt_bpe_toks), len(tgt_prog.split()))
    filled_attention_list = [(tgt_idx, attn_dict.get(tgt_idx, [])) for tgt_idx in range(tgt_prog_len)]
    method_name = get_first_token_before_first_parenthesis(tgt_prog)
    basename = get_src_tgt_path_basename(src_lang, tgt_lang)
    output_path = os.path.join(output_path, f"{basename}.{method_name}.csv")
    tgt_prog_col = [tgt_prog] * len(filled_attention_list)
    src_prog_col = [src_prog] * len(filled_attention_list)
    tgt_token_col = [tgt_token for tgt_token, _ in filled_attention_list]
    src_tokens_col = [src_tokens for _, src_tokens in filled_attention_list]
    tgt_bpe_toks = [tgt_bpe_toks[i] for i in tgt_token_col]
    src_bpe_toks = [[src_bpe_toks[i] for i in src_idxs] for src_idxs in src_tokens_col]
    verified_src_tokens_col = [src_tokens for _, src_tokens in filled_attention_list] # initialize the col
    df = pd.DataFrame({"src_prog": src_prog_col, "tgt_prog": tgt_prog_col, "tgt_token": tgt_token_col,
                       "src_tokens": src_tokens_col, "tgt_bpe_toks": tgt_bpe_toks, "src_bpe_toks": src_bpe_toks,
                       "verified_src_tokens": verified_src_tokens_col})
    df.to_csv(output_path, index=False)


def make_src_tgt_alignments_and_corpus(alignment_direction, token_type, alignment_report_type, alignment_strategy,
                                       input_path, output_path, src_lang, tgt_lang, annotate, reorder, debug):

    src_tgt_alignments, src_tgt_corpus = get_alignments_corpus_fastalign(input_path, output_path, src_lang, tgt_lang, token_type)
    tgt_src_alignments, tgt_src_corpus = get_alignments_corpus_fastalign(input_path, output_path, tgt_lang, src_lang, token_type)

    pbar = tqdm(total=len(src_tgt_alignments))
    if debug:
        _starmap = itertools.starmap
    else:
        pool = Pool(processes=cpu_count())
        _starmap = pool.starmap
    src_out_bpe_path, tgt_out_bpe_path = get_file_paths(output_path,
                                                        src_lang,
                                                        tgt_lang)

    src_bpe_out_fh, tgt_bpe_out_fh = open(src_out_bpe_path, "w+"), open(tgt_out_bpe_path, "w+")
    src_bpe_attns = []
    tgt_bpe_attns = []
    n_success = 0
    args_list = [(alignment_direction, token_type, alignment_report_type, alignment_strategy, annotate, reorder,
                  output_path, src_alignment, src_lang, src_tgt_pair, tgt_alignment, tgt_lang, tgt_src_pair, i)
                    for i, (src_alignment, src_tgt_pair, tgt_alignment, tgt_src_pair) in
                    enumerate(zip(src_tgt_alignments, src_tgt_corpus, tgt_src_alignments, tgt_src_corpus)) if i in twenty_to_index]
    # random.shuffle(args_list)
    # # sample only 50
    # args_list = args_list[:50]
    for is_successful, src_bpe_attn_list, tgt_bpe_attn_list, (src_bpe, tgt_bpe) in _starmap(
            get_and_save_src_tgt_alignments, args_list
    ):
        if is_successful:
            src_bpe_attns.append(src_bpe_attn_list)
            tgt_bpe_attns.append(tgt_bpe_attn_list)
            src_bpe_out_fh.write(src_bpe + "\n")
            tgt_bpe_out_fh.write(tgt_bpe + "\n")
            n_success += 1
            if n_success == 50:
                break
        pbar.update(1)
        pbar.set_description(f"Successful: {n_success}/{len(src_tgt_alignments)}")
    if not debug:
        pbar.close()
        pool.close()
        pool.join()

    save_attentions_json(src_bpe_attns, output_path, src_lang, tgt_lang)
    save_attentions_json(tgt_bpe_attns, output_path, tgt_lang, src_lang)
    src_bpe_out_fh.close()
    tgt_bpe_out_fh.close()
    XLM_preprocess(voc_path=VOCAB_PATH, txt_path=src_out_bpe_path, bin_path=src_out_bpe_path.replace(".bpe", ".pth"))
    XLM_preprocess(voc_path=VOCAB_PATH, txt_path=tgt_out_bpe_path, bin_path=tgt_out_bpe_path.replace(".bpe", ".pth"))
    return None


def get_and_save_src_tgt_alignments(alignment_direction, token_type, alignment_report_type, alignment_strategy, annotate,
                                    reorder, output_path, src_alignment, src_lang, src_tgt_pair, tgt_alignment, tgt_lang,
                                    tgt_src_pair, id):
    try:
        # pdb.set_trace()
        src_alignment_report, src_bpe_attn_list, src_lexer_attn_list, (src_bpe, tgt_bpe) = annotate_pair_with_alignment(
            pair=src_tgt_pair, alignment_string=src_alignment, src_lang=src_lang, tgt_lang=tgt_lang,
            alignment_strategy=alignment_strategy, alignment_type=alignment_report_type,
            alignment_direction=alignment_direction, token_type=token_type,
            annotate=annotate, reorder=reorder
        )
        tgt_alignment_report, tgt_bpe_attn_list, tgt_lexer_attn_list, (
        tgt_bpe_2, src_bpe_2) = annotate_pair_with_alignment(
            pair=tgt_src_pair, alignment_string=tgt_alignment, src_lang=tgt_lang, tgt_lang=src_lang,
            alignment_strategy=alignment_strategy, alignment_type=alignment_report_type,
            alignment_direction=alignment_direction, token_type=token_type,
            annotate=annotate, reorder=reorder
        )
    except (RuntimeError, FastAlignError, SyntaxError) as e:
        print(f"Error {e} for pair: ", src_tgt_pair)
        traceback.print_exc()
        is_successful = False
        src_bpe_attn_list, tgt_bpe_attn_list = [], []
        src_bpe = tgt_bpe = ""
    else:
        assert src_bpe == src_bpe_2
        assert tgt_bpe == tgt_bpe_2
        is_successful = True
        fun_level_output_path = os.path.join(output_path,
                                            f"{id}_" + get_first_token_before_first_parenthesis(src_tgt_pair))
        if not os.path.exists(fun_level_output_path):
            os.mkdir(fun_level_output_path)
        write_alignment_report(src_alignment_report, fun_level_output_path, src_tgt_pair, src_lang, tgt_lang)
        write_alignment_report(tgt_alignment_report, fun_level_output_path, tgt_src_pair, tgt_lang, src_lang)
        # pdb.set_trace()
        if annotate:
            write_annotation_df(src_lexer_attn_list, src_tgt_pair, fun_level_output_path, src_lang, tgt_lang, src_bpe.split(), tgt_bpe.split())
            write_annotation_df(tgt_lexer_attn_list, tgt_src_pair, fun_level_output_path, tgt_lang, src_lang, tgt_bpe_2.split(), src_bpe_2.split())
    finally:
        return is_successful, src_bpe_attn_list, tgt_bpe_attn_list, (src_bpe, tgt_bpe)


def get_alignments_corpus_fastalign(input_path, output_path, src_lang, tgt_lang, token_type):
    fast_aligner = FastAligner(input_path, output_path, src_lang, tgt_lang, token_type)
    fast_aligner.run_fast_align()
    corpus_path = fast_aligner.get_joined_corpus_path()
    alignments_path = fast_aligner.get_alignments_path()
    # corpus_path = "/nobackup/users/shypula/fastalign_sandbox/j2py.txt"
    # alignments_path = "/nobackup/users/shypula/fastalign_sandbox/j2py.diag_optim.align"
    print("Corpus path: ", corpus_path)
    print("Alignments path: ", alignments_path)
    with open(corpus_path, "r") as f:
        corpus = f.readlines()
    with open(alignments_path, "r") as f:
        alignments = f.readlines()
    return alignments, corpus


def save_attentions_json(alignments, output_path, src_lang, tgt_lang):
    src_lang = lang2lang_endings[src_lang]
    tgt_lang = lang2lang_endings[tgt_lang]
    basename = get_src_tgt_path_basename(src_lang, tgt_lang)
    with open(os.path.join(output_path, f"train.{basename}.alignments.json"), "w") as f:
        json.dump(alignments, f)


def get_src_tgt_path_basename(src_lang, tgt_lang):
    src_tgt_string = src_lang + "-" + tgt_lang if src_lang < tgt_lang else tgt_lang + "-" + src_lang
    basename = src_tgt_string + "." + tgt_lang
    return basename


if __name__ == "__main__":
    main()







