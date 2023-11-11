import pdb

from data.ast.SpanAnnotator import SpanAnnotator
from data.ast.PythonSpanAnnotator import PythonSpanAnnotator
from data.ast.JavaSpanAnnotator import JavaSpanAnnotator
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set, Optional
import torch

class FastAlignError(Exception):
    pass

# TODO refactor this to use __init__.py

class Aligner:
    def __init__(self, src_lang: str, tgt_lang: str, src_program_string: str, tgt_program_string: str, alignments: str,
                 alignment_type = "fastAlign", alignment_direction = "forward", token_type = "lexer", reorder=False):

        assert alignment_type in ["fastAlign", "fast_align"], "Unsupported alignment type: " + alignment_type
        assert alignment_direction in ["forward", "backward"], "Unsupported alignment direction: " + alignment_direction
        assert token_type in ["lexer", "bpe"], "Unsupported token type: " + token_type
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.alignments = alignments
        self.alignment_type = alignment_type
        self.alignment_direction = alignment_direction
        self.token_type = token_type
        self.alignment_dict = None
        self.tree_alignment_dict = None
        self.bpe_alignment_stragety = None
        self.reorder = reorder

        if src_lang == 'python':
            self.src_annotator = PythonSpanAnnotator(src_program_string)
        elif src_lang == 'java':
            self.src_annotator = JavaSpanAnnotator(src_program_string)
        else:
            raise Exception('Unsupported source language: ' + src_lang)

        if tgt_lang == 'python':
            self.tgt_annotator = PythonSpanAnnotator(tgt_program_string)
        elif tgt_lang == 'java':
            self.tgt_annotator = JavaSpanAnnotator(tgt_program_string)
        else:
            raise Exception('Unsupported target language: ' + tgt_lang)

        # parse alignments
        self._parse_alignmets()

    def _parse_alignmets(self):
        assert self.alignments is not None
        if self.alignment_type == "fastAlign" or "fast_align":
            self.alignment_dict = self.parse_fast_align_alignments(self.alignments)
        else:
            raise Exception('Unsupported alignment type: ' + self.alignment_type)

    @staticmethod
    def parse_fast_align_alignments(alignment_string):
        """
        Parse the alignment string into a dictionary
        :param alignment_string:
        :return: dictionary of the form {target_token_index: source_token_index}
        """
        alignment_string = alignment_string.strip()
        alignment_dict = defaultdict(list)
        alignment_list = alignment_string.strip().split(" ")
        for i, align_pair in enumerate(alignment_list):
            try:
                src_idx, tgt_idx = align_pair.split("-")
            except ValueError:
                print(align_pair)
                print(alignment_list[:i])
                raise
            assert src_idx.isdigit() and tgt_idx.isdigit()
            alignment_dict[int(tgt_idx)].append(int(src_idx))
        return {k: sorted(v) for k, v in sorted(alignment_dict.items(), key=lambda x: x[0])}

    def get_tgt_idx_aligned_toks(self, tgt_idx):
        """
        query self.alignment_dict for the src tokens index that is aligned to the target token index
        """
        return self.alignment_dict.get(tgt_idx, [])

    def get_alignment_dict(self):
        return self.alignment_dict

    def _build_token2spans(self, annotator):
        annotator.set_span_prog_to_tok2span()

    def _get_token2spans(self, annotator):
        return annotator.get_token2span()

    def build_src_tgt_token2spans(self):
        self._build_token2spans(self.src_annotator)
        self._build_token2spans(self.tgt_annotator)

    def set_tree_alignment_dict(self, strategy):
        if strategy == "simple_min_tree":
            self.tree_alignment_dict = self.simple_min_tree_span_strategy()
        elif strategy == "majority_vote_tree":
            self.tree_alignment_dict = self.majority_vote_min_tree_span_strategy()
        elif strategy == "word":
            self.tree_alignment_dict = self.alignment_dict
        else:
            raise Exception('Unsupported strategy: ' + strategy)

    def reset_tree_alignment_dict(self, alignment_dict):
        self.tree_alignment_dict = alignment_dict

    def reset_alignment_dict(self, alignment_dict):
        self.alignment_dict = alignment_dict

    def get_tgt_idx_aligned_subtree(self, tgt_idx):
        assert self.tree_alignment_dict is not None
        return self.tree_alignment_dict.get(tgt_idx, {})

    def simple_min_tree_span_strategy(self) -> Set:
        assert self.alignment_dict is not None
        tgt_token_to_src_tree = defaultdict(set)
        for tgt, src_tokens in self.alignment_dict.items():
            for src_token in src_tokens:
                tgt_token_to_src_tree[tgt].update(self.src_annotator.get_token_nth_node_span(
                    token_index=src_token, n=0, tree_strategy=True))
        return tgt_token_to_src_tree

    def majority_vote_min_tree_span_strategy(self) -> Set:
        """
        Build on top of simple_min_tree_span_strategy, but use majority vote to pick the best span
        """
        simple_tgt_token_to_src_tree = self.simple_min_tree_span_strategy()
        # print(self.tgt_annotator.token2same_level_tokens)
        voted_tgt_token_to_src_tree = defaultdict(set)
        cached_tgt_tree_to_majority_vote_span = dict()
        for tgt_token in simple_tgt_token_to_src_tree:
            matching_subtree_spans = []
            tgt_min_tree_span = frozenset(self.tgt_annotator.get_token_same_level_tokens(tgt_token))
            # pdb.set_trace()
            if tgt_min_tree_span in cached_tgt_tree_to_majority_vote_span:
                voted_tgt_token_to_src_tree[tgt_token] = cached_tgt_tree_to_majority_vote_span[tgt_min_tree_span]
            else:
                for tgt_token in tgt_min_tree_span:
                    src_tree_span = simple_tgt_token_to_src_tree[tgt_token]
                    if src_tree_span:
                        matching_subtree_spans.append(src_tree_span)
                if matching_subtree_spans:
                    majority_vote_span = self.majority_vote_span(matching_subtree_spans)
                    voted_tgt_token_to_src_tree[tgt_token] = majority_vote_span
                    cached_tgt_tree_to_majority_vote_span[tgt_min_tree_span] = majority_vote_span

        return voted_tgt_token_to_src_tree

    def majority_vote_span(self, matching_subtree_spans: List[Set[int]]):
        """
        :param matching_subtree_spans: list of spans, each of which is a set of src token indices
        :return: the span that is the most common among the spans
        """
        counts = Counter([frozenset(span) for span in matching_subtree_spans])
        return max(counts, key=counts.get)


    def build_bpe_alignments(self, debug=False):
        # TODO double check the index if the alignment is NULL, current increments all tokens by 1, and then
        # if the alignment is NULL, it is aligned to 0
        if self.token_type == "bpe":
            self.build_bpe_alignments_bpe_alignments(debug)
        else:
            self.build_bpe_alignments_lexer_alignments(debug)
        return None

    def build_bpe_alignments_bpe_alignments(self, debug=False):
        src_bpe_tokens, tgt_bpe_tokens = self.get_src_tgt_annotator_bpe_toks()
        self.bpe_alignments = []
        for tgt_idx, tgt_bpe_token in enumerate(tgt_bpe_tokens):
            src_bpe_tokens_idxs = self.alignment_dict.get(tgt_idx, [])
            src_bpe_tokens_idxs = [t + 1 for t in src_bpe_tokens_idxs] \
                                  if src_bpe_tokens_idxs else [len(src_bpe_tokens) - 1]
            if any([src_bpe_token_idx >= len(src_bpe_tokens) for src_bpe_token_idx in src_bpe_tokens_idxs]):
                raise FastAlignError
            self.bpe_alignments.append((tgt_idx, src_bpe_tokens_idxs))
            if debug:
                print("tgt_bpe_idx:", tgt_idx)
                print("tgt_bpe_token:", tgt_bpe_token)
                print("src_bpe_tokens_idxs:", src_bpe_tokens_idxs)
                print("src_bpe_tokens:", [src_bpe_tokens[t] for t in src_bpe_tokens_idxs])
        self.bpe_alignments.append((tgt_idx + 1, [len(src_bpe_tokens) - 1]))
        self.bpe_alignments.append((tgt_idx + 2, [len(src_bpe_tokens) - 1]))  # once again for </s> ?
        max_src = self.bpe_alignments[-1][1][0] + 1
        assert max_src == (len(src_bpe_tokens)), f"max src: {max_src}, len of src bpe tokens: {len(src_bpe_tokens)}"
        assert (max_src) > max(
            [src_token_idx for tgt_idx, src_token_idxs in self.bpe_alignments for src_token_idx in src_token_idxs]), \
            f"max src: {max_src}, max src token idx: {max(src_token_idx for tgt_idx, src_token_idxs in self.bpe_alignments for src_token_idx in src_token_idxs)}" \
            f"where the bpe alignments are: {self.bpe_alignments}"
        self.build_alingments_matrix()


    def build_bpe_alignments_lexer_alignments(self, debug=False):
        assert self.tree_alignment_dict is not None
        src_bpe_tokens, tgt_bpe_tokens = self.get_src_tgt_annotator_bpe_toks()
        src_lexer_tokens = self.src_annotator.get_tokenized_program_list()
        tgt_lexer_tokens = self.tgt_annotator.get_tokenized_program_list()
        self.bpe_alignments = []
        for tgt_idx, tgt_bpe_token in enumerate(tgt_bpe_tokens):
            tgt_lexer_token_idx = self.tgt_annotator.get_bpe_index_to_token_index(tgt_idx)
            src_lexer_token_idxs = self.get_tgt_idx_aligned_subtree(tgt_lexer_token_idx)
            # this occurred becauase FastAlign seems incapable of correctly handling some sequences
            # in which a src index will in fact be out of range of the src_lexer_tokens
            if any([src_lexer_token_idx >= len(src_lexer_tokens) for src_lexer_token_idx in src_lexer_token_idxs]):
                raise FastAlignError
                # src_lexer_token_idxs = [min(src_lexer_token_idx, len(src_lexer_tokens) - 1) for src_lexer_token_idx in
                #                        src_lexer_token_idxs]
            # increment by 1 for </s> and len(src_bpe_tokens) + 1 for final </s>
            src_bpe_token_idxs = ([t + 1 for src_lexer_token_idx in src_lexer_token_idxs
                                  for t in self.src_annotator.get_token_to_bpe_indices(src_lexer_token_idx)] if
                                  src_lexer_token_idxs else [len(src_bpe_tokens)-1])
            # print(f"src bpe token idxs: {src_bpe_token_idxs}, and len of src bpe tokens: {len(src_bpe_tokens)}")
            src_aligned_bpe_tokens = [src_bpe_tokens[bpe_idx] for bpe_idx in src_bpe_token_idxs]
            src_bpe_token_idxs = src_bpe_token_idxs if (src_bpe_token_idxs != []) else [0]
            # we do not add 1, even though there is </s>, because the previous token should attend to the next token's
            # alignment
            self.bpe_alignments.append((tgt_idx, src_bpe_token_idxs))
            assert (self.src_annotator.bpe_tokens_to_joined(src_aligned_bpe_tokens) ==
                    " ".join(src_lexer_tokens[src_lexer_token_idx] for src_lexer_token_idx in src_lexer_token_idxs) or
                    self.src_annotator.bpe_tokens_to_joined(src_aligned_bpe_tokens) == "</s>"), \
                f"{self.src_annotator.bpe_tokens_to_joined(src_aligned_bpe_tokens)} != " \
                f"{' '.join(src_lexer_tokens[src_lexer_token_idx] for src_lexer_token_idx in src_lexer_token_idxs)}"
            if debug:
                tgt_aligned_lexer_token = tgt_lexer_tokens[tgt_lexer_token_idx]
                src_aligned_lexer_tokens = [src_lexer_tokens[src_lexer_token_idx] for src_lexer_token_idx in src_lexer_token_idxs]
                print(
                    f"tgt_bpe_idx: {tgt_idx}, tgt_bpe_token: {tgt_bpe_token}, tgt_lexer_token_idx: {tgt_lexer_token_idx}, "
                    f"tgt_lexer_token: {tgt_aligned_lexer_token}, src_lexer_token_idxs: {src_lexer_token_idxs}, "
                    f"src_lexer_tokens {src_aligned_lexer_tokens}, src_bpe_token_idxs: {src_bpe_token_idxs}, "
                    f"src_bpe_tokens: {src_aligned_bpe_tokens}, ")
        # we need to predict </s> after the last token, add the attention alignment for the last token to the src </s>
        self.bpe_alignments.append((tgt_idx+1, [len(src_bpe_tokens)-1]))
        self.bpe_alignments.append((tgt_idx+2, [len(src_bpe_tokens) - 1])) # once again for </s> ?
        if self.reorder:
            self.bpe_alignments = self.reorder_bpe_alignments(self.bpe_alignments)
        max_src = self.bpe_alignments[-1][1][0] + 1
        assert max_src == (len(src_bpe_tokens)), f"max src: {max_src}, len of src bpe tokens: {len(src_bpe_tokens)}"
        assert (max_src) > max([src_token_idx for tgt_idx, src_token_idxs in self.bpe_alignments for src_token_idx in src_token_idxs])
        self.build_alingments_matrix()
        return None

    def get_src_tgt_annotator_bpe_toks(self):
        src_bpe_tokens = ["</s>"] + self.src_annotator.get_bpe_tokens_list() + ["</s>"]
        tgt_bpe_tokens = self.tgt_annotator.get_bpe_tokens_list()
        return src_bpe_tokens, tgt_bpe_tokens

    @staticmethod
    def find_non_monotonic_alignments(bpe_alignments):
        """
        Finds the non-monotonic alignments in the BPE alignments.
        :return:
        """
        non_monotonic_alignments = []
        previous_src_token_idxs = []
        rolling_tgt_indices = []
        previous_tgt_index = None
        for i, (tgt_idx, src_token_idxs) in enumerate(bpe_alignments):
            if previous_src_token_idxs == src_token_idxs and (tgt_idx == previous_tgt_index + 1):
                if rolling_tgt_indices == []:
                    rolling_tgt_indices.append(previous_tgt_index) # works if we just entered
                rolling_tgt_indices.append(tgt_idx) # works for inside

            # we're in matching and and we break out
            if (previous_src_token_idxs != src_token_idxs
                  or (i == len(bpe_alignments) - 1)
                  or (tgt_idx != previous_tgt_index + 1)):

                if rolling_tgt_indices != []:
                    non_monotonic_alignments.append((rolling_tgt_indices, previous_src_token_idxs))
                    rolling_tgt_indices = []
                # we're not matching and we continue
                else:
                    pass

            previous_src_token_idxs = src_token_idxs
            previous_tgt_index = tgt_idx

        return non_monotonic_alignments

    @staticmethod
    def reorder_non_monotonic_alignments(list_of_alignments: List[Tuple[List[int], List[int]]]):
        """
        Reorders the non-monotonic alignments in the BPE alignments.
        :return:
        """
        new_list = []
        for tgt_token_idxs, src_token_idxs in list_of_alignments:
            new_tuples = Aligner._reorder_helper(tgt_token_idxs, src_token_idxs)
            new_list.extend(new_tuples)
        return new_list

    @staticmethod
    def _reorder_helper(tgt_token_idxs: List[int], src_token_idxs: List[int]):
        if len(tgt_token_idxs) < len(src_token_idxs):
            ## match the tgt_token_idxs to the src_token_idxs backwards
            subset_src_indices = src_token_idxs[-len(tgt_token_idxs):]
            new_list = [(t, [s]) for t, s in zip(tgt_token_idxs, subset_src_indices)]
        elif len(tgt_token_idxs) > len(src_token_idxs):
            new_list = [(t, [s]) for t, s in zip(tgt_token_idxs[:len(src_token_idxs)], src_token_idxs)] + \
                       [(t, [src_token_idxs[-1]]) for t in tgt_token_idxs[len(src_token_idxs):]]
        else:
            new_list = [(t, [s]) for t, s in zip(tgt_token_idxs, src_token_idxs)]
        return new_list

    @staticmethod
    def reorder_bpe_alignments(bpe_alignments):
        """
        Reorders the BPE alignments.
        :return:
        """
        non_monotonic_alignments = Aligner.find_non_monotonic_alignments(bpe_alignments)
        new_alignments = Aligner.reorder_non_monotonic_alignments(non_monotonic_alignments)
        new_alignment_dict = {tgt_idx: src_token_idxs for tgt_idx, src_token_idxs in new_alignments}
        new_bpe_alignments = []
        for tgt_bpe, src_bpe_idxs in bpe_alignments:
            if tgt_bpe in new_alignment_dict:
                src_bpe_idxs = new_alignment_dict[tgt_bpe]
            new_bpe_alignments.append((tgt_bpe, src_bpe_idxs))
        return new_bpe_alignments


    def build_alingments_matrix(self):
        # TODO double check the dimension of the alignment matrix !!
        # add 2 for </s> and </s> ?
        return self.alignment_matrix_from_alignment_list(self.bpe_alignments)

    @staticmethod
    def alignment_matrix_from_alignment_list(alignment_list):
        """
        :param alignment_list: list of tuples (tgt_idx, src_bpe_token_idxs)
        :return:
        """
        # TODO double check the dimension of the alignment matrix !!
        d0 = len(alignment_list)
        # for source tokens, the last token in the target is aligned to the last token in the source </s>
        assert len(alignment_list[-1][1]) == 1, f"{len(alignment_list[-1][1])} != 1"
        d1 = alignment_list[-1][1][0] + 1
        assert d1 > max([src_token_idx for tgt_idx, src_token_idxs in alignment_list for src_token_idx in src_token_idxs]), \
        f"{d1} <= {max([src_token_idx for tgt_idx, src_token_idxs in alignment_list for src_token_idx in src_token_idxs])}"
        attn_mtx = torch.zeros(d0, d1)
        for tgt_idx, src_bpe_token_idxs in alignment_list:
            normalized_attention = 1.0 / len(src_bpe_token_idxs)
            for src_bpe_token_idx in src_bpe_token_idxs:
                attn_mtx[tgt_idx, src_bpe_token_idx] = normalized_attention
        return attn_mtx

    def get_bpe_alignments_list(self, debug=False):
        if not hasattr(self, "bpe_alignments"):
            self.build_bpe_alignments(debug)
        return self.bpe_alignments

    def get_bpe_alignments_matrix(self, debug=False):
        if not hasattr(self, "attn_mtx"):
            self.build_bpe_alignments(debug)
        return self.attn_mtx

    def get_lexer_tree_alignments_list(self, debug=False):
        if not hasattr(self, "tree_alignment_dict"):
            self.build_lexer_tree_alignments(debug)
        return [(k, list(v)) for k, v in self.tree_alignment_dict.items()]

    def get_src_prog(self, tokenized_style=True, apply_bpe=False):
        return self.src_annotator.get_prog(tokenized_style=tokenized_style, apply_bpe=apply_bpe)

    def get_tgt_prog(self, tokenized_style=True, apply_bpe=False):
        return self.tgt_annotator.get_prog(tokenized_style=tokenized_style, apply_bpe=apply_bpe)

    def get_src_tgt_prog(self, tokenized_style=True, apply_bpe=False):
        return (self.src_annotator.get_prog(tokenized_style=tokenized_style, apply_bpe=apply_bpe),
                self.tgt_annotator.get_prog(tokenized_style=tokenized_style, apply_bpe=apply_bpe))













        



