import pdb
from typing import Iterable
from _io import TextIOWrapper
from data.ast.utils import *
import re
from collections import defaultdict
from sys import stdout as STDOUT
from sys import stderr as STDERR
from data.ast.Program import Program

class SpanAnnotator(Program):

    ### Fields available for use in all SpanAnnotator
    BEGIN_PARENS_CHAR = "快" # "α"
    END_PARENS_CHAR = "乐"
    char2special_char = {"(": BEGIN_PARENS_CHAR,
                         ")": END_PARENS_CHAR, }
    special_char2char = {v: k for k, v in char2special_char.items()}

    def __init__(self, input_program = None, obfuscate=False, bpe_model=None, call_super=True):
        if call_super:
            super().__init__(input_program, obfuscate, bpe_model)
        self.spans = None
        self.annotated_prog_str = None
        self.position_dict = None
        self.tokens2spans = None
        self.token2same_level_tokens = None

    def get_spans(self):
        """ Get spans for the input program. To be overridden by subclasses.
        """
        assert self.spans is not None
        raise NotImplementedError()

    def canonicalize_code(self, input_program: str):
        ## Hackey fix for python indentation
        input_program = re.sub(f"     (?={SpanAnnotator.BEGIN_PARENS_CHAR})", "    ", input_program)
        return self.detokenize_code(self.tokenize_code(input_program))

    # TODO pull get_prog up into Program, and also then pull self.remove_class from code up...
    def get_prog(self, tokenized_style=True, apply_bpe=False):
        prog = self.remove_class_from_code(self.input_program)
        prog = " ".join(self.tokenize_code(prog)) if tokenized_style else prog
        return self.apply_bpe(prog) if apply_bpe else prog

    def _set_spans(self):
        """ Set spans for the input program. To be overridden by subclasses.
        """
        raise NotImplementedError()

    def _annotate(self):
        """
        Annotate the program string with the spans. Sets self.annotate_prog_str and returns as well.
        """
        span_list = self.get_spans()
        prog_str = self.input_program
        while len(span_list) > 0:
            start, end = span_list.pop(0)
            prog_str = (prog_str[:start] +
                        self.BEGIN_PARENS_CHAR + prog_str[start:end] + self.END_PARENS_CHAR +
                        prog_str[end:])
            span_list = self.add_one_to_all_elements_greater(start, span_list)
            span_list = self.add_one_to_all_elements_greater(end + 1, span_list)
        self.annotated_prog_str = self.reformat_special_tokens(prog_str)
        self.annotated_prog_str = self.remove_class_from_code(self.annotated_prog_str)
        return self.annotated_prog_str

    def set_position_dict(self):
        assert self.spans not in (None, [])
        self.position_dict = PositionDict(self.spans)
        return None

    def get_position_dict(self):
        assert self.position_dict is not None
        return self.position_dict

    def get_token_spans(self):
        raise NotImplementedError()

    def map_tokenized_program_to_detokenized_program(self):
        raise NotImplementedError()

    def get_tokens2spans(self):
        assert self.tokens2spans is not None
        return self.tokens2spans

    def set_span_prog_to_tok2span(self):
        annotated_prog_string = self.get_annotated_prog_str(tokenized_style=True, prune_spans=True)
        # pdb.set_trace()
        token2spans, token2same_level_tokens, i, current_subtree_tokens = self._span_prog_to_tok2span(
            annotated_prog_string.split()
        )
        self.tokens2spans = token2spans
        self.token2same_level_tokens = token2same_level_tokens
        return None

    def _get_span_prog_idx_2_stripped_idx(self, span_prog_list):
        span_prog_idx_2_stripped_idx = {}
        non_span_counter = 0
        for i, tok in enumerate(span_prog_list):
            if tok not in (self.BEGIN_PARENS_CHAR, self.END_PARENS_CHAR):
                span_prog_idx_2_stripped_idx[i] = non_span_counter
                non_span_counter += 1
            else:
                pass
        print(span_prog_idx_2_stripped_idx)
        return span_prog_idx_2_stripped_idx

    def _span_prog_to_tok2span(self, span_prog_list, token2spans=None, token2same_level_tokens=None, i=0):
        pruned_span_prog_list = [t for t in span_prog_list if t not in self.special_char2char.keys()]
        span_prog_idx_2_stripped_idx = self._get_span_prog_idx_2_stripped_idx(span_prog_list)
        token2spans = defaultdict(list) if token2spans == None else token2spans
        token2same_level_tokens = defaultdict(list) if token2same_level_tokens == None else token2same_level_tokens
        current_subtree_tokens = set()
        current_level_tokens = set()

        while i < len(span_prog_list):
            # if i > (len(span_prog_list) - 6):
            #     pdb.set_trace()
            tok = span_prog_list[i]

            if tok == self.BEGIN_PARENS_CHAR:
                token2spans, _, i, new_span_prog_tokens = self._span_prog_to_tok2span(
                    span_prog_list,
                    token2spans,
                    token2same_level_tokens,
                    i + 1)
                # add all children node's tokens (recursive) to the this subtree's span
                current_subtree_tokens.update(new_span_prog_tokens)

            elif tok == self.END_PARENS_CHAR:  # inside a subtree where it has ended
                ## add the current subtree as a parent for a token
                for subtree_idx in current_subtree_tokens:
                    token2spans[subtree_idx].append(current_subtree_tokens)
                for same_level_idx in current_level_tokens:
                    assert same_level_idx not in token2same_level_tokens, \
                        f"{same_level_idx} already in token2same_level_tokens {token2same_level_tokens}"
                    token2same_level_tokens[same_level_idx] = current_level_tokens
                print("added current level tokens {}".format(current_level_tokens), [pruned_span_prog_list[i] for i in current_level_tokens])
                return token2spans, token2same_level_tokens, i + 1, current_subtree_tokens

            else:
                current_subtree_tokens.add(span_prog_idx_2_stripped_idx[i])
                current_level_tokens.add(span_prog_idx_2_stripped_idx[i])
                i += 1
        

        return token2spans, token2same_level_tokens, i, current_subtree_tokens

    def get_token_same_level_tokens(self, token_idx):
        ## warning that this will not work if we do not build the annotated program without pruned spans
        assert self.token2same_level_tokens is not None
        return self.token2same_level_tokens[token_idx]

    def get_token_nth_node_span(self, token_index, n, tree_strategy = False, check_tree_strategy = False):
        """
        Clarification: tree strategy set to true returns "minimum subtree span" by removing the
        span that exactly matches the token itself
        """
        assert self.tokens2spans is not None, "set_span_prog_to_tok2span() must be called first"
        parent_span_list = self.tokens2spans.get(token_index, [])
        if len(parent_span_list) == 0:
            print("error: didn't find any parent span, token_index is {} \
                        and the token2spans dict is {}".format(token_index, self.tokens2spans), file=STDERR)
            # pdb.set_trace()
            raise ValueError
        else:
            if tree_strategy:
                # remove the span that exactly matches the token itself
                parent_span_list = [span for span in parent_span_list if len(span) != 1]
                if check_tree_strategy:
                    exact_token_span = [span for span in parent_span_list if len(span) == 1]
                    assert [list(span)[0] == token_index for span in exact_token_span], \
                        "token_index was not found in the exact_token_span list {}".format(exact_token_span)
            node_span = parent_span_list[min(n, len(parent_span_list) - 1)]
            assert token_index in node_span, "token_index {} not in node_span {}".format(token_index, node_span)
            return node_span

    @staticmethod
    def remove_class_from_code(code):
        """
        Remove the class declaration from the code if it is there.
        """
        any_char = f" {SpanAnnotator.BEGIN_PARENS_CHAR}{SpanAnnotator.END_PARENS_CHAR}"
        if (re.search(f"public[{any_char}]+class[{any_char}]+Main", code) or
                re.search(f"public[{any_char}]+class[{any_char}]+CLASS_0", code)):
            split = code.split("{")
            joined = "{".join(split[1:])
            split = joined.split("}")
            code = "}".join(split[:-1])
        return code.strip()

    def reformat_special_tokens(self, annotated_prog_str: str):
        l = re.split(f"({SpanAnnotator.BEGIN_PARENS_CHAR})|({SpanAnnotator.END_PARENS_CHAR})", annotated_prog_str)
        return self.canonicalize_code(" ".join([t for t in l if t is not None]).strip())
        # return re.sub("  ", " ", " ".join([t for t in l if t is not None]).strip())
    
    def reformat_indiv_special_tokens(self, annotated_prog_str):
        new_tokens = []
        for token in annotated_prog_str.split():
            toks = re.split(f"({SpanAnnotator.BEGIN_PARENS_CHAR})|({SpanAnnotator.END_PARENS_CHAR})", token)
            new_tokens.extend([t for t in toks if t is not None and t != ""])
        return " ".join(new_tokens)

    def _build_spans_and_annotate(self, prune_spans=True):
        self._set_spans(prune_spans)
        return self._annotate()

    def get_annotated_prog_str(self, tokenized_style=False, apply_bpe=False, prune_spans = True):
        annotated = self.annotated_prog_str if self.annotated_prog_str else self._build_spans_and_annotate(prune_spans)
        if tokenized_style:
            annotated = self.reformat_indiv_special_tokens(annotated)
            tokenized = self.tokenize_code(annotated)
            annotated = " ".join(tokenized)
        if apply_bpe:
            annotated = self.apply_bpe(annotated)
        return annotated

    @staticmethod
    def remove_spans(annotated_prog):
        assert isinstance(annotated_prog, str) or isinstance(annotated_prog, list)
        if type(annotated_prog) == str:
            annotated_prog = annotated_prog.split()
        prog = " ".join([c for c in annotated_prog if c not in SpanAnnotator.special_char2char.keys()]).strip()
        return prog

    @staticmethod
    def add_one_to_all_elements_greater(unit, span_list: List[Tuple[int, int]]):
        """
        alt implementation of add_one_to_all_elements_greater
        def add_one_to_all_elements_greater(unit, span_list: List[List[int]]):
        for span in span_list:
            for i, pos in enumerate(span):
                if pos > unit:
                    span[i] += 1
        return span_list; used within annotate to help incrementally build the tree-annotated program string
        """
        return [(start + (start >= unit), end + (end >= unit)) for start, end in span_list]

    @staticmethod
    def test_span_iter(span_iter: Iterable[Tuple[int, int]], prog_str: str = None, fh: TextIOWrapper = STDOUT):
        """
        Test a span iterator. Ensures that no two spans "overlap," a child can be a child of its parent, but if not a child
        then it cannot overlap. The invariant it tested in the second if statement.
        Naive O(N**2) implementation, but it is possible it could be further optimized.
        """
        correct = True
        already_tested_set = set()
        for beg_i, end_i in span_iter:
            for beg_j, end_j in span_iter:
                set_id = ((beg_i, end_i), (beg_j, end_j))
                if set_id in already_tested_set:
                    continue
                already_tested_set.add(set_id)
                if (beg_i > beg_j) and (end_i > end_j) and (beg_i < end_j):
                    correct = False
                    print("overlapping spans for span i ({} {}), span j ({} {})".format(beg_i, end_i, beg_j, end_j),
                          file=fh)
                    print("overlapping spans for span i ({} {}), span j ({} {})".format(beg_i, end_i, beg_j, end_j),
                          file=fh)
                    if prog_str:
                        span_i = color_substr_with_span(prog_str, (beg_i, end_i), backgr_color="red")
                        span_j = color_substr_with_span(prog_str, (beg_j, end_j), backgr_color="yellow")
                        print(f"span i:\n{span_i},\nspan j:\n{span_j}",
                              file=fh)
        return correct

    @staticmethod
    def prune_spans(prog_str, span_list):
        """
        Given a list of spans, prune the spans that only encompass a single space-delimited token.
        """
        return [span for span in span_list if len(prog_str[span[0]:span[1] + 1].split()) != 1]

    @staticmethod
    def replace_spec(tree_string: str):
        for k, v in SpanAnnotator.special_char2char.items():
            tree_string = tree_string.replace(k, v)
        return tree_string

    @staticmethod
    def parse_parens_tree_string_split_terminals(tree_string: str, offset: int = 0,
                                                 id_counter: int = 0, add_elipsis_for_children=True):
        """
        parse a string of the form "(a (b c) (d e))" and return a Node object
        with the name "a" and children nodes "b c", "d e"
        """
        ## postprocess by replacing special chars with original parens
        assert tree_string[offset] == SpanAnnotator.BEGIN_PARENS_CHAR
        # print("offset", offset, "right, is: ", tree_string[offset:])
        offset += 1
        node = Node(node_id=id_counter)
        id_counter += 1
        buffer = ""
        expanding = False
        while True:
            ## recursively parse the subtree
            if tree_string[offset] == SpanAnnotator.BEGIN_PARENS_CHAR:
                if add_elipsis_for_children:
                    node.increment_name("...")
                child, offset, id_counter = SpanAnnotator.parse_parens_tree_string_split_terminals(tree_string,
                                                                                                   offset, id_counter)
                # add pointer to parent for graphviz
                child.add_parent(node)
                node.add_child(child)
            ## add terminal and reset the buffer
            elif tree_string[offset] == " " and buffer.strip() != "":
                expanding = True
                # buffer = replace_spec(buffer, special_char2char)
                child_terminal = Node(node_id=id_counter, name=buffer, parent=node)
                id_counter += 1
                buffer = ""
                offset += 1
                node.add_child(child_terminal)
            ## return / go up
            elif tree_string[offset] == SpanAnnotator.END_PARENS_CHAR:
                # buffer = replace_spec(buffer, special_char2char)
                if expanding and buffer.strip() != "":
                    child_terminal = Node(node_id=id_counter, name=buffer, parent=node)
                    buffer = ""
                    id_counter += 1
                    node.add_child(child_terminal)
                node.add_name(buffer)
                offset += 1
                return node, offset, id_counter
            ## continue adding
            else:
                buffer += tree_string[offset]
                offset += 1
                if offset > len(tree_string):
                    raise Exception("unexpected end of string")
    @staticmethod
    def parse_parens_tree_string(tree_string: str, offset: int =0, id_counter: int = 0, add_elipsis_for_children=True):
        """
        parse a string of the form "(a (b c) (d e))" and return a Node object
        with the name "a" and children nodes "b c", "d e"
        """
        ## postprocess by replacing special chars with original parens
        assert tree_string[offset] == SpanAnnotator.BEGIN_PARENS_CHAR
        offset += 1
        node = Node(node_id=id_counter)
        id_counter += 1
        while tree_string[offset] != SpanAnnotator.END_PARENS_CHAR:
            if tree_string[offset] == SpanAnnotator.BEGIN_PARENS_CHAR:
                if add_elipsis_for_children:
                    node.increment_name("...")
                child, offset, id_counter = SpanAnnotator.parse_parens_tree_string(tree_string, offset, id_counter)
                # add pointer to parent for graphviz
                child.add_parent(node)
                node.add_child(child)
            else:
                node.increment_name(tree_string[offset])
                offset += 1
        offset += 1
        return node, offset, id_counter



