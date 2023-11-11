import pdb
import warnings
from typing import List, Dict, Tuple
from pathlib import Path
from codegen_sources.preprocessing.bpe_modes.fast_bpe_mode import FastBPEMode, BPEMode
import re

DEFAULT_BPE_DIR = Path(__file__).parents[2].joinpath('data/bpe').joinpath("cpp-java-python")


class Program(object):
    default_bpe_codes = str(DEFAULT_BPE_DIR.joinpath('codes'))
    default_bpe_vocab = str(DEFAULT_BPE_DIR.joinpath('vocab'))


    def __init__(self, input_program = None, obfuscate=False, bpe_model=None):

        if obfuscate:
            input_program = self.obfuscate_code(input_program)
        self.input_program = input_program
        self.bpe_model = bpe_model

    def tokenize_code(self, input_program):
        # breakpoint()
        return self.processor.tokenize_code(input_program)

    def detokenize_code(self, input_program):
        return self.processor.detokenize_code(input_program)

    def obfuscate_code(self, input_program: str, apply_bpe = False):
        obfuscated = self.processor.obfuscate_code(input_program)[0]
        if apply_bpe:
            obfuscated = self.bpe_model.apply_bpe(obfuscated)
        return obfuscated

    def apply_bpe(self, code):
        if not self.bpe_model:
            warnings.warn("No BPE model found. Using default BPE model.")
            self.bpe_model = FastBPEMode(codes = self.default_bpe_codes, vocab_path = self.default_bpe_vocab)
        assert type(code) == str and self.bpe_model, "code must be a string and bpe_model must be set"
        return self.bpe_model.apply_bpe(code)

    @staticmethod
    def bpe_tokens_to_joined(bpe_tokens: List[str]):
        return Program.bpe_string_to_joined(" ".join(bpe_tokens))

    @staticmethod
    def bpe_string_to_joined(bpe_string: str):
        return bpe_string.replace("@@ ", "")

    @staticmethod
    def clean_bpe_tokens(bpe_tokens: List[str]):
        return [t.replace("@@", "") for t in bpe_tokens]

    def _build_tokenized_program(self):
        input_program = self.remove_class_from_code(self.input_program)
        self.tokenized_program_list = self.tokenize_code(input_program)
        self.tokenized_program_string = " ".join(self.tokenized_program_list)
        return None

    def get_tokenized_program_string(self):
        if not hasattr(self, "tokenized_program_string"):
            self._build_tokenized_program()
        return self.tokenized_program_string

    def get_tokenized_program_list(self):
        if not hasattr(self, "tokenized_program_list"):
            self._build_tokenized_program()
        return self.tokenized_program_list

    def _build_bpe_tokens(self):
        tokenized = self.get_tokenized_program_string()
        self.bpe_tokens_list = self.apply_bpe(tokenized).split()
        self.bpe_tokens_string = " ".join(self.bpe_tokens_list)
        return None

    def get_bpe_tokens_string(self):
        if not hasattr(self, "bpe_tokens_string"):
            self._build_bpe_tokens()
        return self.bpe_tokens_string

    def get_bpe_tokens_list(self):
        if not hasattr(self, "bpe_tokens_list"):
            self._build_bpe_tokens()
        return self.bpe_tokens_list

    def get_joined_to_bpe_spans(self, joined_tok_stream: List[str], bpe_tok_stream: List[str], check = True) -> Dict[int, Tuple[int]]:
        joined_tok_idx_to_span = {}
        bpe_idx_to_joined_idx = {}
        bpe_ctr = 0
        for i, token in enumerate(joined_tok_stream):
            ## TODO: this does not work if there is a whitespace between 2 tokens
            ## it would be better to use BPE tokens with @@
            ## A hacky fix is to just strip whitespace first
            token = re.sub("\s+", "", token)
            tok_ctr = 0
            span_start = bpe_ctr
            next_bpe_tok = bpe_tok_stream[bpe_ctr]
            while ((tok_ctr + len(next_bpe_tok)) < len(token)):
                tok_ctr += len(next_bpe_tok)
                bpe_ctr += 1
                next_bpe_tok = bpe_tok_stream[bpe_ctr]
            tok_ctr += len(next_bpe_tok)
            if not len(token) == tok_ctr:
                err = f"""bpe tokens didn't match input tokens, 
                            dict: {joined_tok_idx_to_span}, 
                            current token {token}, token index {i} and bpe index {bpe_ctr}, 
                            tok ctr {tok_ctr}, and len(token) {len(token)}
                            current span is {' '.join(bpe_tok_stream[span_start:bpe_ctr + 1])}
                            next bpe tok {next_bpe_tok}"""
                print(err)
                raise RuntimeError
                # pdb.set_trace()
            assert len(token) == tok_ctr, f"""
                                                bpe tokens didn't match input tokens, 
                                                dict: {joined_tok_idx_to_span}, 
                                                current token {token}, token index {i} and bpe index {bpe_ctr}, 
                                                tok ctr {tok_ctr}, and len(token) {len(token)}
                                                current span is {' '.join(bpe_tok_stream[span_start:bpe_ctr + 1])}
                                                next bpe tok {next_bpe_tok}"""

            # map joined token index to bpe span
            joined_tok_idx_to_span[i] = (span_start, bpe_ctr)
            # map bpe index (every bpe token in-between) to joined the token index
            for bpe_idx in range(span_start, bpe_ctr + 1):
                bpe_idx_to_joined_idx[bpe_idx] = i
            bpe_ctr += 1
        if check:
            for joined_tok_idx, (split_tok_begin, split_tok_end) in joined_tok_idx_to_span.items():
                orig_tok = joined_tok_stream[joined_tok_idx]
                if split_tok_end < len(bpe_tok_stream):
                    joined_bpe_toks = "".join(bpe_tok_stream[split_tok_begin:split_tok_end + 1])
                # todo: check if this else statement is even necessary
                else:
                    joined_bpe_toks = "".join(bpe_tok_stream[split_tok_begin:])
                assert orig_tok == joined_bpe_toks

        return joined_tok_idx_to_span, bpe_idx_to_joined_idx


    def build_token_to_bpe_token_maps(self, check = False):
        # breakpoint()
        tokenized_program = self.get_tokenized_program_list()
        bpe_tokens = self.get_bpe_tokens_list()
        cleaned_bpe_tokens = self.clean_bpe_tokens(bpe_tokens) # remove @@ from bpe tokens
        self.joined_tok_idx_to_bpe_span, self.bpe_idx_to_joined_idx = self.get_joined_to_bpe_spans(
            tokenized_program,
            cleaned_bpe_tokens,
            check
        )
        return None

    def get_token_to_bpe_index_span(self, token_idx: int) -> Tuple[int, int]:
        if not hasattr(self, "joined_tok_idx_to_bpe_span"):
            self.build_token_to_bpe_token_maps()
        return self.joined_tok_idx_to_bpe_span[token_idx]

    def get_token_to_bpe_indices(self, token_idx: int) -> List[int]:
        if not hasattr(self, "joined_tok_idx_to_bpe_span"):
            self.build_token_to_bpe_token_maps()
        if token_idx >= len(self.joined_tok_idx_to_bpe_span):
            pdb.set_trace()
        span = self.joined_tok_idx_to_bpe_span[token_idx]
        return list(range(span[0], span[1] + 1))

    def get_bpe_index_to_token_index(self, bpe_idx: int) -> int:
        if not hasattr(self, "bpe_idx_to_joined_idx"):
            self.build_token_to_bpe_token_maps()
        return self.bpe_idx_to_joined_idx[bpe_idx]

    def get_num_bpe_tokens(self) -> int:
        return len(self.get_bpe_tokens_list())
    
    @staticmethod
    def remove_class_from_code(code: str) -> str:
        whitespace_pattern = "^\s*"
        if (re.search(f"public[{whitespace_pattern}]+class[{whitespace_pattern}]+Main", code) or
                re.search(f"public[{whitespace_pattern}]+class[{whitespace_pattern}]+CLASS_0", code)):
            split = code.split("{")
            joined = "{".join(split[1:])
            split = joined.split("}")
            code = "}".join(split[:-1])
        return code.strip()

    @staticmethod
    def parse_program_stmt(prog_list: List[str]) -> List[List[str]]:
        """ 
        Parse a program (stream of tokens) into a list of statements (stream of tokens)
        """
        raise NotImplementedError
    
    def parse_program_stmt_bpe(self, bpe_tokens: List[str]) -> List[List[str]]:
        prog_str = self.bpe_tokens_to_joined(bpe_tokens)
        lexer_tokens = prog_str.split(" ")
        prog_stmt_list = self.parse_program_stmt(lexer_tokens)
        prog_stmt_bpe_list = [self.apply_bpe(" ".join(stmt)).split() for stmt in prog_stmt_list]
        return prog_stmt_bpe_list
        
    







