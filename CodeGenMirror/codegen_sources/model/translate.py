# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Translate sentences from the input stream.
# The model will be faster if sentences are sorted by length.
# Input sentences must have the same tokenization and BPE codes than the ones used in the model.
#
import json
import os
import argparse
import pathlib
from pathlib import Path
import sys
from tqdm import tqdm
import torch
from codegen_sources.model.src.logger import create_logger
from codegen_sources.preprocessing.lang_processors.cpp_processor import CppProcessor
from codegen_sources.preprocessing.lang_processors.java_processor import JavaProcessor
from codegen_sources.preprocessing.lang_processors.python_processor import (
    PythonProcessor,
)
from codegen_sources.preprocessing.lang_processors.lang_processor import LangProcessor
from codegen_sources.preprocessing.bpe_modes.fast_bpe_mode import FastBPEMode
from codegen_sources.preprocessing.bpe_modes.roberta_bpe_mode import RobertaBPEMode
from codegen_sources.model.src.data.dictionary import (
    Dictionary,
    BOS_WORD,
    EOS_WORD,
    PAD_WORD,
    UNK_WORD,
    MASK_WORD,
)
from codegen_sources.model.src.utils import restore_roberta_segmentation_sentence
from codegen_sources.model.src.model import build_model
from codegen_sources.model.src.utils import AttrDict, TREE_SITTER_ROOT

SUPPORTED_LANGUAGES = ["cpp", "java", "python"]
lang2ending = {
    "python": ".py",
    "cpp": ".cpp",
    "java": ".java",
}

logger = create_logger(None, 0)

lang2func = {
    "python": "def",
    "cpp": ".cpp",
    "java": "public static",
}


def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1:
            return
        yield start
        start += len(sub)  # use start += 1 to find overlapping matches


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Translate programs")

    # model
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(
            Path(__file__)
            .parents[3]
            .joinpath(
                "generate_candidate-and-translate/dump/models/TransCoder_model_2.pth"
            )
        ),
        help="Model path",
    )
    parser.add_argument(
        "--src_lang",
        type=str,
        default="python",
        help=f"Source language, should be either {', '.join(SUPPORTED_LANGUAGES[:-1])} or {SUPPORTED_LANGUAGES[-1]}",
    )
    parser.add_argument(
        "--tgt_lang",
        type=str,
        default="java",
        help=f"Target language, should be either {', '.join(SUPPORTED_LANGUAGES[:-1])} or {SUPPORTED_LANGUAGES[-1]}",
    )
    parser.add_argument(
        "--BPE_path",
        type=str,
        default=str(
            Path(__file__).parents[2].joinpath("data/bpe/cpp-java-python/codes")
        ),
        help="Path to BPE codes.",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=1,
        help="Beam size. The beams will be printed in order of decreasing likelihood.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="input path",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="input directory path, model will translate every file in the "
        "directory that is of the source language",
    )
    parser.add_argument(
        "--output_dir", type=str, default="dump", help="output translated file dir"
    )
    parser.add_argument(
        "--translate_per_function",
        type=bool,
        default=False,
        help="whether to translate one function at a time (multiple calls)",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="whether to print verbose output",
    )
    return parser


class Translator:
    def __init__(self, model_path, BPE_path, device_no=None):
        # reload model
        reloaded = torch.load(model_path, map_location="cpu")
        # change params of the reloaded model so that it will
        # relaod its own weights and not the MLM or DOBF pretrained model
        reloaded["params"]["reload_model"] = ",".join([model_path] * 2)
        reloaded["params"]["lgs_mapping"] = ""
        reloaded["params"]["reload_encoder_for_decoder"] = False
        self.reloaded_params = AttrDict(reloaded["params"])

        # build dictionary / update parameters
        self.dico = Dictionary(
            reloaded["dico_id2word"], reloaded["dico_word2id"], reloaded["dico_counts"]
        )
        self.device_no = device_no
        assert self.reloaded_params.n_words == len(self.dico)
        assert self.reloaded_params.bos_index == self.dico.index(BOS_WORD)
        assert self.reloaded_params.eos_index == self.dico.index(EOS_WORD)
        assert self.reloaded_params.pad_index == self.dico.index(PAD_WORD)
        assert self.reloaded_params.unk_index == self.dico.index(UNK_WORD)
        assert self.reloaded_params.mask_index == self.dico.index(MASK_WORD)

        # build model / reload weights (in the build_model method)
        print(
            "Building model with device number {}...".format(device_no),
            flush=True,
            file=sys.stderr,
        )
        gpu = torch.cuda.is_available() if device_no is None else False
        encoder, decoder = build_model(
            self.reloaded_params, self.dico, gpu=gpu, device_no=self.device_no
        )
        self.encoder = encoder[0]
        self.decoder = decoder[0]
        if self.device_no is not None:
            assert (
                str(next(self.encoder.parameters()).device) == f"cuda:{self.device_no}"
            )
            assert (
                str(next(self.decoder.parameters()).device) == f"cuda:{self.device_no}"
            )
        else:
            if torch.cuda.is_available():
                self.encoder.cuda()
                self.decoder.cuda()
        self.encoder.eval()
        self.decoder.eval()

        # reload bpe
        if getattr(self.reloaded_params, "roberta_mode", False):
            self.bpe_model = RobertaBPEMode()
        else:
            self.bpe_model = FastBPEMode(
                codes=os.path.abspath(BPE_path), vocab_path=None
            )

    def translate_with_transcoder(
        self,
        input,
        lang1,
        lang2,
        suffix1="_sa",
        suffix2="_sa",
        n=1,
        beam_size=1,
        sample_temperature=None,
        device=None,
        detokenize=True,
        max_tokens=None,
        length_penalty=0.5,
        verbose=False,
    ):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if hasattr(self, "device_no"):
            assert f"cuda:{self.device_no}" == device or (
                device == "cpu" and self.device_no is None
            )
            if verbose:
                print(
                    "Using device number in translate {}...".format(self.device_no),
                    flush=True,
                    file=sys.stderr,
                )

        # Build language processors
        assert lang1 in {"python", "java", "cpp"}, lang1
        assert lang2 in {"python", "java", "cpp"}, lang2
        src_lang_processor = LangProcessor.processors[lang1](
            root_folder=TREE_SITTER_ROOT
        )
        tokenizer = src_lang_processor.tokenize_code
        tgt_lang_processor = LangProcessor.processors[lang2](
            root_folder=TREE_SITTER_ROOT
        )
        detokenizer = tgt_lang_processor.detokenize_code

        lang1 += suffix1
        lang2 += suffix2

        assert (
            lang1 in self.reloaded_params.lang2id.keys()
        ), f"{lang1} should be in {self.reloaded_params.lang2id.keys()}"
        assert (
            lang2 in self.reloaded_params.lang2id.keys()
        ), f"{lang2} should be in {self.reloaded_params.lang2id.keys()}"

        with torch.no_grad():

            lang1_id = self.reloaded_params.lang2id[lang1]
            lang2_id = self.reloaded_params.lang2id[lang2]

            # Convert source code to ids
            tokens = [t for t in tokenizer(input)]
            # print(f"Tokenized {params.src_lang} function:")
            # print(tokens)
            tokens = self.bpe_model.apply_bpe(" ".join(tokens)).split()
            tokens = ["</s>"] + tokens + ["</s>"]
            input = " ".join(tokens)
            if max_tokens is not None and len(input.split()) > max_tokens:
                logger.info(
                    f"Ignoring long input sentence of size {len(input.split())}"
                )
                return [f"Error: input too long: {len(input.split())}"] * max(
                    n, beam_size
                )

            # Create torch batch
            len1 = len(input.split())
            len1 = torch.LongTensor(1).fill_(len1).to(device)
            x1 = torch.LongTensor([self.dico.index(w) for w in input.split()]).to(
                device
            )[:, None]
            langs1 = x1.clone().fill_(lang1_id)

            # Encode
            # print("Encoding on device passed in as {} and encoder is on {} and x1 is on {}".format(
            #     device, next(self.encoder.parameters()).device, x1.device), flush=True, file=sys.stderr)
            enc1 = self.encoder("fwd", x=x1, lengths=len1, langs=langs1, causal=False)
            enc1 = enc1.transpose(0, 1)
            if n > 1:
                enc1 = enc1.repeat(n, 1, 1)
                len1 = len1.expand(n)

            # Decode
            max_len = self.reloaded_params.max_len
            # print("Decoding on device passed in as {} and decoder is on {} and enc1 is on {}".format(
            #     device, next(self.decoder.parameters()).device, enc1.device), flush=True, file=sys.stderr)
            if beam_size == 1:
                x2, len2 = self.decoder.generate(
                    enc1,
                    len1,
                    lang2_id,
                    max_len=max_len,
                    sample_temperature=sample_temperature,
                )
            else:
                x2, len2, _ = self.decoder.generate_beam(
                    enc1,
                    len1,
                    lang2_id,
                    max_len=max_len,
                    early_stopping=False,
                    length_penalty=length_penalty,
                    beam_size=beam_size,
                )

            # Convert out ids to text
            tok = []
            for i in range(x2.shape[1]):
                wid = [self.dico[x2[j, i].item()] for j in range(len(x2))][1:]
                wid = wid[: wid.index(EOS_WORD)] if EOS_WORD in wid else wid
                if getattr(self.reloaded_params, "roberta_mode", False):
                    tok.append(restore_roberta_segmentation_sentence(" ".join(wid)))
                else:
                    tok.append(" ".join(wid).replace("@@ ", ""))
            if not detokenize:
                return tok
            results = []
            for t in tok:
                results.append(detokenizer(t))
            return results


if __name__ == "__main__":
    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    ###### DEBUG
    # params.input = "/home/tangpihai/Project/generate_candidate-and-translate/data/transcoder_evaluation_gfg/extract_largest_control/python/ADD_TWO_NUMBERS_WITHOUT_USING_ARITHMETIC_OPERATORS.py"
    # params.output_dir = "/home/tangpihai/Project/generate_candidate-and-translate/dump/no_extract/java/python-java/translation_output_Trans"

    # check parameters
    assert os.path.isfile(
        params.model_path
    ), f"The path to the model checkpoint is incorrect: {params.model_path}"
    assert (
        params.input_dir is None
        or params.input is None
        or os.path.isdir(params.input_dir)
        or os.path.isfile(params.input)
    ), f"The path to the input file/dir path is incorrect: input-{params.input}, input_dir-{params.input_dir}"
    assert os.path.isdir(
        params.output_dir
    ), f"The path to the output directory is incorrect: {params.output_dir}"
    assert os.path.isfile(
        params.BPE_path
    ), f"The path to the BPE tokens is incorrect: {params.BPE_path}"
    assert (
        params.src_lang in SUPPORTED_LANGUAGES
    ), f"The source language should be in {SUPPORTED_LANGUAGES}."
    assert (
        params.tgt_lang in SUPPORTED_LANGUAGES
    ), f"The target language should be in {SUPPORTED_LANGUAGES}."

    # Initialize translator
    translator = Translator(params.model_path, params.BPE_path)

    if params.input:
        input_dir = pathlib.Path(params.input).parent
        input_files = [pathlib.Path(params.input).name]
    else:
        input_dir = params.input_dir
        input_files = [
            f
            for f in os.listdir(params.input_dir)
            if f.endswith(lang2ending[params.src_lang])
        ]

    output_dict = {}
    for input_file in tqdm(input_files):
        input_file_path = f"{input_dir}/{input_file}"
        # read input code from stdin
        src_sent = []
        input = open(input_file_path).read().strip()

        if params.verbose:
            print(f"Input {params.src_lang} function:")
            print(input)
        with torch.no_grad():
            if params.translate_per_function:
                output = ""
                func_location = list(find_all(input, lang2func[params.src_lang]))
                func_location.append(len(input))
                for i in range(len(func_location) - 1):
                    input_subprogram = input[func_location[i] : func_location[i + 1]]
                    output_subprogram = translator.translate_with_transcoder(
                        input_subprogram,
                        lang1=params.src_lang,
                        lang2=params.tgt_lang,
                        beam_size=params.beam_size,
                        verbose=params.verbose,
                    )
                    output += output_subprogram[0] + "\n"
            else:
                output = translator.translate_with_transcoder(
                    input_subprogram,
                    lang1=params.src_lang,
                    lang2=params.tgt_lang,
                    beam_size=params.beam_size,
                    verbose=params.verbose,
                )

        if params.verbose:
            print(f"Translated {params.tgt_lang} function:")
            # for out in output:
            print("=" * 20)
            print(output)

        out_filename = input_file.replace(
            lang2ending[params.src_lang], lang2ending[params.tgt_lang]
        )
        with open(f"{params.output_dir}/{out_filename}", "w") as f:
            f.write(output)
        program_id = input_file.replace(lang2ending[params.src_lang], "")
        output_dict[program_id] = {
            "lang": params.tgt_lang,
            "function": output,
        }

    agg_outut_filename = pathlib.Path(params.output_dir).parent.joinpath(
        "translation_output_Trans.json"
    )
    with open(agg_outut_filename, "w") as f:
        json.dump(output_dict, f)
