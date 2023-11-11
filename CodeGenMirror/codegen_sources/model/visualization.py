import glob
import pdb

from codegen_sources.model.translate import *
from copy import deepcopy
import fastBPE


import re
from typing import List, Tuple, Dict
import seaborn
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from ansi2html import Ansi2HTMLConverter
from IPython.core.display import display, HTML
import subprocess
import argparse
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(description='Visualize the model')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the model')
    parser.add_argument('--output_path', type=str, default=None, help='Name of the model')
    parser.add_argument('--input_path', type=str, default='data/transcoder_evaluation_gfg/java',
                        help='Path to the input file')
    parser.add_argument('--bpe_path', type=str, default='data/bpe/cpp-java-python/vocab', help='Path to the BPE model')
    parser.add_argument('--number_to_process', type=int, default=50, help='Number of examples to process')
    return parser.parse_args()


### copied from other modules

import re
REMOVE_JAVA_MAIN = re.compile(r"//TOFILL[\s\S]*")
REMOVE_JAVA_COMMENTS = re.compile(r"//.*")
REMOVE_JAVA_IMPORTS = re.compile(r"import.*")
REMOVE_STATIC = re.compile('^static')


def process_java_prog_string(java_prog: str, remove_imports = False):
    s = re.sub(REMOVE_JAVA_MAIN, "", java_prog)
    s = re.sub(REMOVE_JAVA_COMMENTS, "", s)
    if remove_imports:
        s = re.sub(REMOVE_JAVA_IMPORTS, "", s)
    s = s.strip()
    s += "}"
    return s

def get_java_method_file(pth_to_java: str, remove_imports = False):
    with open(pth_to_java) as fh:
        java_prog = fh.read()
    return process_java_prog_string(java_prog, remove_imports)


REMOVE_CLASS_PRE = re.compile("public class[^{]*{")
REMOVE_CLASS_POST = re.compile("}$")

def extract_method_from_java_class(java_prog, remove_static = False):
    java_prog = re.sub(REMOVE_CLASS_PRE, "", java_prog)
    java_prog = re.sub(REMOVE_CLASS_POST, "", java_prog)
    if remove_static:
        java_prog = re.sub(REMOVE_STATIC, "", java_prog.strip())
    return java_prog.strip()


class Div:
    conv = Ansi2HTMLConverter()
    soup = BeautifulSoup()

    def __init__(self):
        self.div = self.soup.new_tag("div")

    def add_ansi_text(self, text: str):
        full_html = self.conv.convert(text, full=True)
        new_soup = BeautifulSoup(full_html, "html.parser")
        pre_tag = new_soup.find("pre")
        self.div.append(pre_tag)

    def add_img(self, img_path: str):
        #         assert os.path.exists(img_path)
        img_tag = self.soup.new_tag('img', src=img_path)
        self.div.append(img_tag)

    def add_br(self):
        br_tag = self.soup.new_tag('br')
        self.div.append(br_tag)

    def get_div(self):
        return self.div


class HtmlWriter:
    conv = Ansi2HTMLConverter()
    template = """
    <!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN" "http://www.w3.org/TR/html4/loose.dtd">
    <html>
    <head>
    <meta content="text/html; charset=utf-8" http-equiv="Content-Type"/>
    <title></title>
    <style type="text/css">
       .ansi2html-content { display: inline; white-space: pre-wrap; word-wrap: break-word; }
    .body_foreground { color: #AAAAAA; }
    .body_background { background-color: #000000; }
    .body_foreground > .bold,.bold > .body_foreground, body.body_foreground > pre > .bold { color: #FFFFFF; font-weight: normal; }
    .inv_foreground { color: #000000; }
    .inv_background { background-color: #FFFFFF; }
    .ansi30 { color: #000316; }
    .ansi41 { background-color: #aa0000; }
    .ansi43 { background-color: #dbc01e; }
    .ansi46 { background-color: #00aaaa; }
    img {
  width: 1000px;
  height: auto;
    }​
    </style>
    </head>
    <body class="inv_foreground inv_background" style="font-size: 100%;">
    </body>
    </html>"""
    ## img width: 90%
    def __init__(self):
        self.soup = BeautifulSoup(self.template, 'html.parser')

    def add_text_and_img(self, ansi_text, img, image_last=False):
        new_div_wrapper = Div()
        if image_last:
            new_div_wrapper.add_ansi_text(ansi_text)
            new_div_wrapper.add_br()
            new_div_wrapper.add_img(img)
        else:
            new_div_wrapper.add_img(img)
            new_div_wrapper.add_br()
            new_div_wrapper.add_ansi_text(ansi_text)
        self.soup.body.append(new_div_wrapper.get_div())

    def add_text_and_dot_graph(self, ansi_text, dot_graph_str, out_path, image_last=False):
        dirname = os.path.dirname(out_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        self.save_dot_to_png(dot_graph_str, out_path)
        out_path = re.search("images.*", out_path).group()
        #         print("new out path for the html is {}".format(out_path))
        self.add_text_and_img(ansi_text, out_path, image_last=image_last)

    def add_text(self, ansi_text):
        new_div_wrapper = Div()
        new_div_wrapper.add_ansi_text(ansi_text)
        self.soup.body.append(new_div_wrapper.get_div())

    def add_image(self, img):
        new_div_wrapper = Div()
        new_div_wrapper.add_img(img)
        self.soup.body.append(new_div_wrapper.get_div())

    def get_html_string(self):
        return str(self.soup)

    def get_soup(self):
        return self.soup

    def ipython_display(self):
        display(HTML(str(self.soup)))

    def save_dot_to_png(self, dot_graph: str, out_path: str):
        tmp_path = os.path.splitext(out_path)[0] + ".dot"
        with open(tmp_path, "w") as fh:
            fh.write(dot_graph)
        with open(out_path, "w") as fh:
            subprocess.run(["dot", "-Tpng", tmp_path], check=True, stdout=fh)
        os.remove(tmp_path)


class Visualizer(Translator):

    def __init__(self, model_path, BPE_path, device_no=0):
        self.model_path = model_path
        self.BPE_path = BPE_path
        self.device_no = device_no
        super().__init__(self.model_path, self.BPE_path, self.device_no)

    def load_model(self):
        # reload model
        reloaded = torch.load(self.model_path, map_location="cpu")
        # change params of the reloaded model so that it will
        # relaod its own weights and not the MLM or DOBF pretrained model
        reloaded["params"]["reload_model"] = ",".join([self.model_path] * 2)
        reloaded["params"]["lgs_mapping"] = ""
        reloaded["params"]["reload_encoder_for_decoder"] = False
        self.reloaded_params = AttrDict(reloaded["params"])

        # build dictionary / update parameters
        self.dico = Dictionary(
            reloaded["dico_id2word"], reloaded["dico_word2id"], reloaded["dico_counts"]
        )
        # self.device_no = device_no
        assert self.reloaded_params.n_words == len(self.dico)
        assert self.reloaded_params.bos_index == self.dico.index(BOS_WORD)
        assert self.reloaded_params.eos_index == self.dico.index(EOS_WORD)
        assert self.reloaded_params.pad_index == self.dico.index(PAD_WORD)
        assert self.reloaded_params.unk_index == self.dico.index(UNK_WORD)
        assert self.reloaded_params.mask_index == self.dico.index(MASK_WORD)

        # build model / reload weights (in the build_model method)
        print("Building model with device number {}...".format(self.device_no), flush=True, file=sys.stderr)
        gpu = torch.cuda.is_available() #if self.device_no is None else True
        encoder, decoder = build_model(self.reloaded_params, self.dico, gpu=gpu,
                                       device_no=self.device_no)
        self.encoder = encoder[0]
        self.decoder = decoder[0]
        if self.device_no is not None:
            assert str(next(self.encoder.parameters()).device) == f"cuda:{self.device_no}"
            assert str(next(self.decoder.parameters()).device) == f"cuda:{self.device_no}"
        else:
            self.encoder.cuda()
            self.decoder.cuda()
        self.encoder.eval()
        self.decoder.eval()

    def reset_model(self):
        self.load_model()

    def get_bpe_model(self):
        return fastBPE.fastBPE(os.path.abspath(self.pth_to_vocab))

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
        device="cuda:0",
        detokenize=True,
        max_tokens=None,
        length_penalty=0.5,
    ):
        if hasattr(self, "device_no"):
            assert f"cuda:{self.device_no}" == device, f"device number mismatch, cuda:{self.device_no} != {device}"
            print("Using device number in translate {}...".format(self.device_no), flush=True, file=sys.stderr)

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

            inp_bpe_tokens = []
            for i in range(x1.shape[1]):
                wid = [self.dico[x1[j, i].item()] for j in range(len(x1))]
                inp_bpe_tokens.append(deepcopy(wid))

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
            out_bpe_tok = []
            out_bpe_tokens = [] # this one for recordin bpe elements
            for i in range(x2.shape[1]):
                wid = [self.dico[x2[j, i].item()] for j in range(len(x2))][1:]
                out_bpe_tokens.append(deepcopy(wid))
                wid = wid[: wid.index(EOS_WORD)] if EOS_WORD in wid else wid
                if getattr(self.reloaded_params, "roberta_mode", False):
                    out_bpe_tok.append(restore_roberta_segmentation_sentence(" ".join(wid)))
                else:
                    out_bpe_tok.append(" ".join(wid).replace("@@ ", ""))
            # if not detokenize:
            #     return out_bpe_tok
            results = []
            for t in out_bpe_tok:
                results.append(detokenizer(t))
                # tok is tgt bpe tokens
            return results, out_bpe_tokens, inp_bpe_tokens


    def bpe_tokens_to_joined(self, bpe_tokens: List[str]):
        return " ".join(bpe_tokens).replace("@@ ", "").split()

    def clean_bpe_tokens(self, bpe_tokens: List[str]):
        return [t.replace("@@", "") for t in bpe_tokens]


    def get_joined_to_bpe_spans(self, joined_tok_stream: List[str], bpe_tok_stream: List[str], test = False) -> Dict[int, Tuple[int]]:
        joined_tok_idx_to_span = {}
        bpe_ctr = 0
        for i, token in enumerate(joined_tok_stream):
            tok_ctr = 0
            span_start = bpe_ctr
            next_bpe_tok = bpe_tok_stream[bpe_ctr]
            while ((tok_ctr + len(next_bpe_tok)) < len(token)):
                tok_ctr += len(next_bpe_tok)
                bpe_ctr += 1
                next_bpe_tok = bpe_tok_stream[bpe_ctr]
            tok_ctr += len(next_bpe_tok)
            assert len(token) == tok_ctr, f"""
                                                bpe tokens didn't match input tokens, 
                                                dict: {joined_tok_idx_to_span}, 
                                                current token {token}, token index {i} and bpe index {bpe_ctr}, 
                                                tok ctr {tok_ctr}, and len(token) {len(token)}
                                                current span is {' '.join(bpe_tok_stream[span_start:bpe_ctr + 1])}
                                                next bpe tok {next_bpe_tok}"""

            joined_tok_idx_to_span[i] = (span_start, bpe_ctr)
            bpe_ctr += 1
        if test:
            for joined_tok_idx, (split_tok_begin, split_tok_end) in joined_tok_idx_to_span.items():
                orig_tok = joined_tok_stream[joined_tok_idx]
                if split_tok_end < len(bpe_tok_stream):
                    joined_bpe_toks = "".join(bpe_tok_stream[split_tok_begin:split_tok_end + 1])
                # todo: check if this else statement is even necessary
                else:
                    joined_bpe_toks = "".join(bpe_tok_stream[split_tok_begin:])
                assert orig_tok == joined_bpe_toks, f"""
                                                        joined tokens don't match original tokens, 
                                                        dict: {joined_tok_idx_to_span}, 
                                                        current token {orig_tok}, token index {joined_tok_idx} and bpe index {split_tok_begin}, 
                                                        tok ctr {len(orig_tok)}, and len(token) {len(orig_tok)},
                                                        current span is {' '.join(bpe_tok_stream[split_tok_begin:split_tok_end + 1])}
                                                        next bpe tok {bpe_tok_stream[split_tok_end]}"""

        return joined_tok_idx_to_span


    def get_attention_maps(self):
        assert len(self.decoder.scores) == 6
        running_tensor = None
        for attention_tensor in self.decoder.scores.values():
            #     attention_tensor = attention_tensor.unsqueeze(0)
            if running_tensor is None:
                running_tensor = attention_tensor
            else:
                running_tensor = torch.cat((running_tensor, attention_tensor), dim=0)
        return running_tensor


    def aggregate_attention_maps_layer_head(self, attention_tensor: torch.Tensor):
        # (Layers X Heads X Output X Input)
        attention_avg = attention_tensor.mean(dim=0).mean(dim=0)
        return attention_avg

    def attention_maps_to_list(self, attention_tensor: torch.Tensor):
        # (Layers X Heads X Output X Input)
        layers_list = list(attention_tensor)
        layers_heads_list = [list(layer) for layer in layers_list]
        return layers_heads_list


    def matrix_slice_avg(self, span_begin, span_end, attn_mtx, axis=0, keepdims=True):
        assert axis in (0, 1)
        if axis == 0:
            avg_vector = attn_mtx[span_begin:span_end + 1, :].mean(axis=0, keepdims=keepdims)
        else:
            avg_vector = attn_mtx[:, span_begin:span_end + 1].mean(axis=1, keepdims=keepdims)
        return avg_vector


    def aggregate_attn_matrix_span(self, tok_id_to_span_dict, attn_matrix, axis):
        aggregate_mtx = None
        for tok_id, (span_begin, span_end) in tok_id_to_span_dict.items():
            avg_vector = self.matrix_slice_avg(span_begin, span_end, attn_matrix, axis=axis, keepdims=True)
            if aggregate_mtx is None:
                aggregate_mtx = avg_vector
            else:
                aggregate_mtx = torch.cat((aggregate_mtx, avg_vector), dim=axis)
        return aggregate_mtx

    def draw(self, data, x, y, ax):
        return seaborn.heatmap(data,
                               # xticklabels=None, yticklabels=None,
                               square=False,
                               vmin=0.0, vmax=1.0,
                               cbar=False, ax=ax)
        # return seaborn.heatmap(data,
        #                        xticklabels=x, square=False, yticklabels=y, vmin=0.0, vmax=1.0,
        #                        cbar=False, ax=ax)

    def make_heatmap(self, attn_map, input_tokens, output_tokens, output_path: str, figsize = (20,20), dpi = 100):
        fig, ax = plt.subplots(figsize=figsize)
        # input is x axis (axis 1) and output is y axis (axis 0)
        svm = self.draw(attn_map, input_tokens, output_tokens, ax)
        figure = svm.get_figure()
        figure.savefig(output_path, dpi=dpi)
        plt.clf()
        plt.close()

    def make_layer_heatmaps(self, attn_maps: List[torch.tensor], input_tokens, output_tokens, output_path: str,
                          layer_number: int, figsize = (20,20), dpi = 100):
        """
        :param path_to_logs: path to the log files
        :param measure: ppl, comp_acc, bleu
        :param exp2metrics: dictionary of experiment name to dict of metrics (e.g. ppl, bleu, comp_acc) holding metrics for that experiment
        :return: None
        Plots all experiments for the given measure on one plot
        """
        layer_number = str(layer_number + 1)
        baseneme, suffix = os.path.splitext(output_path)
        n_maps = len(attn_maps)
        fig, ax_list = plt.subplots(1, n_maps, figsize=(5*n_maps, 5))
        for i, (attn_map, input_tokens, output_tokens, ax) in enumerate(zip(attn_maps, input_tokens, output_tokens, ax_list)):
            self.draw(attn_map.cpu(), input_tokens, output_tokens, ax)
            ax.set_title(f"Attention plot for layer {layer_number} head {i+1}")
        fig.suptitle(f"Attention heatmaps for layer {layer_number}")
        outpath = baseneme + f"_layer_{layer_number}.{suffix}"
        fig.savefig(outpath, dpi=dpi)
        plt.clf()
        plt.close()
        return outpath

    def make_all_heatmaps(self, all_layer_attn_maps: List[List[torch.tensor]], input_tokens, output_tokens, output_path: str, figsize = (20,20), dpi = 100):
        """
        :param path_to_logs: path to the log files
        :param measure: ppl, comp_acc, bleu
        :param exp2metrics: dictionary of experiment name to dict of metrics (e.g. ppl, bleu, comp_acc) holding metrics for that experiment
        :return: None
        Plots all experiments for the given measure on one plot
        """
        output_paths = []
        for i, layer_maps in enumerate(all_layer_attn_maps):
            out_path = self.make_layer_heatmaps(layer_maps, input_tokens, output_tokens, output_path,
                                     layer_number=i, figsize=figsize, dpi=dpi)
            output_paths.append(out_path)
        return output_paths


    def make_output_and_attn_map_for_program(self, input_prog, lang1, lang2, output_path, test = False):
        self.reset_model()
        output_string, input_bpe_tokens, output_bpe_tokens = self.translate_with_transcoder(input_prog, lang1, lang2)

        output_string, input_bpe_tokens, output_bpe_tokens = output_string[0], input_bpe_tokens[0], output_bpe_tokens[0]

        joined_inp_tokens = self.bpe_tokens_to_joined(input_bpe_tokens)
        joined_out_tokens = self.bpe_tokens_to_joined(output_bpe_tokens)

        input_bpe_tokens = self.clean_bpe_tokens(input_bpe_tokens)
        output_bpe_tokens = self.clean_bpe_tokens(output_bpe_tokens)

        inp_tok2bpe_span = self.get_joined_to_bpe_spans(joined_inp_tokens, input_bpe_tokens, test=test)
        out_tok2bpe_span = self.get_joined_to_bpe_spans(joined_out_tokens, output_bpe_tokens, test=test)

        attn_maps = self.get_attention_maps()
        avg_attn_map = self.aggregate_attention_maps_layer_head(attn_maps)

        inp_agg_attn_map = self.aggregate_attn_matrix_span(inp_tok2bpe_span, avg_attn_map, axis=1)
        agg_attn_map = self.aggregate_attn_matrix_span(out_tok2bpe_span, inp_agg_attn_map, axis=0)
        self.make_heatmap(agg_attn_map, joined_inp_tokens, joined_out_tokens, output_path)
        return output_string

    def make_outputs_and_all_attn_maps_for_program(self, input_prog, lang1, lang2, output_path, test = False):
        self.reset_model()
        output_string, input_bpe_tokens, output_bpe_tokens = self.translate_with_transcoder(input_prog, lang1, lang2)

        output_string, input_bpe_tokens, output_bpe_tokens = output_string[0], input_bpe_tokens[0], output_bpe_tokens[0]

        joined_inp_tokens = self.bpe_tokens_to_joined(input_bpe_tokens)
        joined_out_tokens = self.bpe_tokens_to_joined(output_bpe_tokens)

        input_bpe_tokens = self.clean_bpe_tokens(input_bpe_tokens)
        output_bpe_tokens = self.clean_bpe_tokens(output_bpe_tokens)

        inp_tok2bpe_span = self.get_joined_to_bpe_spans(joined_inp_tokens, input_bpe_tokens, test=test)
        out_tok2bpe_span = self.get_joined_to_bpe_spans(joined_out_tokens, output_bpe_tokens, test=test)

        attn_maps = self.get_attention_maps()
        attn_maps_list_of_lists = self.attention_maps_to_list(attn_maps)
        replacement_list = []
        for layer_list in attn_maps_list_of_lists:
            replacement_layer_list = []
            for head_attn_map in layer_list:
                inp_agg_attn_map = self.aggregate_attn_matrix_span(inp_tok2bpe_span, head_attn_map, axis=1)
                agg_attn_map = self.aggregate_attn_matrix_span(out_tok2bpe_span, inp_agg_attn_map, axis=0)
                replacement_layer_list.append(agg_attn_map)
            replacement_list.append(replacement_layer_list)
        attn_maps_list_of_lists = replacement_list
        output_paths = self.make_all_heatmaps(attn_maps_list_of_lists, joined_inp_tokens, joined_out_tokens, output_path)
        return output_string, output_paths


    def make_html_from_java_prog(self, input_java_prog_path: str, output_html_dir):
        java_prog = get_java_method_file(input_java_prog_path, remove_imports=True)
        java_prog = extract_method_from_java_class(java_prog, remove_static=True)
        print(java_prog)

        problem_name = os.path.splitext(os.path.basename(input_java_prog_path))[0].lower()
        output_image_dir = os.path.join(output_html_dir, "images")
        if not os.path.exists(output_html_dir):
            os.makedirs(output_image_dir)
        output_attn_map_png_path = os.path.join(output_image_dir, problem_name + ".png")
        output_html_path = os.path.join(output_html_dir, problem_name + ".html")
        out, output_paths = self.make_outputs_and_all_attn_maps_for_program(java_prog, "java", "python",
                                                              output_attn_map_png_path, test=True)
        html_string = f"Original java program was:\n{java_prog}\n{'-'*40}\n{'-'*40}\n{'-'*40}\n"
        html_string += f"Output python program was:\n{out}\n{'-'*40}\n{'-'*40}\n{'-'*40}\n"
        html_string += f"And the attention plots are below:"
        html_writer = HtmlWriter()
        ## html needs the relative path
        # html_png_path = os.path.join("images", problem_name + ".png")
        html_writer.add_text(html_string)
        for output_png_path in output_paths:
            local_path = output_png_path[output_png_path.index("images"):]
            html_writer.add_image(local_path)
        with open(output_html_path, "w") as fh:
            fh.write(html_writer.get_html_string())

def main():
    args = get_args()
    model_path = args.model_path
    input_path = args.input_path
    bpe_path = args.bpe_path
    output_path = args.output_path
    n = args.number_to_process

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(os.path.join(output_path, "images")):
        os.makedirs(os.path.join(output_path, "images"))

    model = Visualizer(model_path, bpe_path)
    files = glob.glob(f"{input_path}/*java")
    files = files[:n]
    for i, file in enumerate(tqdm(files)):
        model.make_html_from_java_prog(file, output_path)


if __name__ == "__main__":
    main()


