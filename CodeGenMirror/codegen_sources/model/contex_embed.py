import glob
import os.path
import pdb

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from codegen_sources.model.visualization import *
from typing import List, Tuple, Dict, Any
import numpy as np
from scipy.spatial import distance_matrix
from data.ast.PythonProgram import PythonProgram
from data.ast.JavaProgram import JavaProgram
import ot
from copy import deepcopy
import warnings

epsilon = 1e-1
max_iter = 10000
tol = 1e-3
verbose = 1

def last_index(list_, element):
    return len(list_) - 1 - list_[::-1].index(element)

def remove_all_after_last_elem(list_, elem):
    return list_[:last_index(list_, elem)+1]

def sinkhorn(c, p, q, epsilon, max_iter, tol, verbose):
    K = np.exp(-c / epsilon)

    if verbose >= 2:
        print('[SINKHORN] c = ')
        print(c)
        print(f'[SINKHORN] epsilon = {epsilon}')
        print('[SINKHORN] K = exp(-c / epsilon) = ')
        print(K)

    b = q
    b_old = b
    convg = False
    for k in range(max_iter):
        a = p / (np.matmul(K, b))
        b = q / (np.matmul(K.T, a))
        b_diff = np.linalg.norm(b - b_old)
        b_old = b

        if verbose >= 2:
            print(f'[SINKHORN] iter = {k + 1}, b = {b}, |b - b_old| = {b_diff}')

        if b_diff < tol:
            convg = True
            break

    if verbose >= 1:
        if convg:
            print(f'[SINKHORN] converges under tolerance {tol}'
                  f' at iteration {k + 1}')
        else:
            print(f'[SINKHORN] fails to converge under tolerance {tol}'
                  f' after {k + 1} iterations. Last b_diff = {b_diff}')

    return np.expand_dims(a, axis=1) * K * np.expand_dims(b, axis=0)


def get_args():
    parser = argparse.ArgumentParser(description='Visualize the model')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the model')
    parser.add_argument('--output_path', type=str, default=None, help='Name of the model')
    parser.add_argument('--bpe_path', type=str, default='data/bpe/cpp-java-python/vocab', help='Path to the BPE model')
    parser.add_argument('--number_to_process', type=int, default=50, help='Number of examples to process')
    parser.add_argument('--temp', type=float, default=1.0, help='Temperature for scaling the softmax')
    parser.add_argument('--aggregate_stmt', type=bool, default=True, help='aggregate embeddings on stmt level')
    return parser.parse_args()


class Embedding_Visualizer(Visualizer):

    def get_contex_embed(self,
                         input: str,
                         lang1,
                         suffix1="_sa",
                         device="cuda:0",
                         max_tokens=None,
                         aggregate_stmt=True,
                         ):
        # pdb.set_trace()

        if hasattr(self, "device_no"):
            assert f"cuda:{self.device_no}" == device, f"device number mismatch, cuda:{self.device_no} != {device}"
            print("Using device number in translate {}...".format(self.device_no), flush=True, file=sys.stderr)

        # Build language processors
        assert lang1 in {"python", "java", "cpp"}, lang1
        src_lang_processor = LangProcessor.processors[lang1](
            root_folder=TREE_SITTER_ROOT
        )
        detokenizer = src_lang_processor.detokenize_code
        tokenizer = src_lang_processor.tokenize_code

        lang1 += suffix1

        assert (
                lang1 in self.reloaded_params.lang2id.keys()
        ), f"{lang1} should be in {self.reloaded_params.lang2id.keys()}"

        with torch.no_grad():

            lang1_id = self.reloaded_params.lang2id[lang1]

            # Convert source code to ids
            tokens = [t for t in tokenizer(input)]
            # print(f"Tokenized {params.src_lang} function:")
            # print(tokens)
            tokens = self.bpe_model.apply_bpe(" ".join(tokens)).split()
            if "python" in lang1:
                tokens = remove_all_after_last_elem(tokens, "DEDENT")
            tokens = ["</s>"] + tokens + ["</s>"]
            input = " ".join(tokens)
            if max_tokens is not None and len(input.split()) > max_tokens:
                logger.info(
                    f"Ignoring long input sentence of size {len(input.split())}"
                )
                raise ValueError(
                    f"Input sentence of size {len(input.split())} is too long"
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
            layer_cache = {}
            self.encoder.layer_cache = layer_cache
            enc1 = self.encoder("fwd", x=x1, lengths=len1, langs=langs1, causal=False)
            detokenized = detokenizer(" ".join(inp_bpe_tokens[0][1:-1]).replace("@@ ", ""))
            for k, v in layer_cache.items():
                layer_cache[k] = v.squeeze(0)

            enc1 = enc1.squeeze(1)
            inp_bpe_tokens = inp_bpe_tokens[0]

            if aggregate_stmt:
                # pdb.set_trace()
                enc1, stmt_chunks, _ = self.aggregate_attn_matrix_bpe_2_stmt(inp_bpe_tokens[1:-1], lang1, enc1, axis=0)
                for k, v in layer_cache.items():
                    layer_cache[k], _ = self.aggregate_attn_matrix_bpe_2_stmt(inp_bpe_tokens[1:-1], lang1, v, axis=0)
                inp_bpe_tokens = stmt_chunks
            return enc1, inp_bpe_tokens, detokenized, layer_cache

    def get_dual_embed(self,
                       input1,
                       input2,
                       lang1,
                       lang2,
                       aggregate_stmt=True,
                       suffix1="_sa",
                       suffix2="_sa",
                       device="cuda:0",
                       max_tokens=None,
                       ):
        # pdb.set_trace()
        embed_1, inp_1_bpe, inp_1_detok, layer_cache_1 = self.get_contex_embed(input1, lang1, suffix1, device,
                                                                               max_tokens, aggregate_stmt)
        # pdb.set_trace()
        embed_2, inp_2_bpe, inp_2_detok, layer_cache_2 = self.get_contex_embed(input2, lang2, suffix2, device,
                                                                               max_tokens, aggregate_stmt)
        # pdb.set_trace()
        return (embed_1, inp_1_bpe, inp_1_detok, layer_cache_1), (embed_2, inp_2_bpe, inp_2_detok, layer_cache_2)

    @staticmethod
    def dot_product(embed_1, embed_2):
        return torch.matmul(embed_1, embed_2.transpose(0, 1))

    def dot_product_softmax(self, embed_1, embed_2, temperature=1.0):
        a_dot_b = self.dot_product(embed_1, embed_2) / temperature
        a_dot_b = F.softmax(a_dot_b, dim=0)
        return a_dot_b

    @staticmethod
    def cosine_similarity_matrix(embed_1, embed_2):
        a_dot_b = Embedding_Visualizer.dot_product(embed_1, embed_2)
        a_norm = torch.norm(embed_1, dim=1).unsqueeze(1)
        b_norm = torch.norm(embed_2, dim=1).unsqueeze(1)
        return a_dot_b / (a_norm * b_norm.transpose(0, 1))

    def cosine_similarity_matrix_softmax(self, embed_1, embed_2, temperature=1.0):
        a_dot_b = self.cosine_similarity_matrix(
            embed_1, embed_2
        )  # (batch_size, batch_size)
        a_dot_b = F.softmax(a_dot_b / temperature, dim=0)
        return a_dot_b

    def get_sinkhorn_from_cosine_diff(self, embed_1, embed_2, eps = 0.1):
        cosine_diff = 1 - self.cosine_similarity_matrix(embed_1, embed_2)
        # T = sinkhorn(c=cosine_diff.cpu().numpy(),
        #                          p=np.ones(cosine_diff.shape[0]) / cosine_diff.shape[0],
        #                          q=np.ones(cosine_diff.shape[1]) / cosine_diff.shape[1],
        #                          epsilon=epsilon,
        #                          max_iter=max_iter,
        #                          tol=tol,
        #                          verbose=verbose)
        T = ot.sinkhorn(np.ones(cosine_diff.shape[0]) / cosine_diff.shape[0],
                        np.ones(cosine_diff.shape[1]) / cosine_diff.shape[1],
                        cosine_diff.cpu().numpy(),
                        eps)
        return torch.tensor(T) #/ max(T.max(), 1e-10)

    def get_sinkhorn_from_euclidian_diff(self, embed_1, embed_2, eps = 0.1):
        euclidian_diff = distance_matrix(embed_1.cpu().numpy(), embed_2.cpu().numpy())
        T = ot.sinkhorn(np.ones(euclidian_diff.shape[0]) / euclidian_diff.shape[0],
                        np.ones(euclidian_diff.shape[1]) / euclidian_diff.shape[1],
                        euclidian_diff,
                        eps)
        # T = sinkhorn(c=euclidian_diff, p=np.ones(euclidian_diff.shape[0]) / euclidian_diff.shape[0],
        #              q=np.ones(euclidian_diff.shape[1]) / euclidian_diff.shape[1], epsilon=epsilon,
        #              max_iter=max_iter, tol=tol, verbose=verbose)
        return torch.tensor(T) #/ max(T.max(), 1e-10)

    # TODO re-factor this so that we re-normalize the bpe token weights by other meaningful features of mass on lexer tokens
    # TODO
    def get_ot_from_cosine_diff(self, embed_1, embed_2):
        cosine_diff = 1 - self.cosine_similarity_matrix(embed_1, embed_2)
        T = ot.emd(np.ones(cosine_diff.shape[0]) / cosine_diff.shape[0],
               np.ones(cosine_diff.shape[1]) / cosine_diff.shape[1],
                       cosine_diff.cpu().numpy())
        return torch.tensor(T)

    def get_ot_from_euclidian_diff(self, embed_1, embed_2):
        euclidian_diff = distance_matrix(embed_1.cpu().numpy(), embed_2.cpu().numpy())
        T = ot.emd(np.ones(euclidian_diff.shape[0]) / euclidian_diff.shape[0],
               np.ones(euclidian_diff.shape[1]) / euclidian_diff.shape[1],
                       euclidian_diff)
        return torch.tensor(T)

    def get_ot_from_mtx(self, c):
        T = ot.emd(np.ones(c.shape[0]) / c.shape[0],
               np.ones(c.shape[1]) / c.shape[1],
                       c)
        return torch.tensor(T)

    def get_ot_from_mtx_hist(self, src, tgt, c):
        src = src.double()
        tgt = tgt.double()
        if not sum(src) == 1:
            warnings.warn("src torch must sum to 1, got {}".format(sum(src)))
        if not sum(tgt) == 1:
            warnings.warn("tgt torch must sum to 1, got {}".format(sum(tgt)))
        src = src.cpu().numpy()
        tgt = tgt.cpu().numpy()
        if not sum(src) == 1:
            warnings.warn("src np must sum to 1, got {}".format(sum(src)))
        if not sum(tgt) == 1:
            warnings.warn("tgt np must sum to 1, got {}".format(sum(tgt)))
        T = ot.emd(src, tgt, c)
        return torch.tensor(T)

    def draw(self, data, x, y, ax):
        return seaborn.heatmap(data,
                               xticklabels=x, square=False, yticklabels=y, vmin=0.0, vmax=1.0,
                               cbar=True, ax=ax)

    def matrix_slice_avg(self, span_begin, span_end, attn_mtx, axis=0, keepdims=True):
        # deprecated
        assert axis in (0, 1)
        if axis == 0:
            avg_vector = attn_mtx[span_begin:span_end + 1, :].mean(axis=0, keepdims=keepdims)
        else:
            avg_vector = attn_mtx[:, span_begin:span_end + 1].mean(axis=1, keepdims=keepdims)
        return avg_vector

    @staticmethod
    def matrix_slice_reduce(span_begin, span_end, attn_mtx, axis=0, keepdims=True, operator = "avg",):
        assert axis in (0, 1)
        assert operator in ("avg", "mean", "average", "max", "min", "sum", "prod", "product")
        if operator in ("avg", "mean", "average"):
            _reduce_func = torch.mean
        elif operator == "max":
            _reduce_func = torch.max
        elif operator == "min":
            _reduce_func = torch.min
        elif operator == "sum":
            _reduce_func = torch.sum
        elif operator in ("prod", "product"):
            # do log sum exp
            if axis == 0:
                avg_vector = torch.log(attn_mtx[span_begin:span_end + 1, :]).sum(axis=0, keepdims=keepdims)
                avg_vector = torch.exp(avg_vector)
            else:
                avg_vector = torch.log(attn_mtx[:, span_begin:span_end + 1]).sum(axis=1, keepdims=keepdims)
                avg_vector = torch.exp(avg_vector)
            return avg_vector
        if axis == 0:
            avg_vector = _reduce_func(attn_mtx[span_begin:span_end + 1, :], axis=0, keepdims=keepdims)
        else:
            avg_vector = _reduce_func(attn_mtx[:, span_begin:span_end + 1], axis=1, keepdims=keepdims)
        return avg_vector

    @staticmethod
    def aggregate_attn_matrix_span(tok_id_to_span_dict, attn_matrix, axis, operator="avg"):
        aggregate_mtx = None
        for tok_id, (span_begin, span_end) in tok_id_to_span_dict.items():
            avg_vector = Embedding_Visualizer.matrix_slice_reduce(span_begin, span_end, attn_matrix, axis=axis, keepdims=True, operator=operator)
            if aggregate_mtx is None:
                aggregate_mtx = avg_vector
            else:
                aggregate_mtx = torch.cat((aggregate_mtx, avg_vector), dim=axis)
        return aggregate_mtx

    @staticmethod
    def aggregate_attn_matrix_bpe_2_stmt(bpe_tok_stream: List[str], lang, attn_matrix, axis, operator="sum"):
        """
        :param bpe_tok_stream: will be bpe tokens without </s> and </s>
        """
        assert "</s>" not in bpe_tok_stream, f"</s> should not be in bpe_tok_stream {bpe_tok_stream}"
        if lang == "python" or lang == "python_sa":
            program_processor = PythonProgram
        elif lang == "java" or lang == "java_sa":
            program_processor = JavaProgram
        else:
            raise ValueError("Unsupported language: {}".format(lang))
        processor = program_processor(" ".join(bpe_tok_stream).replace("@@ ", ""))
        prog_stmt_bpe = processor.parse_program_stmt_bpe(bpe_tok_stream)
        tok_id_to_span_dict = {0: (0, 0)}
        bpe_tok_stream = 1 # start at 1 because of </s>
        for stmt_id, stmt_bpe in enumerate(prog_stmt_bpe):
            tok_id_to_span_dict[stmt_id+1] = (bpe_tok_stream, bpe_tok_stream + len(stmt_bpe) - 1)
            bpe_tok_stream += len(stmt_bpe)
        stmt_id += 2
        tok_id_to_span_dict[stmt_id] = (bpe_tok_stream, bpe_tok_stream) # for final </s>
        # join bpe toks at each level
        prog_stmt_chunks = ["</s>"]
        for stmt_toks in prog_stmt_bpe:
            prog_stmt_chunks.append(" ".join(stmt_toks).replace("@@ ", ""))
        prog_stmt_chunks.append("</s>")
        # calculate mass in each chunk
        prog_stmt_mass = torch.tensor([1] + [len(stmt_toks) for stmt_toks in prog_stmt_bpe] + [1]).double()
        prog_stmt_mass/=prog_stmt_mass.sum()
        return Embedding_Visualizer.aggregate_attn_matrix_span(tok_id_to_span_dict, attn_matrix, axis=axis, operator=operator), prog_stmt_chunks, prog_stmt_mass

    def make_annotated_attn_map(self, attn_map, output_tokens, input_tokens, output_path: str, figsize=(20, 20),
                                dpi=100):
        fig, ax = plt.subplots(figsize=figsize)
        # input is x axis (axis 1) and output is y axis (axis 0)
        attn_map = np.flip(attn_map.numpy(), axis=0)
        input_tokens = input_tokens[::-1]
        svm = self.draw(attn_map, output_tokens, input_tokens, ax)
        svm.set_xticklabels(svm.get_xticklabels(), rotation=90)
        svm.set_yticklabels(svm.get_yticklabels(), rotation=0)
        figure = svm.get_figure()
        # plt.yticks(rotation=45)
        # plt.xticks(rotation=45)
        figure.subplots_adjust(bottom=0.30, left=0.30)
        figure.savefig(output_path, dpi=dpi)
        plt.clf()
        plt.close()

    def make_dual_embed_map(self, input_tokens, output_tokens, inp_lang, out_lang, output_path: str, temp=1.0,
                            figsize=(20, 20), dpi=100):
        embed_1, embed_2 = self.get_dual_embed(input_tokens, output_tokens, inp_lang, out_lang)
        attn_map = self.dot_product_softmax(embed_1[0], embed_2[0], temp)  # [0] is the embedding
        # pdb.set_trace()
        self.make_annotated_attn_map(attn_map.cpu(), embed_2[1], embed_1[1], output_path, figsize,
                                     dpi)  # [1] is the bpe tokens
        return embed_1[2], embed_2[2]  # [2] is the detokenized tokens

    def make_multi_layer_dual_embed_map(self, input_tokens, output_tokens, inp_lang, out_lang, output_path: str,
                                        temp=1.0, aggregate_stmt = True, figsize=(20, 20), dpi=100):
        embed_1, embed_2 = self.get_dual_embed(input_tokens, output_tokens, inp_lang, out_lang, False)
        attn_map = self.dot_product_softmax(embed_1[0], embed_2[0], temp)  # [0] is the embedding

        # attn_map = self.cosine_similarity_matrix_softmax(embed_1[0], embed_2[0], temp)  # [0] is the embedding
        # pdb.set_trace()
        self.make_annotated_attn_map(attn_map.cpu(), embed_2[1], embed_1[1], output_path, figsize,
                                     dpi)  # [1] is the bpe tokens
        layer_cache_1 = embed_1[3]
        layer_cache_2 = embed_2[3]
        # pdb.set_trace()
        for k in layer_cache_1:
            layer_cache_1[k] = layer_cache_1[k].cpu()
            layer_cache_2[k] = layer_cache_2[k].cpu()
            attn_map = self.dot_product_softmax(layer_cache_1[k], layer_cache_2[k], temp)
            # attn_map = self.cosine_similarity_matrix_softmax(layer_cache_1[k], layer_cache_2[k], temp)
            tgt_toks = deepcopy(embed_2[1])
            src_toks = deepcopy(embed_1[1])
            if aggregate_stmt:
                attn_map, tgt_toks, _ = self.aggregate_attn_matrix_bpe_2_stmt(tgt_toks[1:-1],
                                                                           out_lang,
                                                                           attn_map,
                                                                           axis=1,
                                                                           operator="mean")
                attn_map, src_toks, _ = self.aggregate_attn_matrix_bpe_2_stmt(src_toks[1:-1],
                                                                           inp_lang,
                                                                           attn_map,
                                                                           axis=0,
                                                                           operator="mean")
                attn_map/=attn_map.sum(dim=0)

            new_output_path = output_path.replace(".png", "_" + str(k) + ".png")
            self.make_annotated_attn_map(attn_map.cpu(), tgt_toks, src_toks, new_output_path, figsize, dpi)

            cosine_sim = self.cosine_similarity_matrix(layer_cache_1[k], layer_cache_2[k])

            new_output_path = output_path.replace(".png", ".cosine_sim_" + str(k) + ".png")
            self.make_annotated_attn_map(cosine_sim.cpu(), embed_2[1], embed_1[1], new_output_path, figsize, dpi)


            earth_movers_distance_cosine = self.get_ot_from_cosine_diff(layer_cache_1[k], layer_cache_2[k])
            tgt_toks = deepcopy(embed_2[1])
            src_toks = deepcopy(embed_1[1])
            if aggregate_stmt:
                earth_movers_distance_cosine, tgt_toks, _ = self.aggregate_attn_matrix_bpe_2_stmt(tgt_toks[1:-1],
                                                                                               out_lang,
                                                                                               earth_movers_distance_cosine,
                                                                                               axis=1,
                                                                                               operator="sum")
                earth_movers_distance_cosine, src_toks, _ = self.aggregate_attn_matrix_bpe_2_stmt(src_toks[1:-1],
                                                                                               inp_lang,
                                                                                               earth_movers_distance_cosine,
                                                                                               axis=0,
                                                                                               operator="sum")
            earth_movers_distance_cosine /= earth_movers_distance_cosine.sum(dim=0)
            new_output_path = output_path.replace(".png", ".cosine_earth_movers_distance_" + str(k) + ".png")
            self.make_annotated_attn_map(earth_movers_distance_cosine.cpu(),
                                         tgt_toks,
                                         src_toks,
                                         new_output_path,
                                         figsize,
                                         dpi)

            if aggregate_stmt:
                agg_cosine_sim, tgt_toks, tgt_masses = self.aggregate_attn_matrix_bpe_2_stmt(tgt_toks[1:-1],
                                                                       out_lang,
                                                                       cosine_sim,
                                                                       axis=1,
                                                                       operator="mean")
                agg_cosine_sim, src_toks, src_masses = self.aggregate_attn_matrix_bpe_2_stmt(src_toks[1:-1],
                                                                       inp_lang,
                                                                       agg_cosine_sim,
                                                                       axis=0,
                                                                       operator="mean")

                ot_plan_agg = self.get_ot_from_mtx_hist(src_masses, tgt_masses, agg_cosine_sim)
                ot_plan_agg /= ot_plan_agg.sum(dim=0)
                new_output_path = output_path.replace(".png", ".cosine_earth_movers_distanceagg_" + str(k) + ".png")
                self.make_annotated_attn_map(ot_plan_agg.cpu(),
                                             tgt_toks,
                                             src_toks,
                                             new_output_path,
                                             figsize,
                                             dpi)

            earth_movers_distance_euclidian = self.get_ot_from_euclidian_diff(layer_cache_1[k],
                                                                                    layer_cache_2[k])
            tgt_toks = deepcopy(embed_2[1])
            src_toks = deepcopy(embed_1[1])
            if aggregate_stmt:
                earth_movers_distance_euclidian, tgt_toks, _ = self.aggregate_attn_matrix_bpe_2_stmt(tgt_toks[1:-1],
                                                                                                  out_lang,
                                                                                                  earth_movers_distance_euclidian,
                                                                                                  axis=1,
                                                                                                  operator="sum")
                earth_movers_distance_euclidian, src_toks, _ = self.aggregate_attn_matrix_bpe_2_stmt(src_toks[1:-1],
                                                                                                  inp_lang,
                                                                                                  earth_movers_distance_euclidian,
                                                                                                  axis=0,
                                                                                                  operator="sum")
            earth_movers_distance_euclidian /= earth_movers_distance_euclidian.sum(dim=0)
            new_output_path = output_path.replace(".png", ".euclidian_earth_movers_distance_" + str(k) + ".png")
            self.make_annotated_attn_map(earth_movers_distance_euclidian, tgt_toks, src_toks, new_output_path,
                                         figsize, dpi)

            if aggregate_stmt:
                euclidian_dist = torch.from_numpy(distance_matrix(layer_cache_1[k].cpu().numpy(), layer_cache_2[k].cpu().numpy()))
                agg_euclidian_sim, tgt_toks, tgt_masses = self.aggregate_attn_matrix_bpe_2_stmt(tgt_toks[1:-1],
                                                                       out_lang,
                                                                       euclidian_dist,
                                                                       axis=1,
                                                                       operator="mean")
                agg_euclidian_sim, src_toks, src_masses = self.aggregate_attn_matrix_bpe_2_stmt(src_toks[1:-1],
                                                                       inp_lang,
                                                                       agg_euclidian_sim,
                                                                       axis=0,
                                                                       operator="mean")
                ot_plan_agg = self.get_ot_from_mtx_hist(src_masses, tgt_masses, agg_euclidian_sim)
                ot_plan_agg /= ot_plan_agg.sum(dim=0)
                new_output_path = output_path.replace(".png", ".euclidian_earth_movers_distanceagg_" + str(k) + ".png")
                self.make_annotated_attn_map(ot_plan_agg.cpu(),
                                             tgt_toks,
                                             src_toks,
                                             new_output_path,
                                             figsize,
                                             dpi)

            earth_movers_distance_cosine = self.get_sinkhorn_from_cosine_diff(layer_cache_1[k], layer_cache_2[k])
            tgt_toks = deepcopy(embed_2[1])
            src_toks = deepcopy(embed_1[1])
            if aggregate_stmt:
                earth_movers_distance_cosine, tgt_toks, _ = self.aggregate_attn_matrix_bpe_2_stmt(tgt_toks[1:-1],
                                                                                               out_lang,
                                                                                               earth_movers_distance_cosine,
                                                                                               axis=1,
                                                                                               operator="sum")
                earth_movers_distance_cosine, src_toks, _ = self.aggregate_attn_matrix_bpe_2_stmt(src_toks[1:-1],
                                                                                               inp_lang,
                                                                                               earth_movers_distance_cosine,
                                                                                               axis=0,
                                                                                               operator="sum")
            earth_movers_distance_cosine /= earth_movers_distance_cosine.sum(dim=0)
            new_output_path = output_path.replace(".png", ".cosine_earth_movers_distancesink_" + str(k) + ".png")
            self.make_annotated_attn_map(earth_movers_distance_cosine, tgt_toks, src_toks, new_output_path,
                                         figsize, dpi)
            earth_movers_distance_euclidian = self.get_sinkhorn_from_euclidian_diff(layer_cache_1[k],
                                                                                    layer_cache_2[k])
            tgt_toks = deepcopy(embed_2[1])
            src_toks = deepcopy(embed_1[1])
            if aggregate_stmt:
                earth_movers_distance_euclidian, tgt_toks, _ = self.aggregate_attn_matrix_bpe_2_stmt(tgt_toks[1:-1],
                                                                                                  out_lang,
                                                                                                  earth_movers_distance_euclidian,
                                                                                                  axis=1,
                                                                                                  operator="sum")
                earth_movers_distance_euclidian, src_toks, _ = self.aggregate_attn_matrix_bpe_2_stmt(src_toks[1:-1],
                                                                                                  inp_lang,
                                                                                                  earth_movers_distance_euclidian,
                                                                                                  axis=0,
                                                                                                  operator="sum")
            earth_movers_distance_euclidian /= earth_movers_distance_euclidian.sum(dim=0)
            new_output_path = output_path.replace(".png", ".euclidian_earth_movers_distancesink_" + str(k) + ".png")
            self.make_annotated_attn_map(earth_movers_distance_euclidian, tgt_toks, src_toks, new_output_path,
                                         figsize, dpi)

        return embed_1[2], embed_2[2]  # [2] is the detokenized tokens

    def make_html_from_java_prog(self, problem_name, output_html_dir, lang1_prog, lang2_prog, temp=1.0,
                                 aggregate_stmt = True, lang1="java", lang2="python"):
        output_image_dir = os.path.join(output_html_dir, "images")
        if not os.path.exists(output_html_dir):
            os.makedirs(output_image_dir)
        output_attn_map_png_path = os.path.join(output_image_dir, problem_name + ".png")
        output_html_path = os.path.join(output_html_dir, problem_name + ".html")
        detok_lang1, detok_lang_2 = self.make_multi_layer_dual_embed_map(lang1_prog, lang2_prog, lang1, lang2,
                                                                         output_attn_map_png_path, temp, aggregate_stmt)

        html_string = f"Original {lang1} program was:\n{detok_lang1}\n{'-' * 40}\n{'-' * 40}\n{'-' * 40}\n"
        html_string += f"Output {lang2} program was:\n{detok_lang_2}\n{'-' * 40}\n{'-' * 40}\n{'-' * 40}\n"
        html_string += f"And the attention plots are below:"
        html_writer = HtmlWriter()
        ## html needs the relative path
        # html_png_path = os.path.join("images", problem_name + ".png")
        html_writer.add_text(html_string)
        local_path = output_attn_map_png_path[output_attn_map_png_path.index("images"):]
        html_writer.add_text("Encoder Output Dot Product Similarity Map")
        html_writer.add_image(local_path)
        # pdb.set_trace()
        base_path = os.path.splitext(output_attn_map_png_path)[0]
        layer_cache_images = glob.glob(base_path + "_*")
        layer_cache_images_indexed = []

        cosine_sim_images = glob.glob(base_path + ".cosine_sim_*")
        cosine_sim_images_indexed = {}

        cosine_earth_movers_images = glob.glob(base_path + ".cosine_earth_movers_distance_*")
        cosine_earth_movers_images_indexed = {}

        euclidian_earth_movers_images = glob.glob(base_path + ".euclidian_earth_movers_distance_*")
        euclidian_earth_movers_images_indexed = {}

        cosine_earth_movers_sink_images = glob.glob(base_path + ".cosine_earth_movers_distancesink_*")
        cosine_earth_movers_sink_images_indexed = {}

        euclidian_earth_movers_sink_images = glob.glob(base_path + ".euclidian_earth_movers_distancesink_*")
        euclidian_earth_movers_sink_images_indexed = {}

        cosine_earth_movers_aggregate_images = glob.glob(base_path + ".cosine_earth_movers_distanceagg_*")
        cosine_earth_movers_aggregate_images_indexed = {}

        euclidian_earth_movers_aggregate_images = glob.glob(base_path + ".euclidian_earth_movers_distanceagg_*")
        euclidian_earth_movers_aggregate_images_indexed = {}

        for img_pth in layer_cache_images:
            print(img_pth)
            index = re.search("(?<=_)(\d+)(?=.png)", img_pth).group(1)
            layer_cache_images_indexed.append((int(index), img_pth))

        for img_pth in cosine_sim_images:
            print(img_pth)
            index = re.search("(?<=.cosine_sim_)(\d+)(?=.png)", img_pth).group(1)
            cosine_sim_images_indexed[int(index)] = img_pth

        for img_pth in cosine_earth_movers_images:
            print(img_pth)
            index = re.search("(?<=.cosine_earth_movers_distance_)(\d+)(?=.png)", img_pth).group(1)
            cosine_earth_movers_images_indexed[int(index)] = img_pth

        for img_pth in euclidian_earth_movers_images:
            print(img_pth)
            index = re.search("(?<=.euclidian_earth_movers_distance_)(\d+)(?=.png)", img_pth).group(1)
            euclidian_earth_movers_images_indexed[int(index)] = img_pth

        for img_pth in cosine_earth_movers_sink_images:
            print(img_pth)
            index = re.search("(?<=.cosine_earth_movers_distancesink_)(\d+)(?=.png)", img_pth).group(1)
            cosine_earth_movers_sink_images_indexed[int(index)] = img_pth

        for img_pth in euclidian_earth_movers_sink_images:
            print(img_pth)
            index = re.search("(?<=.euclidian_earth_movers_distancesink_)(\d+)(?=.png)", img_pth).group(1)
            euclidian_earth_movers_sink_images_indexed[int(index)] = img_pth

        for img_pth in cosine_earth_movers_aggregate_images:
            print(img_pth)
            index = re.search("(?<=.cosine_earth_movers_distanceagg_)(\d+)(?=.png)", img_pth).group(1)
            cosine_earth_movers_aggregate_images_indexed[int(index)] = img_pth

        for img_pth in euclidian_earth_movers_aggregate_images:
            print(img_pth)
            index = re.search("(?<=.euclidian_earth_movers_distanceagg_)(\d+)(?=.png)", img_pth).group(1)
            euclidian_earth_movers_aggregate_images_indexed[int(index)] = img_pth

        layer_cache_images_indexed.sort(key=lambda x: x[0])
        for i, layer_cache_image in layer_cache_images_indexed:
            local_path = layer_cache_image[layer_cache_image.index("images"):]
            html_writer.add_text(f"Encoder Output Dot Product Similarity Map (layer {i})")
            html_writer.add_image(local_path)
            if i in cosine_sim_images_indexed:
                local_path = cosine_sim_images_indexed[i][cosine_sim_images_indexed[i].index("images"):]
                html_writer.add_text(f"Encoder Output Cosine Similarity Map (layer {i})")
                html_writer.add_image(local_path)
            if i in cosine_earth_movers_images_indexed:
                local_path = cosine_earth_movers_images_indexed[i][cosine_earth_movers_images_indexed[i].index("images"):]
                html_writer.add_text(f"Encoder Output Cosine Earth Mover's Distance Map (layer {i})")
                html_writer.add_image(local_path)
            if i in cosine_earth_movers_aggregate_images_indexed:
                local_path = cosine_earth_movers_aggregate_images_indexed[i][cosine_earth_movers_aggregate_images_indexed[i].index("images"):]
                html_writer.add_text(f"Encoder Output Cosine Earth Mover's Distance Aggregate Map (layer {i})")
                html_writer.add_image(local_path)
            if i in euclidian_earth_movers_images_indexed:
                local_path = euclidian_earth_movers_images_indexed[i][euclidian_earth_movers_images_indexed[i].index("images"):]
                html_writer.add_text(f"Encoder Output Euclidian Earth Mover's Distance Map (layer {i})")
                html_writer.add_image(local_path)
            if i in euclidian_earth_movers_aggregate_images_indexed:
                local_path = euclidian_earth_movers_aggregate_images_indexed[i][euclidian_earth_movers_aggregate_images_indexed[i].index("images"):]
                html_writer.add_text(f"Encoder Output Euclidian Earth Mover's Distance Aggregate Map (layer {i})")
                html_writer.add_image(local_path)
            if i in cosine_earth_movers_sink_images_indexed:
                local_path = cosine_earth_movers_sink_images_indexed[i][cosine_earth_movers_sink_images_indexed[i].index("images"):]
                html_writer.add_text(f"Encoder Output Cosine Earth Mover's Distance Sink Map (layer {i})")
                html_writer.add_image(local_path)
            if i in euclidian_earth_movers_sink_images_indexed:
                local_path = euclidian_earth_movers_sink_images_indexed[i][euclidian_earth_movers_sink_images_indexed[i].index("images"):]
                html_writer.add_text(f"Encoder Output Euclidian Earth Mover's Distance Sink Map (layer {i})")
                html_writer.add_image(local_path)


        with open(output_html_path, "w") as fh:
            fh.write(html_writer.get_html_string())


def main():
    args = get_args()
    model_path = args.model_path
    bpe_path = args.bpe_path
    output_path = args.output_path
    n = args.number_to_process
    temp = args.temp
    aggregate_stmt = args.aggregate_stmt

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(os.path.join(output_path, "images")):
        os.makedirs(os.path.join(output_path, "images"))

    model = Embedding_Visualizer(model_path, bpe_path)
    python_pth = "anonymizedpath/offline_dataset/valid.java_sa-python_sa.python_sa.bpe"
    java_pth = "anonymizedpath/offline_dataset/valid.java_sa-python_sa.java_sa.bpe"

    def get_parallel_corpus(path):
        with open(path, "r") as f:
            lines = f.readlines()
        lines = [line.split(" | ") for line in lines]
        names, programs = zip(*lines)
        names = [line.replace("@@ ", "") for line in names]
        names = [line.strip() for line in names]
        # programs = [line.strip().split() for line in programs]
        programs = [line.strip() for line in programs]
        programs = [line.replace("@@ ", "") for line in programs]  # feed in bpe tokenized programs
        return names, programs

    names, python_corpus = get_parallel_corpus(python_pth)
    names_2, java_corpus = get_parallel_corpus(java_pth)
    assert names == names_2, "names are not the same"
    # files = glob.glob(f"{input_path}/*java")
    # files = files[:n]
    pbar = tqdm(total=n)
    for i, (name, java_prog, python_prog) in enumerate(zip(names, java_corpus, python_corpus)):
        # print(f"{i}/{len(names)}")
        model.make_html_from_java_prog(name, output_path, java_prog, python_prog, temp, aggregate_stmt)
        pbar.update(1)
        if i == n:
            break


if __name__ == "__main__":
    main()