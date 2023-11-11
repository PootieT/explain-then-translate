import os
import pdb
import sys
import json
import random
import time
from abc import ABC
from concurrent.futures import ProcessPoolExecutor
import pickle
from pathlib import Path
from typing import *
import torch

from tqdm import tqdm
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi


sys.path.append(str(Path(__file__).parents[1].joinpath("dataset_builder")))

from codegen_sources.model.src.utils import TREE_SITTER_ROOT
from codegen_sources.preprocessing.lang_processors.lang_processor import LangProcessor
from dataset_builder.utils import get_source_code_from_prompt, find_all, CANONICAL2SHORT, get_gold_program
from evaluation.CodeBLEU.calc_code_bleu import get_code_bleu_from_examples
from scripts.unixcoder import UniXcoder

lang2ext = {
    "python": "py",
    "java": "java",
    "cpp": "cpp"
}


class Retriever(ABC):
    def __init__(self, method, lang):
        self.method = method
        self.lang = lang
        self.processor = LangProcessor.processors[self.lang](TREE_SITTER_ROOT)

    def safe_tokenize_code(self, code: str):
        try:
            tokens = self.processor.tokenize_code(code)
        except Exception as e:
            try:
                tokens = self.processor.tokenize_code(code.encode('utf-8', 'replace').decode())
            except:
                tokens = code.split()
        return tokens

    @staticmethod
    def remove_obfuscated_tokens(code, dico: str):
        # dico format: FUNC_0 clampLong | VAR_0 valueLong | VAR_1 result
        dico = {} if dico == "" else {t.split()[0]: t.split()[1] for t in dico.split(" | ")}
        for k in dico.keys():
            code = code.replace(k, "")
        return code

    def safe_obfuscate(self, code):
        try:
            return self.processor.obfuscate_code(code)
        except:
            return code, ""

    @staticmethod
    def return_topk_candidates(scores: np.array, code_ids: Union[List[str]], top_k: Optional[int]=None):
        """ scores should be (query size) X (corpus size) or (corpus size) matrix """
        if top_k is not None:
            if len(scores.shape) > 1:
                scores = np.flip(np.argsort(scores, axis=1), axis=1)[:, :top_k]
            else:
                scores = np.argsort(scores)[::-1][:top_k]
        if code_ids is not None:
            return scores, code_ids
        return scores

    def tokenize(self, codes: List[str], tokenized: bool = False):
        raise NotImplementedError

    def build_corpus(self, codes: List[str], tokenized: bool=False):
        raise NotImplementedError

    def retrieve(self, code: Union[str, List[str]],  tokenized: bool=False, code_id: Union[None, str, List[str]]=None, top_k: Optional[int]=None) -> Union[np.array, Tuple[np.array, str]]:
        raise NotImplementedError


class RandomRetriever(Retriever):
    def __init__(self, seed=42):
        super().__init__("", "java")
        self.corpus_len = None
        self.seed = seed
        pass

    def build_corpus(self, codes: List[str], tokenized: bool=False):
        self.corpus_len = len(codes)
        pass

    def retrieve(self, code: Union[str, List[str]],  tokenized: bool=False, code_id: Union[None, str, List[str]]=None, top_k: Optional[int]=None) -> Union[np.array, Tuple[np.array, str]]:
        np.random.seed(42)
        assert self.corpus_len is not None
        query_len = len(code) if isinstance(code, list) else 1
        scores = np.random.rand(query_len, self.corpus_len).squeeze()
        return self.return_topk_candidates(scores, code_id, top_k)


class CodeBLEURetriever(Retriever):
    def __init__(self, method, lang):
        super().__init__(method, lang)
        self.corpus_codes = None
        self.method = method
        self.lang = lang
        self.weights = method.replace("codebleu_","").replace("_",",")
        self.CACHE_PATH=Path(__file__).absolute().parents[1].joinpath(f"data/transcoder_test_set/valid_retrieval_few_shot/codeblue_cache_{lang}.pkl")
        os.makedirs(self.CACHE_PATH.parent, exist_ok=True)

    def weight_scores(self, scores: Tuple[float, float, float, float]):
        weights = [float(w) for w in self.weights.split(",")]
        scores_weighted = sum([weights[i]*scores[i] for i in range(4)])
        return scores_weighted

    def load_intersection_ids(self):
        assert os.path.isfile(self.GOLD_FILE)
        return pd.read_csv(self.GOLD_FILE).TARGET_CLASS

    def build_corpus(self, codes: List[str], tokenized: bool=False):
        if tokenized:
            codes = [self.processor.detokenize_code(c) for c in codes]
        self.corpus_codes = codes

    def retrieve(
        self,
        code: Union[str, List[str]],
        tokenized: bool=False,
        code_id: Union[None, str, List[str]]=None,
        top_k: Optional[int]=None
    ) -> Union[np.array, Tuple[np.array, str]]:
        if os.path.isfile(self.CACHE_PATH):
            print(f"found cached scores, loading from {self.CACHE_PATH}")
            scores_breakdown_matrix = pickle.load(open(self.CACHE_PATH, "rb"))
            assert len(scores_breakdown_matrix) == len(code) and len(scores_breakdown_matrix[0]) == len(self.corpus_codes),\
                "cache doesn't match current query, delete cache!"
            scores = np.zeros([len(code), len(self.corpus_codes)])
            for i in tqdm(range(len(code))):
                for j in range(len(self.corpus_codes)):
                    scores[i, j] = self.weight_scores(scores_breakdown_matrix[i][j])
        else:
            if isinstance(code, str):
                code = [code]
            if tokenized:
                code = [self.processor.detokenize_code(c) for c in code]
            scores = np.zeros([len(code), len(self.corpus_codes)])
            scores_breakdown_matrix = [[[] for _ in range(len(self.corpus_codes))] for _ in range(len(code))]
            for i, query in tqdm(enumerate(code), total=len(code)):
                for j, ref in tqdm(enumerate(self.corpus_codes), total=len(self.corpus_codes)):
                    score, score_breakdown = get_code_bleu_from_examples(
                        pre_references=[[ref]], hypothesis=[query], lang=self.lang,
                        params=self.weights, verbose=False, return_scores=True)
                    scores[i, j] = score
                    scores_breakdown_matrix[i][j] = score_breakdown
                    # pdb.set_trace()
            pickle.dump(scores_breakdown_matrix, open(self.CACHE_PATH, "wb"))
        return self.return_topk_candidates(scores, code_id, top_k)


class BM25Retriever(Retriever):
    def __init__(self, method: str, lang: str):
        super().__init__(method, lang)
        self.method = method
        self.lang = lang
        self.bm25 = None
        self.processor = LangProcessor.processors[self.lang](TREE_SITTER_ROOT)

    def tokenize(self, codes: List[str], tokenized: bool=False):
        if not tokenized and self.method != "bm25_obf":
            codes = [" ".join(self.processor.tokenize_code(c)) for c in codes]
        elif tokenized and self.method == "bm25_obf":
            codes = [self.processor.detokenize_code(c) for c in codes]

        if self.method.startswith("bm25_obf"):
            codes_obf = [self.safe_obfuscate(c) for c in tqdm(codes, desc="obfuscating")]
            if self.method == "bm25_obf_del":
                codes_obf = [(self.remove_obfuscated_tokens(c, d), d) for c,d in tqdm(codes_obf, desc="deleting obf vars")]
            fail_cnt = sum([codes[i] in c for i, (c,d) in enumerate(codes_obf)])
            print(f"during obfuscation, {fail_cnt} ({fail_cnt/len(codes)*100:.3f}%) program failed, using original")
            corpus_keys = [self.safe_tokenize_code(t[0]) for t in tqdm(codes_obf, desc="tokenizing after obfuscating")]
        elif self.method == "bm25_sig":
            corpus_keys = [extract_signature(c) for c in codes]
        elif self.method == "bm25_sigkey":
            corpus_keys = [extract_signature(c) + extract_keywords(c) for c in codes]
        else:
            corpus_keys = [c.split() for c in codes]
        return corpus_keys

    def build_corpus(self, codes: List[str], tokenized: bool=False):
        corpus_keys = self.tokenize(codes, tokenized)
        self.bm25 = BM25Okapi(corpus_keys)

    def retrieve(self, code: Union[str, List[str]],  tokenized: bool=False, code_id: Union[None, str, List[str]]=None, top_k: Optional[int]=None) -> Union[np.array, Tuple[np.array, str]]:
        assert self.bm25 is not None, "build corpus first before retrieve!"
        if isinstance(code, str):
            code = self.tokenize([code], tokenized)[0]
            doc_scores = self.bm25.get_scores(code)
        else:
            codes = self.tokenize(code, tokenized)
            doc_scores = np.vstack([self.bm25.get_scores(c) for c in tqdm(codes, desc="retrieving bm25 scores")])
        return self.return_topk_candidates(doc_scores, code_id, top_k)


class DenseRetriever(Retriever):
    def __init__(self, method: str, lang: str):
        super().__init__(method, lang)
        self.model_name = method
        self.lang = lang
        self.model = None
        self.tokenizer = None
        self.corpus_emb = None
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.processor = LangProcessor.processors[self.lang](TREE_SITTER_ROOT)
        self.load_model(self.model_name)

    def load_model(self, model_str):
        if model_str.lower() == "unixcoder":
            self.model = UniXcoder("microsoft/unixcoder-base").to(self.device)
            self.tokenizer = self.model.tokenize
        else:
            raise NotImplementedError

    def get_code_embedding_batch(self, codes, bs=256, corpus_embs=None):
        num_batches = (len(codes)-1) // bs + 1
        res = []
        with torch.no_grad():
            for b in tqdm(range(num_batches)):
                tokens_ids = self.tokenizer(codes[b*bs:b*bs+bs], max_length=512,
                                                mode="<encoder-only>", padding="batch")
                source_ids = torch.tensor(tokens_ids).to(self.device)
                _, func_embedding = self.model(source_ids)
                if corpus_embs is not None:
                    func_embedding /= func_embedding.norm(dim=1)[:,None]
                    sim_to_corpus = torch.mm(func_embedding, corpus_embs.T)
                    res.append(sim_to_corpus)
                else:
                    res.append(func_embedding)
        res = torch.cat(res, dim=0)
        return res

    def build_corpus(self, codes: List[str], tokenized: bool=False):
        if tokenized:
            codes = [self.processor.detokenize_code(c) for c in codes]
        corpus_emb = self.get_code_embedding_batch(codes)
        corpus_emb /= corpus_emb.norm(dim=1)[:, None]
        self.corpus_emb = corpus_emb

    def retrieve(self, code: Union[str, List[str]],  tokenized: bool=False, code_id:Union[None, str, List[str]]=None, top_k: Optional[int]=None) -> Union[np.array, Tuple[np.array, str]]:
        assert self.corpus_emb is not None, "run build_corpus first"
        if isinstance(code, str):
            code = [code]
        if tokenized:
            code = [self.processor.detokenize_code(c) for c in code]
        program_sims = self.get_code_embedding_batch(code, corpus_embs=self.corpus_emb).cpu().numpy()
        return self.return_topk_candidates(program_sims, code_id, top_k)


def retrieve_queries_parallel(query_codes: List[str], query_ids: List[str], retriever:Retriever, tokenized, top_k):
    jobs = []
    executor = ProcessPoolExecutor()
    for c, i in zip(query_codes, query_ids):
        jobs.append(
            executor.submit(
                retriever.retrieve, c, tokenized, i, top_k
            )
        )

    indices = np.zeros([len(query_ids), top_k])
    for job in tqdm(jobs):
        top_k_indices, i = job.result()
        indices[query_ids.index(i)] = top_k_indices
    return indices


def write_retrieved_functions(
    corpus: List[str],
    target_codes: List[str],
    query_codes: List[str],
    query_ids: List[str],
    top_k_indices: np.array,
    out_path: str,
    third_lang: Optional[str]=None,
    third_codes: Optional[List[str]]=None
):
    df = []
    for top_k_index, query_code, query_id in zip(top_k_indices, query_codes, query_ids):
        df_row = {"TARGET_CLASS": query_id, "code": query_code}
        for k, i in enumerate(top_k_index):
            df_row[f"top{k}_src"] = corpus[int(i)] if isinstance(corpus[int(i)], str) else ""
            df_row[f"top{k}_tgt"] = target_codes[int(i)] if isinstance(target_codes[int(i)], str) else ""
            if third_codes is not None:
                df_row[f"top{k}_{third_lang}"] = third_codes[int(i)] if isinstance(third_codes[int(i)], str) else ""
        df.append(df_row)
    df = pd.DataFrame(df)
    df.to_csv(out_path, index=False)


def simple_tests():
    val_dir = "../data/transcoder_all_train_dev/cobol"
    query_codes = [open(f"{val_dir}/{f}").read() for f in os.listdir(val_dir)]
    query_ids = [f.replace('.cbl', '') for f in os.listdir(val_dir)]

    train_df = pd.read_json("../data/transcoder_all_train/train.java-cobol.no-success-True.jsonl",
                            lines=True, orient="records", nrows=1000)
    retriever = BM25Retriever("", lang="cobol")
    retriever.build_corpus(train_df["code"], tokenized=True)
    # sanity check
    print("======= query code =======")
    print(query_codes[0])
    print("======= retrieved code =======")
    print( train_df["code"][retriever.retrieve(query_codes[0], tokenized=False, top_k=10)[1]])

    # performance comparison (parallel is slightly faster w/ 4 cores)
    train_df = pd.read_json("../data/transcoder_all_train/train.java-cobol.no-success-True.jsonl",
                            lines=True, orient="records")
    retriever = BM25Retriever("", lang="cobol")
    retriever.build_corpus(train_df["code"], tokenized=True)

    start = time.perf_counter()
    for i in range(len(query_codes)):
        retriever.retrieve(query_codes[i], tokenized=False, top_k=10)
    print(f"for loop: {time.perf_counter()-start} seconds")
    start = time.perf_counter()
    retrieve_queries_parallel(query_codes, query_ids, retriever, tokenized=False, top_k=10)
    print(f"parallel: {time.perf_counter() - start} seconds")


def get_intersection_deterministic(l1: Iterable, l2: Iterable, subset: Optional[float]=None):
    if len(l1) < len(l2):
        l2_set = set(l2)
        intersection = [i for i in l1 if i in l2_set]
    else:
        l1_set = set(l1)
        intersection = [i for i in l2 if i in l1_set]
    if subset is not None:
        np.random.shuffle(intersection)
        intersection = intersection[:int(len(intersection)*subset)]
    return intersection


def retrieve_few_shot(
    corpus_dir:str,
    query_dir: str,
    src_lang: str,
    tgt_lang: str,
    out_dir: str,
    method:str,
    topk: int,
    oracle_retrieval: bool=False,
    multi_process=True,
    subset_corpus: Optional[float]=None,
    third_prompt_lang: Optional[str]=None,
    corpus_success_only=False,
):
    print(f"Begin retrieval for {src_lang}-{tgt_lang} {method}, oracle={oracle_retrieval}, subset={subset_corpus}")
    query_lang = tgt_lang if oracle_retrieval else src_lang
    query_df = get_codes_from_completion(query_dir, query_lang, True)

    retriever = get_retriever(method, query_lang)

    # TODO instead of loading from one completion dir, load from gold program dataset
    source_df = get_codes_from_completion(corpus_dir, src_lang, corpus_success_only, True)
    target_df = get_codes_from_completion(corpus_dir, tgt_lang, corpus_success_only, True)
    intersection_ids = get_intersection_deterministic(source_df.index, target_df.index, subset_corpus)

    if third_prompt_lang is not None:
        third_df = get_codes_from_completion(corpus_dir, third_prompt_lang)
        intersection_ids = get_intersection_deterministic(intersection_ids, third_df.index)
        third_df = third_df.loc[intersection_ids]
    target_df = target_df.loc[intersection_ids]
    source_df = source_df.loc[intersection_ids]

    print("building corpus ...")
    corpus_codes = source_df["tokenized_code"] if not oracle_retrieval else target_df["tokenized_code"]
    retriever.build_corpus(corpus_codes, tokenized=True)

    os.makedirs(out_dir, exist_ok=True)

    out_path = f"{out_dir}/{src_lang}-{tgt_lang}{'-'+third_prompt_lang if third_prompt_lang is not None else ''}" \
               f"_{method}_top{topk}{'_oracle' if oracle_retrieval else ''}" \
               f"{'_subset'+str(subset_corpus) if subset_corpus is not None else ''}.csv"

    if multi_process:
        print("retrieving queries in parallel...")
        indices = retrieve_queries_parallel(query_df["code"], query_df["name"].tolist(), retriever, tokenized=False, top_k=topk)
    else:
        indices = retriever.retrieve(query_df["code"], tokenized=False, top_k=topk)

    src_code_col = "normalized_code" if "normalized_code" in source_df.columns else "code"
    tgt_code_col = "normalized_code" if "normalized_code" in target_df.columns else "code"
    write_retrieved_functions(source_df[src_code_col], target_df[tgt_code_col],
                              query_df["code"], query_df["name"], indices, out_path,
                              third_lang=third_prompt_lang,
                              third_codes=third_df["code"] if third_prompt_lang is not None else None)


def retrieve_few_shot_indices_only(
    corpus_dir:str,
    query_dir: str,
    src_lang: str,
    tgt_lang: str,
    method:str,
    topk: int,
    multi_process=True,
):
    print(f"Begin retrieval for {src_lang}-{tgt_lang} {method}")
    query_lang = src_lang
    query_df = get_codes_from_completion(query_dir, query_lang, False, True)
    retriever = get_retriever(method, query_lang)
    source_df = get_codes_from_completion(corpus_dir, src_lang, False, True)

    print("building corpus ...")
    corpus_codes = source_df["tokenized_code"]
    retriever.build_corpus(corpus_codes, tokenized=True)

    if multi_process:
        print("retrieving queries in parallel...")
        indices = retrieve_queries_parallel(query_df["code"], query_df["name"].tolist(), retriever, tokenized=False, top_k=topk)
    else:
        indices = retriever.retrieve(query_df["code"], tokenized=False, top_k=topk)
    src_to_retrieved = {n:[query_df["name"][int(i)] for i in idx] for n,idx in zip(query_df["name"], indices)}
    return src_to_retrieved, query_df, indices

def retrieve_few_shot_using_gold(
    corpus_dir:str,
    query_dir: str,
    out_dir: str,
    src_lang: str,
    tgt_lang: str,
    method:str,
    topk: int,
    multi_process=True,
):
    tgt_translator = __import__(f"humaneval_to_{tgt_lang}").Translator()
    src_to_retrieved, query_df, indices = retrieve_few_shot_indices_only(corpus_dir, query_dir, src_lang, tgt_lang, method, topk, multi_process)
    gold_df = []
    for name in query_df["name"]:
        gold_program = get_gold_program(tgt_translator, query_df.original[0].split("/")[-2], tgt_lang, name)
        gold_df.append({
            "name": name,
            "code": gold_program
        })
    gold_df = pd.DataFrame(gold_df)

    src_lang_short = CANONICAL2SHORT[src_lang] if src_lang in CANONICAL2SHORT else src_lang
    tgt_lang_short = CANONICAL2SHORT[tgt_lang] if tgt_lang in CANONICAL2SHORT else tgt_lang
    out_path = f"{out_dir}/{src_lang_short}-{tgt_lang_short}_gold_{method}_top{topk}.csv"
    write_retrieved_functions(query_df["code"], gold_df["code"],
                              query_df["code"], query_df["name"], indices, out_path)


def get_section_from_prompt(prompt_str, header: str):
    last_section_idx = list(find_all(prompt_str, "### "))[-1]
    target_section_indices = list(find_all(prompt_str.lower(), f"### {header.lower()}"))
    assert target_section_indices[-1] != last_section_idx, "Cannot use last section as query section because it is incomplete in prompt"
    start = prompt_str.find("\n", target_section_indices[-1]) + 1
    end = prompt_str.find("### ", start)
    return prompt_str[start: end].strip() + "\n\n"


def get_shortest_passing_completion(results: List[Dict[str,str]]) -> Tuple[int, Optional[Dict[str,str]]]:
    min_len = 1000000
    out_result = None
    out_i = 0
    for i, result in enumerate(results):
        if result["status"] == "OK" and len(result["program"]) < min_len:
            out_result = result
            min_len = len(result["program"])
            out_i = i
    return out_i, out_result


def get_codes_from_completion(data_path: str, lang: str, success_only=False, tokenize=False) -> pd.DataFrame:
    if os.path.isfile(data_path) and data_path.endswith(".json"):  # MultiPL-E
        data = json.load(open(data_path))
        eval_data = []
    elif os.path.isdir(data_path):  # completion results
        data = [json.load(open(f"{data_path}/{p}")) for p in os.listdir(data_path) if not p.endswith(".results.json")]
        eval_data = [json.load(open(f"{data_path}/{p}")) for p in os.listdir(data_path) if p.endswith(".results.json")]
    else:
        raise Exception("Query programs need to be in MultiPlE translation_prompt.jsonl format")

    # if need to filter based on status, filter for shortest completion with success status
    data = {d["name"]: d for d in data}
    eval_data = {d["name"]: d for d in eval_data}
    if success_only and not lang.lower().startswith("py"):
        assert not data_path.endswith(".jsonl"), "to load non-python data and filter with success, you have to input completion results"
        for name, eval_res in eval_data.items():
            best_idx, best_res = get_shortest_passing_completion(eval_res["results"])
            if best_res is None:
                del eval_data[name], data[name]
            eval_res["results"] = [best_res]
            data[name]["completions"] = [data[name]["completions"][best_idx]]
    df = pd.DataFrame(data).T
    if lang in CANONICAL2SHORT and CANONICAL2SHORT[lang] == data[list(data.keys())[0]]["language"]:  # then we are fishing for target generations
        df["code"] = [row["prompt"]+row["completions"][0] for i, row in df.iterrows()]
    else:  # simply look in the translation prompt for source code or intermediate code
        df["code"] = [get_section_from_prompt(p, lang) for p in df.translation_prompt]

    if tokenize:
        r = Retriever("", lang)
        df["tokenized_code"] = [" ".join(r.safe_tokenize_code(c)) for c in df.code]

    return df


def get_retriever(method, query_lang):
    if method.startswith("bm25"):
        retriever = BM25Retriever(method=method, lang=query_lang)
    elif method == "random":
        retriever = RandomRetriever(seed=42)
    elif method.startswith("codebleu"):
        retriever = CodeBLEURetriever(method=method, lang=query_lang)
    else:
        retriever = DenseRetriever(method=method, lang=query_lang)
    return retriever


def retrieve_few_shot_with_third(
    corpus_dir: str,
    query_dir: str,
    src_lang: str,
    tgt_lang: str,
    third_lang: str,
    out_dir: str,
    method: str,
    topk: int,
    retrieval_lang: str
):
    print(f"Begin retrieval for {src_lang}-{tgt_lang} {method}, retrieval_lang={retrieval_lang}, third_lang={third_lang}")
    src_query_ids = [f.replace(f'.{lang2ext[src_lang]}', '') for f in os.listdir(f"{query_dir}/{src_lang}")]
    tgt_query_ids = [f.replace(f'.{lang2ext[tgt_lang]}', '') for f in os.listdir(f"{query_dir}/{tgt_lang}")]
    third_lang_query_ids = [f.replace(f'.{lang2ext[third_lang]}', '') for f in os.listdir(f"{query_dir}/{third_lang}")]

    query_ids = get_intersection_deterministic(src_query_ids, tgt_query_ids)
    query_ids = get_intersection_deterministic(query_ids, third_lang_query_ids)
    query_lang = retrieval_lang
    query_codes = [open(f"{query_dir}/{query_lang}/{i}.{lang2ext[query_lang]}").read() for i in query_ids]

    retriever = get_retriever(method, query_lang)
    source_df = pd.read_csv(f"{corpus_dir}/{src_lang}.csv").set_index("TARGET_CLASS")
    target_df = pd.read_csv(f"{corpus_dir}/{tgt_lang}.csv").set_index("TARGET_CLASS")
    third_df = pd.read_csv(f"{corpus_dir}/{third_lang}.csv").set_index("TARGET_CLASS")
    intersection_ids = get_intersection_deterministic(source_df.index, target_df.index)
    intersection_ids = get_intersection_deterministic(intersection_ids, third_df.index)
    target_df = target_df.loc[intersection_ids]
    source_df = source_df.loc[intersection_ids]
    third_df = third_df.loc[intersection_ids]

    print("building corpus ...")
    corpus_codes = source_df["tokenized_code"] if retrieval_lang==src_lang else \
                   target_df["tokenized_code"] if retrieval_lang==tgt_lang else third_df["tokenized_code"]
    retriever.build_corpus(corpus_codes, tokenized=True)

    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{src_lang}-{tgt_lang}_{method}_top{topk}_ret-{retrieval_lang}.csv"
    indices = retriever.retrieve(query_codes, tokenized=False, top_k=topk)

    write_retrieved_functions(source_df["code"], target_df["code"],
                              query_codes, query_ids, indices, out_path)


def combine_retrieval_results(paths: List[str], top_k: int):
    out_df = []
    dfs = [pd.read_csv(p).set_index("TARGET_CLASS") for p in paths]
    assert [len(dfs[0]) == len(dfs[i]) for i in range(1, len(dfs))]
    # reorder the rows to same order
    for i, df in enumerate(dfs[1:]):
        dfs[i] = df.loc[dfs[0].index]
    num_query = len(dfs[0])
    for i in tqdm(range(num_query)):
        curr_k, pid = 0, dfs[0].index[i]
        src_codes = set()
        out_df_row = {"TARGET_CLASS": pid}
        for k in range(top_k):
            for df in dfs:
                src_code, tgt_code = df.at[pid, f"top{k}_src"], df.at[pid, f"top{k}_tgt"]
                if src_code not in src_codes:
                    src_codes.add(src_code)
                    out_df_row[f"top{curr_k}_src"] = src_code
                    out_df_row[f"top{curr_k}_tgt"] = tgt_code
                    curr_k += 1
                    if curr_k == top_k:
                        break
            if curr_k == top_k:
                break
        out_df.append(out_df_row)

    out_df = pd.DataFrame(out_df)
    lang_pair = paths[0].split("/")[-1].split("_")[0]
    hash_id = str(hash("|".join(p.split("/")[-1] for p in paths)))[:6]
    og_dir = Path(paths[0]).parent
    out_df.to_csv(f"{og_dir}/{lang_pair}_ensemble_{hash_id}_top{top_k}.csv")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    ### regular 2 lang retrieval
    # retrieve_few_shot(
    #     corpus_dir="../dump/python-java/humaneval-python-java-PTremove-MTsummary-completion",
    #     query_dir="../translation_prompts/python-java/humaneval-python-java-PTremove-completion.json",
    #     out_dir="../few_shot_prompts/java",
    #     src_lang="python",
    #     tgt_lang="java",
    #     method="bm25",
    #     topk=8,
    #     oracle_retrieval=False,
    #     multi_process=False,
    #     subset_corpus=None,
    #     # third_prompt_lang="summary",
    #     # query_file="../dump/transcoder_all_train_dev/java/cobol-java/codex_translation_bm25_s4.json"
    # )

    ### ensemble multiple retrieval results
    # root="../data/transcoder_all_train_dev/retrieval_few_shot"
    # combine_retrieval_results(paths=[
    #     f"{root}/cobol-java_codebleu_0.4_0.1_0.4_0.1_top32_oracle.csv",
    #     f"{root}/cobol-java_codebleu_1_0_0_0_top32_oracle.csv",
    #     f"{root}/cobol-java_codebleu_0_0_1_0_top32_oracle.csv",
    #     f"{root}/cobol-java_codebleu_0_1_0_0_top32_oracle.csv",
    # ], top_k=32)

    ### 3 way retrieval (retrieve using third lang)
    # retrieve_few_shot_with_third(
    #     corpus_dir="../data/transcoder_all_train/parallel_dataset_for_codex",
    #     query_dir="../data/transcoder_all_train_dev",
    #     out_dir="../data/transcoder_all_train_dev/retrieval_few_shot",
    #     src_lang="cobol",
    #     tgt_lang="java",
    #     third_lang="python",
    #     method="bm25",
    #     topk=32,
    #     retrieval_lang="python"
    # )

    ### retrieval from gold programs
    for tgt_lang in ["pl"]:
        retrieve_few_shot_using_gold(
            corpus_dir=f"../translation_prompts/py-{tgt_lang}/humaneval-py-{tgt_lang}-PTremove-completion.json",
            query_dir=f"../translation_prompts/py-{tgt_lang}/humaneval-py-{tgt_lang}-PTremove-completion.json",
            out_dir=f"../few_shot_prompts/{tgt_lang}",
            src_lang="python",
            tgt_lang=tgt_lang,
            method="bm25",
            topk=4,
            multi_process=False,
        )