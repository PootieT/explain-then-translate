from pprint import pprint
from typing import *
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class Scorer:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right")
        if "codegen" in model_name:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, revision="main")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        if "16B" in model_name:
            self.model = self.model.half()

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.model.config.eos_token_id

        if torch.cuda.is_available():
            self.device = torch.device(torch.cuda.current_device())
            self.model = self.model.to(self.device)
        else:
            self.device = torch.device("cpu")

    def to_tokens_and_logprobs(self, input_texts: List[str]):
        # cite: https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075/17
        input_ids = self.tokenizer(input_texts, padding=True, return_tensors="pt", truncation=True).input_ids
        with torch.no_grad():
            gen_probs = torch.tensor([])
            bs, start = input_ids.shape[0], 0
            while gen_probs.shape[0] != input_ids.shape[0]:
                try:
                    curr_input_ids = input_ids[start:start + bs]
                    curr_input_ids = curr_input_ids.to(self.device)

                    outputs = self.model(curr_input_ids)
                    probs = torch.log_softmax(outputs.logits, dim=-1).detach()

                    # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
                    probs = probs[:, :-1, :]
                    curr_input_ids = curr_input_ids[:, 1:]
                    curr_gen_probs = torch.gather(probs, 2, curr_input_ids[:, :, None]).squeeze(-1)
                    if start == 0:
                        gen_probs = curr_gen_probs
                    else:
                        gen_probs = torch.cat([gen_probs, curr_gen_probs], dim=0)
                    start = gen_probs.shape[0]
                except RuntimeError as e:
                    print(f"runtime error: {e}\nhalfing the batch size and trying again.")
                    bs = bs // 2
                    torch.cuda.empty_cache()
                    print(f"current input shape: {input_ids.shape}, trying batch size of {bs}")
                    if bs == 0:
                        raise Exception("Batch size cannot be zero. Sequence too long, consider smaller model, larger GPU, or loading model on multiple GPUs")
        batch = []
        for input_sentence, input_probs in zip(input_ids, gen_probs):
            text_sequence = []
            for token, p in zip(input_sentence, input_probs):
                # append non-padding tokens/logprobs only
                if token not in self.tokenizer.all_special_ids:
                    text_sequence.append((self.tokenizer.decode(token), p.item()))
            batch.append(text_sequence)
        return batch

    def get_conditional_logprob(self, prefix_list: List[str], targets: List[str], normalize: bool=True):
        input_texts = [prefix+target for prefix, target in zip(prefix_list, targets)]
        logprobs = self.to_tokens_and_logprobs(input_texts)
        prefix_ids = self.tokenizer(prefix_list, padding=False).input_ids
        logprob_sums = []
        for i, logprob in enumerate(logprobs):
            logprob_sum = sum([p for _, p in logprob[len(prefix_ids[i]):]])
            if normalize:
                logprob_sum /= (len(logprob)-len(prefix_ids[i]))
            logprob_sums.append(logprob_sum)
        return logprob_sums


def get_few_shot_string(few_shot_file:str, is_coder: bool=True, shots: int = 4) -> str:
    fs_str = open(few_shot_file).read()
    # remove instructions
    fs_str = fs_str[fs_str.find("###"):]
    sections = [[p[:p.find("\n")].strip(), p[p.find("\n")+1:].strip()] for p in fs_str.split("### ")[1:]]
    updated_sections = []
    for s in range(shots):
        src_code_idx = s*3  # input few shot files should have 3 sections: src_code, explanation, tgt_code
        exp_idx = s*3+1
        if is_coder:
            updated_sections.append(sections[exp_idx])
            updated_sections.append(sections[src_code_idx])
        else:
            updated_sections.append(sections[src_code_idx])
            updated_sections.append(sections[exp_idx])

    fs_str = "".join([f"### {header}\n\n{content}\n\n" for header, content in updated_sections])
    return fs_str


def get_reviewer_score(code: str, explanations: List[str], scorer: Scorer, few_shot_file:Optional[str]=None, shots:int=1):
    fs_str = get_few_shot_string(few_shot_file, is_coder=False, shots=shots) if few_shot_file else ""
    instruction = "Can you explain what this Python program does in a couple of sentences?"
    prompt_prefix = f"{instruction}\n\n" \
                    f"{fs_str}" \
                     f"### Python version\n\n" \
                     f"{code}\n\n" \
                     f"### Explanation\n\n"
    score = scorer.get_conditional_logprob([prompt_prefix]*len(explanations), explanations)
    return score


def get_coder_score(code: str, explanations: List[str], scorer: Scorer, few_shot_file:Optional[str]=None, shots:int=1):
    fs_str = get_few_shot_string(few_shot_file, is_coder=True, shots=shots) if few_shot_file else ""
    instruction = "Can you write a Python program given this explanation?"
    prompt_prefix = [
        f"{instruction}\n\n" \
        f"{fs_str}" \
        f"### Explanation\n\n" \
        f"{exp}\n\n" \
        f"### Python version\n\n"
        for exp in explanations
    ]
    score = scorer.get_conditional_logprob(prompt_prefix, [code] * len(explanations))
    return score


def get_coder_reviewer_score(code: str, explanations: List[str], scorer: Scorer, few_shot_file:Optional[str]=None, shots:int=1):
    coder_score = get_coder_score(code, explanations, scorer, few_shot_file, shots)
    reviewer_score = get_reviewer_score(code, explanations, scorer, few_shot_file, shots)
    return [c+r for c, r in zip(coder_score, reviewer_score)]


if __name__ == "__main__":

    scorer = Scorer("distilgpt2")
    # input_texts = ["One plus one is two", "Good morning", "Hello, how are you?"]
    # batch = scorer.to_tokens_and_logprobs(input_texts)
    # pprint(batch)

    few_shot_file = "../few_shot_prompts/java/py-java_translate_MTexplain.txt"
    shots = 2
    code = "def hello_world():\n    print('hello world!')"
    explanations = [
        "this function `hello_world` prints the message 'hello world!' to the console.",
        "this function `hello_world` prints the message 'hello world?' to the console.",
        "this function adds 1 to the input integer"
    ]
    coder_score = get_coder_score(code, explanations, scorer, few_shot_file, shots)
    print(f"Coder score:\n{coder_score}")
    reviewer_score = get_reviewer_score(code, explanations, scorer, few_shot_file, shots)
    print(f"Reviewer score:\n{reviewer_score}")
    cr_score = get_coder_reviewer_score(code, explanations, scorer, few_shot_file, shots)
    print(f"Coder-Reviewer score:\n{cr_score}")