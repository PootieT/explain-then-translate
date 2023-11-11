import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from local_huggingface_model import _stop_at_stop_token
from typing  import List

# Necessary export
name = "codegen"


class Stopper:
    """
    A stateful StoppingCriteria that stops generation when all outputs generated by a
    batch have ended with a stop token *at some point* during generation. Thus,
    the final tensors produced by model.generate may not all end with a stop token,
    and we have to trim them to the first stop token.

    The batch_size argument is the number of tensors we expect to generate, thus its
    the product of the input batch size and num_return_sequences to model.generate.
    """

    def __init__(self, tokenizer, stop_tokens: List[str], batch_size: int):
        # Bitvector that tracks whether we've generated a stop token for every tensor
        # in the input batch.
        self.stops = torch.zeros(batch_size, dtype=bool).cuda()
        # Builds a list of tokenized stop tokens, each repeated for every batch element.
        # Note that we cannot use tokenizer(stop_tokens, return_tensors="pt"). That
        # would add padding which would not work with code in __call__ below.
        self.stop_token_tensors = [
            torch.Tensor(t).cuda().repeat(batch_size, 1)
            for t in tokenizer(stop_tokens)["input_ids"]
        ]

    def __call__(self, input_ids, scores, **kwargs):
        seq_len = input_ids.shape[1]
        for stop_token_tensor in self.stop_token_tensors:
            stop_token_len = stop_token_tensor.shape[1]
            if seq_len < stop_token_len:
                # The sequences are presently shorter than this stop token. This
                # is unlikely if we have a non-trivial prompt, but can occur.
                continue
            # The suffix of input_id of length stop_token_len. Without this,
            # the equality check below would always fail.
            suffix = input_ids[:, -stop_token_len:]
            # Check equality and reduce with logical and along dimension 1. We
            # get boolean for every input in input_ids, to check if its
            # equal to stop_token_tensor.
            did_stop = torch.all(torch.eq(suffix, stop_token_tensor), 1)
            self.stops = self.stops.logical_or(did_stop)

        return self.stops.all().item()


MODEL = "Salesforce/codegen-16B-multi"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = (
    AutoModelForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True).half().cuda()
)


def _completion_tensors(
    prompt: str, stopper, max_length: int, temperature: float, n: int, top_p
):
    """
    Produces n samples.
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.cuda()
    max_length = max_length + input_ids.flatten().size(0)
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            num_return_sequences=n,
            max_length=max_length,
            pad_token_id=tokenizer.eos_token_id + 2,
            stopping_criteria=[stopper]
        )
    return output


def _decode_single_output(output_tensor, prompt):
    detok_hypo_str = tokenizer.decode(output_tensor, clean_up_tokenization_spaces=False)
    bos = "<|endoftext|>"
    # This may only be needed for Incoder
    if detok_hypo_str.startswith(bos):
        detok_hypo_str = detok_hypo_str[len(bos) :]
    # Skip the prompt (which may even have stop_tokens)
    return detok_hypo_str[len(prompt) :]


def completions(prompt: str, max_tokens: int, temperature: float, n: int, top_p, stop):
    stops = ["<|endoftext|>"]
    stops.extend(stop)
    stopper = Stopper(tokenizer, stops, batch_size=n)
    output_tensors = _completion_tensors(
        prompt, stopper, max_tokens, temperature, n, top_p
    )
    return [
        _stop_at_stop_token(_decode_single_output(output_tensor, prompt), stop)
        for output_tensor in output_tensors
    ]
