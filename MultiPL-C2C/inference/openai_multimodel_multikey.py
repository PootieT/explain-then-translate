"""
This module exposes a code completion function that implements a subset of the
OpenAI completion interface (for Python). Under the hood, it can us
several keys to query Codex, or dispatch requests to other model servers
that follow the OpenAI HTTP API.
"""
import pdb
from typing import List, Tuple, Union, Dict
import time
import aiohttp
import asyncio
import logging

from dataset_builder.utils import TOFILL_TOKEN, CANONICAL2SHORT, MULTI_INTERMEDIATE_GENERATION_FLAG
from inference.chatgpt_utils import cleanup_completion, cleanup_completion_simple, find_all, cap

logger = logging.getLogger(__name__)

def assoc_to_dict_list(alist):
    """
    Given a list of (key, value) pairs, return a dictionary of lists.
    """
    unique_keys = set(key for (key, _) in alist)
    return {k: [v for (k2, v) in alist if k2 == k] for k in unique_keys}


def now():
    """
    The current time in seconds.
    """
    return int(time.time())


class OtherModelKey:
    def __init__(self, model_name, url):
        self.model_name = model_name
        self.url = url
        self.headers = {}


class OpenAIAPIKeyWithRates:
    def __init__(self, key):
        self.key = key
        # List of (time, tokens_used) pairs
        self.requests = []
        self.tokens_used = 0
        # openAI setting
        if key.startswith("sk-"):
            # self.url = "https://api.openai.com/v1/completions"
            self.url = "https://api.openai.com/v1/chat/completions"
            self.headers = {"Authorization": f"Bearer {self.key}"}
        else:  # azure setting
            # apparently azure allows regular completion with chatgpt
            self.url = "https://symdistill.openai.azure.com/openai/deployments/gpt-35-turbo-0301/completions?api-version=2022-12-01"
            # self.url = "https://symdistill.openai.azure.com/openai/deployments/gpt-35-turbo-0301/chat/completions?api-version=2023-03-15-preview"
            self.headers = {"Content-Type": "application/json", "api-key": self.key}

    def update_tokens_used(self):
        """
        The number of tokens used with this API key over the last minute.
        """
        current_time = now()
        self.requests = [
            (time, tokens)
            for (time, tokens) in self.requests
            if time + 65 >= current_time
        ]
        self.tokens_used = sum(tokens for (time, tokens) in self.requests)

    def request(self, tokens_used):
        logger.debug(f"Used {tokens_used} tokens on {self.key}")
        self.requests.append((now(), tokens_used))


def fill_translation_prompt(input: Union[List, str], completion: str):
    if isinstance(input, list):
        for i, turn in enumerate(input):
            if turn["content"] is None:
                turn["content"] = completion
                break
    else:
        input = input.replace(TOFILL_TOKEN, completion, 1)
    return input


def get_incomplete_translation_prompt(input: Union[Dict, str]):
    if isinstance(input, dict):
        new_prompt = []
        for i, turn in enumerate(input):
            if turn["content"] is not None:
                new_prompt.append(turn)
            else:
                break
    else:
        new_prompt = input[:input.find(TOFILL_TOKEN)]
    return new_prompt


def get_intermediate_stops(incomplete_prompt, request_body):
    # pdb.set_trace()
    start = list(find_all(incomplete_prompt, "###"))[-1]
    end = incomplete_prompt.find("\n", start)
    section_title = incomplete_prompt[start: end].lower()
    section_title_set = set(section_title.split())
    stops = ["\n#"]
    if "latex" in section_title:
        stops.extend(["\\end{algorithm}"])
    elif any([w in section_title for w in ["comment", "summary"]]):
        stops.extend(["\n```"])
    elif "rewritten" in section_title:
        stops.extend(["\n```"])
    elif "version" in section_title_set and any([l in section_title_set for l in CANONICAL2SHORT]):
        # this is debug version, we want the intermediate stop be exactly the same as
        # actual generation stop
        stops = request_body["stop"]

    return stops


def prompt_incomplete(prompt, request_body):
    if "message" in request_body and isinstance(request_body["messages"], list):
        if isinstance(prompt, str):
            return False
        return any([t["content"] is None for t in prompt])
    else:
        if isinstance(request_body["prompt"], str) and TOFILL_TOKEN in prompt:
            return True
    return False


class MultiModelMultiKeyCompletion:
    def __init__(self, openai_api_keys: List[str], other_models: List[Tuple[str, str]]):
        # An association list mapping API keys to a list of tuples
        self.openai_api_keys = [OpenAIAPIKeyWithRates(key) for key in openai_api_keys]
        self.openai_api_keys_lock = asyncio.Lock()
        self.openai_api_keys_available = asyncio.Semaphore(len(openai_api_keys))
        self.client_session = aiohttp.ClientSession()

        # self.other_models maps each model name to a list of URLs that serve that model.
        self.other_models = assoc_to_dict_list(
            [(name, OtherModelKey(name, url)) for (name, url) in other_models]
        )

        self.other_model_names = set(model for (model, _) in other_models)
        # self.other_models_semaphores maps each model name to a semaphore that has
        # the number of available URLs for that model.
        self.other_models_semaphores = {}
        for model_name, urls in self.other_models.items():
            self.other_models_semaphores[model_name] = asyncio.Semaphore(len(urls))

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.client_session.close()

    async def get_least_used_key(
        self, model_name, estimated_usage: int
    ) -> Union[OpenAIAPIKeyWithRates, str]:
        # Easy case: we are contacting a self-hosted model server.
        if model_name in self.other_model_names:
            # Acquire the semaphore for model_name. Will block if no URLs are available.
            await self.other_models_semaphores[model_name].acquire()
            model_url = self.other_models[model_name].pop()
            return model_url

        await self.openai_api_keys_available.acquire()  # decrement semaphore, blocking at 0

        # Get the key on which we have used the fewest tokens.
        async with self.openai_api_keys_lock:
            for key in self.openai_api_keys:
                key.update_tokens_used()
            key = min(self.openai_api_keys, key=lambda key: key.tokens_used)
            self.openai_api_keys.remove(key)

        # Even though we have the key, let's self-rate limit if we think we will
        # be rate-limited by OpenAI.
        while key.tokens_used + estimated_usage > 120000:
            logger.debug(f"Sleeping because key {key.key} has {key.tokens_used} tokens")
            await asyncio.sleep(1)
            key.update_tokens_used()
        logger.debug(f"Using key {key.key} with {key.tokens_used} tokens used")

        return key

    async def release_key(self, key, usage):
        async with self.openai_api_keys_lock:
            # Easy case: this self-hosted model is now available.
            if isinstance(key, OtherModelKey):
                self.other_models[key.model_name].append(key)
                self.other_models_semaphores[key.model_name].release()
                return

            self.openai_api_keys.append(key)
            key.request(usage)
            self.openai_api_keys_available.release()

    async def completion(
        self, model, prompt, max_tokens: int, temperature, n, top_p, stop
    ):
        request_body = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n": n,
            "top_p": top_p,
            "stop": stop,
        }
        if model.startswith("gpt-3.5-turbo") and isinstance(prompt, list):
            chat_completion = True
            prompt_field = "messages"
        else:
            chat_completion = False
            prompt_field = "prompt"
        request_body[prompt_field] = prompt
        multi_intermediate_generation = False
        while True:
            try:
                key = await self.get_least_used_key(model, max_tokens * n)
                if model == "gpt-3.5-turbo":
                    if key.url.startswith("https://api.openai") and "chat" not in key.url:
                        key.url = key.url.replace("v1/completions", "v1/chat/completions")
                    else:
                        del request_body["model"]

                # pdb.set_trace()
                # multiturn conversation filling out intermediate steps, keeping the same key
                while prompt_incomplete(prompt, request_body):
                    # only need one response, with no stop tokens
                    incomplete_prompt = get_incomplete_translation_prompt(prompt)
                    intermediate_stops = get_intermediate_stops(incomplete_prompt, request_body)
                    if incomplete_prompt.startswith(MULTI_INTERMEDIATE_GENERATION_FLAG):
                        incomplete_prompt = incomplete_prompt.replace(MULTI_INTERMEDIATE_GENERATION_FLAG, "")
                        multi_intermediate_generation = True
                    if chat_completion:
                        request_body.update({"n": n if multi_intermediate_generation else 1, "stop": None, "messages": incomplete_prompt})
                    else:
                        # need to be careful with this n intermediate step generation
                        request_body.update({"n": n if multi_intermediate_generation else 1,
                                             "stop": intermediate_stops,
                                             "prompt": incomplete_prompt})
                    async with self.client_session.post(
                            key.url,
                            json=request_body,
                            headers=key.headers,
                    ) as response:
                        if response.status == 200:
                            response_json = await response.json()
                            if not multi_intermediate_generation:
                                if not chat_completion:  # regular completion
                                    intermediate_completion = response_json["choices"][0]["text"]
                                else:  # chat completion
                                    intermediate_completion = response_json["choices"][0]["message"]["content"] \
                                        if isinstance(incomplete_prompt, str) else response_json["choices"][0]["text"]
                                prompt = fill_translation_prompt(prompt, intermediate_completion)
                            else:
                                # if only generating intermediate response, generate multiple beams, respond all at once
                                assert not chat_completion
                                prompts = [fill_translation_prompt(prompt.replace(MULTI_INTERMEDIATE_GENERATION_FLAG, ""),
                                                                choice["text"]) for choice in response_json["choices"]]
                                prompt = prompts[0]
                        elif response.status == 408:
                            logger.warning(
                                f"Operation timeout, releasing the key"
                            )
                            await self.release_key(key, fudge)
                        elif response.status == 429:
                            fudge = max(10000, 150000 - key.tokens_used)
                            logger.warning(
                                f"Rate limited with {key.key}. Adding {fudge} tokens."
                            )
                            await self.release_key(key, fudge)  # # Guess
                            # await asyncio.sleep(5)
                        else:
                            logger.error(
                                f"Error {response.status}. Sleeping for 5 seconds."
                            )
                            logger.error(await response.text())
                            await asyncio.sleep(5)

                # extracting final answer
                if chat_completion:
                    request_body.update({"n": n if not multi_intermediate_generation else 1, "stop": stop, "messages": prompt if isinstance(prompt, list) else request_body["messages"]})
                else:
                    request_body.update({"n": n if not multi_intermediate_generation else 1, "stop": stop, "prompt": prompt})
                async with self.client_session.post(
                        key.url,
                        json=request_body,
                        headers=key.headers,
                ) as response:
                    if response.status == 200:
                        response_json = await response.json()
                        if type(response_json) == list:
                            await self.release_key(key, 0)
                            return prompts if multi_intermediate_generation else request_body.get("messages", None), response_json

                        await self.release_key(
                            key, (response_json["usage"]["total_tokens"])
                        )

                        if model.startswith("gpt-3.5-turbo") and "message" in request_body:
                            prompt_message = prompt if isinstance(prompt, str) else prompt[-1]["content"]
                            return request_body.get(prompt_field, None), \
                                   [cleanup_completion(choice["message"]["content"], prompt_message)
                                    for choice in response_json["choices"]]
                        else:
                            return prompts if multi_intermediate_generation else request_body.get(prompt_field, None), \
                                   [cleanup_completion_simple(choice["text"], prompt)
                                    for choice in response_json["choices"]]
                    elif response.status == 408:
                        logger.warning(
                            f"Operation timeout, releasing the key"
                        )
                        await self.release_key(key, fudge)
                    elif response.status == 429:
                        fudge = max(10000, 150000 - key.tokens_used)
                        logger.warning(
                            f"Rate limited with {key.key}. Adding {fudge} tokens."
                        )
                        await self.release_key(key, fudge)  # # Guess
                    else:
                        logger.error(
                            f"Error {response.status}. Sleeping for 5 seconds."
                        )
                        logger.error(await response.text())
                        await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Exception from {key.url}, prompt is:\n{prompt}")
                await self.release_key(key, 0)
                return [] if multi_intermediate_generation else request_body.get(prompt_field, None), []
