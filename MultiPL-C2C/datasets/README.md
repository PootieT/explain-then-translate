# Datasets 

This directory contains the original problems in HumanEval as well as MBPP, 
along with HumanEval programs sample from MultiPL-E in 18 languages of
different quality.

The original data from MultiPL-E include:

- mbpp
- mbpp-typed
- originals (HumanEval)
- originals-with-cleaned-doctests
- sanitized-mbpp.json

As an addition with MultiPL-C2C, we included the following data directories:

- humaneval_multi
- humaneval_multi_all
- humaneval_multi_wrong

### Gold Solutions

`humaneval_multi` contains (canonical) gold solution to HumanEval in 18 languages,
sampled from GPT-3.5-turbo-0301 around the time of March, 2023. Each problem 
contains on solution, which heuristically select as the shortest passing 
solution to each problem in HumanEval. For code translation prompts in 
MultiPL-C2C in directions other than Python -> X, we use programs here as the 
source program to translate into target programs.

### Unverified samples

`humaneval_multi_all` contains list of unique sampled solutions to with their 
sampled frequency in 18 languages. The format looks like this:

```json
{
      "HumanEval_134_check_if_last_char_is_a_letter": {
        "Sampled C++ program # 1": 8,
        "Sampled C++ program # 2": 2
      }
}
```

Where each unique program is mapped to the number of times it is sampled 
throughout the experiments in our work. When generating prompts using this 
set of solutions, to mimic sampling output of GPT-3.5, we perform weighted 
sampling given the sample frequencies of each program.

### Verified incorrect samples

`humaneval_multi_wrong` is organized in similar format as `humaneval_multi_all`
except all programs included here are sampled programs from GPT-3.5 that do not
pass unit tests. It is used as ablations in our paper to test the sensitivity
language models are to incorrect in-context examples.

### Usage Warning

As these are generations from GPT-3.5 or language models in general, these 
should NOT be included in any training data in future language model training
at all. Training on these program solutions would defeat the purpose of 
evaluation benchmarks such as MultiPL-C2C, MultiPL-E that attempts to evaluate
langauge model on programs never seen before.