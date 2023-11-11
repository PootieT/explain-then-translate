# CodeGen (Mirror)
TODO
This repo is mostly modified, adapted, and filtered from [Meta's CodeGen repo](). We use scripts from this repo
to perform ablation studies (program obfuscation), evaluation on transcoder dataset, and completion 
post processing (remove non-programmatic fragments and extract functions only). 

## Project Setup

If you are in pycharm 
make sure you mark `CodeGenMirror` as `Sources Root` (under folder panel on
the left, right click `CodeGenMirror` directory -> `Mark Directory as` -> 
`Source Root`)

Or, 
```bash
export PYTHONPATH='/path/to/explain-then-translate/CodeGenMirror'
```

## Environment
```shell
./CodeGenMirror/install_env.sh
```

## TransCoder Dataset
TODO

As mentioned in Appendix A of our [paper](), we found transcoder dataset to contain 
significant amount of errors in test cases as well as gold solutions. We did our best
to sanitize C++ to Python translation direction, as it was used to evaluation code 
translation in previous papers ([BigCode](),[Chen et al 2023]()). However we caution against 
evaluating on such dataset due to its simplicity and potential of additional errors
(in other translation directions involving Java).

We release the sanitized transcoder dataset in `CodeGenMirror/data/transcoder_evaluation_gfg_fixed` 
along with its un-fixed version (`CodeGenMirror/data/transcoder_evaluation_gfg`) in the same format
for easy comparison. Correspondingly, `CodeGenMirror/data/transcoder_test_set_fixed` contains the 
tokenized program solutions in the orginal data format provided by TransCoder authors.

For details of how we fixed the dataset, and evaluation results using Codex, please
refer to our paper Appendix A/B.