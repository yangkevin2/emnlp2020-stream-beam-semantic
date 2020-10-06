# A Streaming Approach For Efficient Batched Beam Search

Code for semantic parsing experiments for the EMNLP 2020 paper "A Streaming Approach For Efficient Batched Beam Search" by Kevin Yang, Violet Yao, John DeNero, and Dan Klein (https://arxiv.org/abs/2010.02164). For the code for machine translation and syntactic parsing experiments see https://github.com/yangkevin2/emnlp2020-stream-beam-mt and https://github.com/yangkevin2/emnlp2020-stream-beam-syntactic. 

## Setup

This code is forked from https://github.com/donglixp/lang2logic. We modify only the code in `seq2seq/jobqueries/attention` since all the code is basically repeated for each dataset. (Note that by providing the right paths and hyperparameters one can train models for all of ATIS, JOBS, and GEO from there). Follow the instructions in the original repo for installation.

Model training uses `main.py` (all datasets are small and quick to train, and commands are identical to the existing github). In particular, make sure to follow the instructions in the original repo to run `data.py` for each dataset to preprocess the data. Inference uses `sample.py`, with the `-sample` flag controlling which method to decode with. 

## Example command for sampling

The following command is for Var-Stream on CPU for the JOBS dataset, run from `seq2seq/jobqueries/attention`. Change `-gpuid` to use GPU. `--ap` is delta, `--mc` is M in the paper. 

`python sample.py -gpuid -1 -sample 5 -beam_size 10 -ap 10 -mc 3 -batch_size 10`

After preloading the data and training the models for the other datasets too, you can run the following command from the same place to do inference on e.g. ATIS.

`python sample.py -gpuid -1 -sample 5 -data_dir ../../atis/data/ -model atis_checkpoint_dir/model_seq2seq_attention -beam_size 10 -ap 10 -mc 3 -batch_size 10`