# FTFD-FewShot-FTKG completion
 Few-shot Learning on Fuzzy Temporal Knowledge Graph Completion via Modeling Fuzzy Semantics and Dynamic Attention Network
 #  our source code and data for the paper:

## Introduction

## Dataset
The dataset can be downloaded from https://github.com/Jasper-Wu/FILT Unzip it to the directory ./FTFD.
>
## Run the code

# CUDA_VISIBLE_DEVICES=0 python main_FTFD.py --dataset ICEWS0515-ff --data_path ./ICEWS0515-ff/data --few 5 --data_form Pre-Train --max_neighbor 100 --batch_size 512