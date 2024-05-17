# FTFD-FewShot-FTKG completion
 Few-shot Learning on Fuzzy Temporal Knowledge Graph Completion via Modeling Fuzzy Semantics and Dynamic Attention Network

## Dataset
The dataset can be downloaded from https://github.com/Jasper-Wu/FILT Unzip it to the directory ./FTFD.

##  Requirements
To install the various python dependences (including tensorflow)

python 3.9

cuda 11.3

pytorch 1.12

>
## Run the code
```
CUDA_VISIBLE_DEVICES=0 python main_FTFD.py --dataset ICEWS0515-ff --data_path ./ICEWS0515-ff/data --few 5 --data_form Pre-Train --max_neighbor 100 --batch_size 512

```