import argparse

import torch


def get_params():
    args = argparse.ArgumentParser()
    args.add_argument("-data", "--dataset", default="ICEWS05-15", type=str)  # ["ICEWS05-15", "ICEWS18"]
    args.add_argument("-path", "--data_path", default="./ICEWS05-15", type=str)
    args.add_argument("-form", "--data_form", default="Pre-Train", type=str)  # ["Pre-Train", "In-Train", "Discard"]
    args.add_argument("-seed", "--seed", default=None, type=int)
    args.add_argument("-few", "--few", default=1, type=int)
    args.add_argument("-nq", "--num_query", default=5, type=int)
    args.add_argument("-metric", "--metric", default="MRR", choices=["MRR", "Hits@10", "Hits@5", "Hits@1"])

    args.add_argument("-dim", "--embed_dim", default=100, type=int)
    args.add_argument("-bs", "--batch_size", default=512, type=int)
    args.add_argument("-lr", "--learning_rate", default=0.001, type=float)  # 0.001
    args.add_argument("-es_p", "--early_stopping_patience", default=10, type=int)

    # args.add_argument("-epo", "--epoch", default=100000, type=int)
    # args.add_argument("-prt_epo", "--print_epoch", default=100, type=int)
    # args.add_argument("-eval_epo", "--eval_epoch", default=1000, type=int)
    # args.add_argument("-ckpt_epo", "--checkpoint_epoch", default=1000, type=int)
    args.add_argument("-epo", "--epoch", default=100000, type=int)
    args.add_argument("-prt_epo", "--print_epoch", default=100, type=int)
    args.add_argument("-eval_epo", "--eval_epoch", default=1000, type=int)
    args.add_argument("-ckpt_epo", "--checkpoint_epoch", default=1000, type=int)

    args.add_argument("-b", "--beta", default=5, type=float)	# 5
    args.add_argument("-m", "--margin", default=1.0, type=float)	# default: 1
    args.add_argument("-p", "--dropout_p", default=0.5, type=float)
    args.add_argument("-abla", "--ablation", default=False, type=bool)

    args.add_argument("-gpu", "--device", default=0, type=int)

    args.add_argument("-prefix", "--prefix", default="exp1", type=str)
    args.add_argument("-step", "--step", default="train", type=str, choices=['train', 'test', 'dev'])
    args.add_argument("-log_dir", "--log_dir", default="log", type=str)
    args.add_argument("-state_dir", "--state_dir", default="state", type=str)
    args.add_argument("-eval_ckpt", "--eval_ckpt", default=None, type=str)
    args.add_argument("-eval_by_rel", "--eval_by_rel", default=False, type=bool)
    args.add_argument("-embed_model", "--embed_model", default="TTransE", type=str)
    args.add_argument("-max_neighbor", "--max_neighbor", default=50, type=int)

    args = args.parse_args()
    params = {}
    for k, v in vars(args).items():
        params[k] = v

    if args.dataset == 'ICEWS05-15':
        params['embed_dim'] = 100
    elif args.dataset == 'ICEWS18':
        params['embed_dim'] = 100

    params['device'] = torch.device('cuda:'+str(args.device))

    return params


data_dir = {
    'train_tasks_in_train': '/train_in_train_dic.npy',
    'train_tasks': '/train_dic.npy',
    'test_tasks': "/test_dic.npy",
    'dev_tasks': "/valid_dic.npy",

    'rel2candidates_in_train': '/rel2candidates_in_train.npy',
    'rel2candidates': '/rel2candidates.npy',

    'e1rel_e2_in_train': '/e1relt_e2_in_train.npy',
    'e1rel_e2': '/e1relt_e2.npy',

    'ent2ids': '/ent2ids',
    'ent2vec': '/ent2vec.npy',
    'rel2ids': '/rel2ids',
    'time2vec': '/time2vec.npy',
    'rel2vec': '/rel2vec.npy',
}
