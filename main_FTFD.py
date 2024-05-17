from trainer_FTFD import *
from params import *
from data_loader import *
import json
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

if __name__ == '__main__':
    params = get_params()

    print("---------Parameters---------")
    for k, v in params.items():
        print(k + ': ' + str(v))
    print("----------------------------")

    # control random seed
    if params['seed'] is not None:
        SEED = params['seed']
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        np.random.seed(SEED)
        random.seed(SEED)

    # select the dataset
    for k, v in data_dir.items():
        data_dir[k] = params['data_path']+v
        print(data_dir[k])
    tail = '_in_train'
    #  if params['data_form'] == 'In-Train':
    #    tail = '_in_train'
    # new_dict = np.load('ICEWS05/train_dic.npy', allow_pickle='TRUE').item()
    dataset = dict()
    print("loading train_tasks{} ... ...".format(tail))
    dataset['train_tasks'] = np.load((data_dir['train_tasks'+tail]), allow_pickle='TRUE').item()
    print("loading test_tasks ... ...")
    dataset['test_tasks'] = np.load((data_dir['test_tasks']), allow_pickle='TRUE').item()
    print("loading dev_tasks ... ...")
    dataset['dev_tasks'] = np.load((data_dir['dev_tasks']), allow_pickle='TRUE').item()
    print("loading rel2candidates{} ... ...".format(tail))
    dataset['rel2candidates'] = np.load((data_dir['rel2candidates'+tail]), allow_pickle='TRUE').item()
    dataset['rel2can'] = np.load((data_dir['rel2candidates']), allow_pickle='TRUE').item()
    # dataset['rel2candidates'] = np.load((data_dir['rel2candidates' + tail]), allow_pickle='TRUE').item()
    print(len(dataset['rel2candidates'].keys()))
    print("loading e1rel_e2{} ... ...".format(tail))
    dataset['e1rel_e2'] = np.load((data_dir['e1rel_e2'+tail]), allow_pickle='TRUE').item()
    if params['data_form'] == 'Pre-Train':
        print('loading embedding ... ...')
        dataset['ent2emb'] = np.load(params['data_path']+'/ent2vec.npy')  # 实体嵌入
        dataset['rel2emb'] = np.load(params['data_path']+'/rel2vec.npy')  # 关系嵌入
        dataset['time2emb'] = np.load(params['data_path'] + '/time2vec.npy')

    print("----------------------------")

    # # data_loader
    train_data_loader = DataLoader(dataset, params, step='train')
    dev_data_loader = DataLoader(dataset, params, step='dev')
    test_data_loader = DataLoader(dataset, params, step='test')
    data_loaders = [train_data_loader, dev_data_loader, test_data_loader]
    #
    # trainer
    trainer = Trainer(data_loaders, dataset, params)

    if params['step'] == 'train':
        trainer.train()
        print("test")
        print(params['prefix'])
        trainer.reload()
        trainer.eval(istest=True)
    elif params['step'] == 'test':
        print(params['prefix'])
        if params['eval_by_rel']:
            trainer.eval_by_relation(istest=True)
        else:
            trainer.eval(istest=True)
    elif params['step'] == 'dev':
        print(params['prefix'])
        if params['eval_by_rel']:
            trainer.eval_by_relation(istest=False)
        else:
            trainer.eval(istest=False)
