import torch
import torch.nn as nn


class Embedding(nn.Module):    # 实体的嵌入
    def __init__(self, dataset, parameter):
        super(Embedding, self).__init__()
        self.device = parameter['device']
        # self.ent2id = dataset['ent2id']
        #  在这段代码所在的上下文中，可以推测出数据集（dataset）是一个字典，其中包含了实体（entities）和它们对应的ID（id）的映射关系。
        #  通过将'ent2id'键对应的值赋给self.ent2id，表示在这个模型中使用了实体到ID的映射关系，以便后续的处理和嵌入操作。
        self.es = parameter['embed_dim']

        num_ent = 10488
        self.embedding = nn.Embedding(num_ent, self.es)
        if parameter['data_form'] == 'Pre-Train':
            self.ent2emb = dataset['ent2emb']
            self.embedding.weight.data.copy_(torch.from_numpy(self.ent2emb))
        elif parameter['data_form'] in ['In-Train', 'Discard']:
            nn.init.xavier_uniform_(self.embedding.weight)  # 使用xavier_uniform_方法对嵌入层的权重进行随机初始化

    def forward(self, triples):
        idx = [[[t[0], t[2]] for t in batch] for batch in triples]
        idx = torch.LongTensor(idx).to(self.device)
        return self.embedding(idx)



