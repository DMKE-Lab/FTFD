import torch
import torch.nn as nn


class T_Embedding(nn.Module):
    def __init__(self, dataset, parameter):
        super(T_Embedding, self).__init__()
        self.device = parameter['device']
        self.es = parameter['embed_dim']
        # self.rel2id = dataset['rel2id']

        num_rel = 4017
        self.norm_vector = nn.Embedding(num_rel, self.es)

        nn.init.xavier_uniform_(self.norm_vector.weight)

    def forward(self, triples):
        rel_emb = [[[[t[3]]] for t in batch] for batch in triples]
        rel_emb = torch.LongTensor(rel_emb).to(self.device)
        return self.norm_vector(rel_emb)



