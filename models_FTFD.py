from embedding import *
from hyper_embedding import *
from time_embedding import *
from collections import OrderedDict
import torch
import json
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable


class RelationMetaLearner(nn.Module):
    def __init__(self, few, embed_size=100, num_hidden1=500, num_hidden2=200, out_size=100, dropout_p=0.5):
        super(RelationMetaLearner, self).__init__()
        self.embed_size = embed_size
        self.few = few
        self.out_size = out_size
        self.rel_fc1 = nn.Sequential(OrderedDict([
            ('fc',   nn.Linear(2*embed_size, num_hidden1)),
            ('bn',   nn.BatchNorm1d(few)),
            ('relu', nn.LeakyReLU()),
            ('drop', nn.Dropout(p=dropout_p)),
        ]))
        self.rel_fc2 = nn.Sequential(OrderedDict([
            ('fc',   nn.Linear(num_hidden1, num_hidden2)),
            ('bn',   nn.BatchNorm1d(few)),
            ('relu', nn.LeakyReLU()),
            ('drop', nn.Dropout(p=dropout_p)),
        ]))
        self.rel_fc3 = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(num_hidden2, out_size)),
            ('bn', nn.BatchNorm1d(few)),
        ]))
        nn.init.xavier_normal_(self.rel_fc1.fc.weight)
        nn.init.xavier_normal_(self.rel_fc2.fc.weight)
        nn.init.xavier_normal_(self.rel_fc3.fc.weight)

    def forward(self, inputs):
        size = inputs.shape
        x = inputs.contiguous().view(size[0], size[1], -1)
        x = self.rel_fc1(x)
        x = self.rel_fc2(x)
        x = self.rel_fc3(x)
        x = torch.mean(x, 1)

        return x.view(size[0], 1, 1, self.out_size)


class LSTM_attn(nn.Module):
    def __init__(self, embed_size=100, n_hidden=200, out_size=100, layers=1, dropout=0.5):
        super(LSTM_attn, self).__init__()
        self.embed_size = embed_size
        self.n_hidden = n_hidden
        emb_num = self.embed_size*2
        self.out_size = out_size
        self.layers = layers
        self.dropout = dropout
        self.lstm = nn.LSTM(200, self.n_hidden, self.layers, bidirectional=True, dropout=self.dropout)
        #self.gru = nn.GRU(self.embed_size*2, self.n_hidden, self.layers, bidirectional=True)
        self.out = nn.Linear(self.n_hidden*2*self.layers, self.out_size)

    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, self.n_hidden*2, self.layers)
        attn_weight = torch.bmm(lstm_output, hidden).squeeze(2).cuda()

        soft_attn_weight = F.softmax(attn_weight, 1)
        context = torch.bmm(lstm_output.transpose(1,2), soft_attn_weight)
        context = context.view(-1, self.n_hidden*2*self.layers)
        return context

    def forward(self, inputs):
        size = inputs.shape
        inputs = inputs.contiguous().view(size[0], size[1], -1)
        input = inputs.permute(1, 0, 2)
        hidden_state = Variable(torch.zeros(self.layers*2, size[0], self.n_hidden)).cuda()
        cell_state = Variable(torch.zeros(self.layers*2, size[0], self.n_hidden)).cuda()
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (hidden_state, cell_state))  # LSTM
        output = output.permute(1, 0, 2)
        attn_output = self.attention_net(output, final_cell_state)

        outputs = self.out(attn_output)
        return outputs.view(size[0], 1, 1, self.out_size)    


class EmbeddingLearner(nn.Module):
    def __init__(self):
        super(EmbeddingLearner, self).__init__()

    def forward(self, h, t, r, o, pos_num, norm):  # o time
        dim_temp = h.shape[0]
        # norm[256, 5, 1, 1, 100]
        norm = norm[:,:1,:,:]
        # norm[256, 1, 1, 1, 100]
        t = t[:, :pos_num, :, :]
        h = h[:, :pos_num, :, :]
        o = h[:, :pos_num, :, :]
        #  h[256, 5, 1, 100]
        h = torch.unsqueeze(h, dim=1)
        t = torch.unsqueeze(t, dim=1)
        h = h - torch.sum(h * norm, -1, True) * norm  # Handling of head entities
        t = t - torch.sum(t * norm, -1, True) * norm  # Handling of tail entities
        h = torch.squeeze(h, dim=1)  # [256, 5, 1, 100]
        t = torch.squeeze(t, dim=1)  # [256, 5, 1, 100]
        h = h.reshape(dim_temp, 2*pos_num, 1, 50)
        t = t.reshape(dim_temp, 2*pos_num, 1, 50)
        o = o.reshape(dim_temp, 2*pos_num, 1, 50)
        r = r.reshape(dim_temp, 2*pos_num, 1, 50)
        score = -torch.norm(h + r + o - t, 2, -1).squeeze(2)  # s+r+t-o
        p_score = score[:, :pos_num]
        n_score = score[:, pos_num:]
        return p_score, n_score


def save_grad(grad):
    global grad_norm
    grad_norm = grad


class MetaR(nn.Module):
    def __init__(self, dataset, parameter, num_symbols, embed = None):
        super(MetaR, self).__init__()
        self.device = parameter['device']
        self.beta = parameter['beta']
        self.dropout_p = parameter['dropout_p']
        self.embed_dim = parameter['embed_dim']
        self.margin = parameter['margin']
        self.abla = parameter['ablation']
        # self.rel2id = dataset['rel2id']
        self.num_rel = 10488
        self.embedding = Embedding(dataset, parameter)
        self.h_embedding = H_Embedding(dataset, parameter)
        self.t_embedding = T_Embedding(dataset, parameter)
        self.few = parameter['few']
        self.dropout = nn.Dropout(0.5)
        self.symbol_emb = nn.Embedding(num_symbols + 1, self.embed_dim, padding_idx = num_symbols)
        self.num_hidden1 = 500
        self.num_hidden2 = 200
        self.lstm_dim = 700  # parameter['lstm_hiddendim']
        self.lstm_layer = 4  # parameter['lstm_layers']

        self.symbol_emb.weight.data.copy_(torch.from_numpy(embed))

        self.h_emb = nn.Embedding(self.num_rel, self.embed_dim)
        init.xavier_uniform_(self.h_emb.weight)

        self.gcn_w = nn.Linear(3*self.embed_dim, self.embed_dim)       # change log  gcn图卷积神经网络
        self.gcn_b = nn.Parameter(torch.FloatTensor(self.embed_dim))   # change log
        self.attn_w = nn.Linear(self.embed_dim, 1)                                                       

        self.gate_w = nn.Linear(self.embed_dim, 1)
        self.gate_b = nn.Parameter(torch.FloatTensor(1))

        init.xavier_normal_(self.gcn_w.weight)                         # change log
        init.constant_(self.gcn_b, 0)                                  # change log
        init.xavier_normal_(self.attn_w.weight)

        self.symbol_emb.weight.requires_grad = False
        self.h_norm = None

        if parameter['dataset'] == 'ICEWS05-15':
            self.relation_learner = LSTM_attn(embed_size=100, n_hidden=100, out_size=50,layers=2, dropout=0.5)
        elif parameter['dataset'] == 'ICEWS18':
            self.relation_learner = LSTM_attn(embed_size=100, n_hidden=self.lstm_dim, out_size=100, layers=self.lstm_layer, dropout=self.dropout_p)
        self.embedding_learner = EmbeddingLearner()
        self.loss_func = nn.MarginRankingLoss(self.margin)
        self.rel_q_sharing = dict()
        self.norm_q_sharing = dict()


    def neighbor_encoder(self, connections, num_neighbors, iseval):
        '''
        connections: (batch, 200, 2)
        num_neighbors: (batch,)
        '''
        num_neighbors = num_neighbors.unsqueeze(1)
        entity_self = connections[:,0,0].squeeze(-1)
        relations = connections[:,:,1].squeeze(-1)
        entities = connections[:,:,2].squeeze(-1)
        times = connections[:,:,3].squeeze(-1)
        rel_embeds = self.dropout(self.symbol_emb(relations)) # (batch, 200, embed_dim)
        ent_embeds = self.dropout(self.symbol_emb(entities)) # (batch, 200, embed_dim)
        time_embeds = self.dropout(self.symbol_emb(times))
        entself_embeds = self.dropout(self.symbol_emb(entity_self))
        if not iseval:
            entself_embeds = entself_embeds.squeeze(1)

        concat_embeds = torch.cat((rel_embeds, ent_embeds, time_embeds), dim=-1) # (batch, 200, 2*embed_dim)

        out = self.gcn_w(concat_embeds) + self.gcn_b       # out gcn former change log
        out = F.leaky_relu(out)                            # out gcn former change log

        attn_out = self.attn_w(out)
        attn_weight = F.softmax(attn_out, dim=1)
        out_attn = torch.bmm(out.transpose(1,2), attn_weight)
        out_attn = out_attn.squeeze(2)
        gate_tmp = self.gate_w(out_attn) + self.gate_b
        gate = torch.sigmoid(gate_tmp)
        out_neigh = torch.mul(out_attn, gate)
        out_neighbor = out_neigh + torch.mul(entself_embeds, 1.0-gate)

        return out_neighbor

    def split_concat(self, positive, negative):
        pos_neg_e1 = torch.cat([positive[:, :, 0, :],
                                negative[:, :, 0, :]], 1).unsqueeze(2)
        pos_neg_e2 = torch.cat([positive[:, :, 1, :],
                                negative[:, :, 1, :]], 1).unsqueeze(2)
        return pos_neg_e1, pos_neg_e2


    def forward(self, task, iseval=False, curr_rel='', support_meta=None, istest=False):
        # transfer task string into embedding
        support, support_negative, query, negative = [self.embedding(t) for t in task]
        norm_vector = self.h_embedding(task[0])
        t_vector = self.t_embedding(task[0])
        few = support.shape[1]              # num of few
        num_sn = support_negative.shape[1]  # num of support negative
        num_q = query.shape[1]              # num of query
        num_n = negative.shape[1]           # num of query negative
        support_left_connections, support_left_degrees, support_right_connections, support_right_degrees = support_meta[0]
        support_left = self.neighbor_encoder(support_left_connections, support_left_degrees, iseval)
        support_right = self.neighbor_encoder(support_right_connections, support_right_degrees, iseval)  # 1024 100
        support_few = torch.cat((support_left, support_right), dim=-1)  # 1024 200
        support_few = support_few.view(support_few.shape[0], 2, self.embed_dim)  # 1024 2 100

        for i in range(self.few-1):
            support_left_connections, support_left_degrees, support_right_connections, support_right_degrees = support_meta[i+1]
            support_left = self.neighbor_encoder(support_left_connections, support_left_degrees, iseval)
            support_right = self.neighbor_encoder(support_right_connections, support_right_degrees, iseval)
            support_pair = torch.cat((support_left, support_right), dim=-1)  # tanh
            support_pair = support_pair.view(support_pair.shape[0], 2, self.embed_dim)
            support_few = torch.cat((support_few, support_pair), dim=1)
        support_few = support_few.view(support_few.shape[0], self.few, 2, self.embed_dim)  # 1024 10 100 -> 1024 5 2 100
        rel = self.relation_learner(support_few)
        rel.retain_grad()
        # relation for support  rel[256, 1, 1, 50]
        rel_s = rel.expand(-1, few+num_sn, -1, -1)
        t_vector.reshape(rel_s.shape)
        # rel_s[256, 10, 1, 50]  t_vector[256, 5, 1, 1, 100]
        if iseval and curr_rel != '' and curr_rel in self.rel_q_sharing.keys():
            rel_q = self.rel_q_sharing[curr_rel]
            time_q = t_vector;
        else:
            if not self.abla:
                # split on e1/e2 and concat on pos/neg
                sup_neg_e1, sup_neg_e2 = self.split_concat(support, support_negative)
                dim_temp = sup_neg_e1.shape[0]
                p_score, n_score = self.embedding_learner(sup_neg_e1, sup_neg_e2, rel_s, t_vector, few, norm_vector)
                # revise norm_vector  h, t, r,t_vector pos_num, norm
                # y = torch.Tensor([1024*5]).view(1024, 5).to(self.device)
                y = torch.Tensor([1]).to(self.device)

                self.zero_grad()
                #normalization = 0.0001 * (torch.sum(norm_vector**2) + torch.sum(rel_s**2))
                loss = self.loss_func(p_score, n_score, y)
                #loss = self.loss_func(p_score, n_score, y) + normalization
                loss.backward(retain_graph=True)
                grad_meta = rel.grad
                gradient_temp = self.beta*grad_meta
                # t_vector[256, 5, 1, 1, 100]
                # self.beta * grad_meta[256, 1, 1, 50]
                # t_vector.reshape(rel.shape)
                # print(rel.shape)
                # print(t_vector.shape)
                rel_q = rel - self.beta*grad_meta
                # time_q = t_vector - self.beta * grad_meta
                gradient_temp = gradient_temp.expand(-1, few+num_sn, -1, -1)
                gradient_temp = torch.unsqueeze(gradient_temp, dim=2)
                gradient_temp = gradient_temp.reshape(dim_temp, 5, 1, 1, 100)
                time_q = t_vector - gradient_temp
                norm_q = norm_vector - gradient_temp			# hyper-plane update
            else:
                time_q = t_vector
                rel_q = rel
                norm_q = norm_vector

            self.rel_q_sharing[curr_rel] = rel_q
            self.h_norm = norm_vector.mean(0)
            self.h_norm = self.h_norm.unsqueeze(0)
        # print(rel_q.shape)
        rel_q = rel_q.expand(-1, num_q + num_n, -1, -1)
        # time_q = time_q[:,:1,:,:]
        que_neg_e1, que_neg_e2 = self.split_concat(query, negative)  # [bs, nq+nn, 1, es]
        if iseval:
            norm_q = self.h_norm
        p_score, n_score = self.embedding_learner(que_neg_e1, que_neg_e2, rel_q, time_q, num_q, norm_q)

        return p_score, n_score
