import random
import numpy as np
import time

class DataLoader(object):
    def __init__(self, dataset, parameter, step='train'):
        self.step = step
        self.curr_rel_idx = 0
        self.tasks = dataset[step+'_tasks']
        self.rel2candidates = dataset['rel2candidates']
        self.rel2can =  dataset['rel2can']
        self.e1rel_e2 = dataset['e1rel_e2']
        self.all_rels = sorted(list(self.tasks.keys()))  # hash表存，key是relation
        # if step == "train": print(len(self.rel2candidates.keys()))
        self.num_rels = len(self.all_rels)
        self.few = parameter['few']
        self.bs = parameter['batch_size']
        self.nq = parameter['num_query']

        if step != 'train':
            self.eval_triples = []  #
            for rel in self.all_rels:
                self.eval_triples.extend(self.tasks[rel][self.few:])
            self.curr_tri_idx = 0

    def next_one(self):
        # shift curr_rel_idx to 0 after one circle of all relations

        if self.curr_rel_idx % self.num_rels == 0:
            # 如果当前关系的索引 curr_rel_idx 对关系总数 num_rels 取模等于0，表示已经遍历了一轮所有关系。
            random.shuffle(self.all_rels)
            self.curr_rel_idx = 0

        # get current relation and current candidates

        curr_rel = self.all_rels[self.curr_rel_idx]
        self.curr_rel_idx = (self.curr_rel_idx + 1) % self.num_rels
        curr_cand = self.rel2candidates[str(curr_rel)]
        while len(curr_cand) <= 3 or len(self.tasks[curr_rel]) <= 10:
            # print(len(curr_cand))
            # print(len(self.tasks[curr_rel]))
            curr_rel = self.all_rels[self.curr_rel_idx]
            self.curr_rel_idx = (self.curr_rel_idx + 1) % self.num_rels
            curr_cand = self.rel2candidates[str(curr_rel)]
        # get current tasks by curr_rel from all tasks and shuffle it
        curr_tasks = self.tasks[curr_rel]
        curr_tasks_idx = np.arange(0, len(curr_tasks), 1)
        curr_tasks_idx = np.random.choice(curr_tasks_idx, self.few+self.nq)
        support_triples = [curr_tasks[i] for i in curr_tasks_idx[:self.few]]
        query_triples = [curr_tasks[i] for i in curr_tasks_idx[self.few:]]
        # construct support and query negative triples
        support_negative_triples = []
        for triple in support_triples:
            e1, rel, e2, t = triple
            while True:
                for negative in curr_cand:
                    # if (negative not in self.e1rel_e2[str(e1) + str(rel) + str(t)]) \
                    #         and negative != e2:
                    if negative != e2:
                        break
                break
            support_negative_triples.append([e1, rel, negative, t])

        negative_triples = []
        for triple in query_triples:
            e1, rel, e2, t = triple
            while True:
                for negative in curr_cand:
                    # if (negative not in self.e1rel_e2[str(e1) + str(rel) + str(t)]) \
                    #         and negative != e2:
                    if negative != e2:
                        break
                break
            negative_triples.append([e1, rel, negative, t])

        return support_triples, support_negative_triples, query_triples, negative_triples, curr_rel

    def next_batch(self):
        next_batch_all = [self.next_one() for _ in range(self.bs)]

        support, support_negative, query, negative, curr_rel = zip(*next_batch_all)
        return [support, support_negative, query, negative], curr_rel

    def next_one_on_eval(self):
        if self.curr_tri_idx == self.num_tris:
            return "EOT", "EOT"

        # get current triple
        # query_triple = self.eval_triples[int(key)]
        query_triple = self.eval_triples[self.curr_tri_idx]
        a, b, c, d = query_triple
        self.curr_tri_idx += 1
        curr_rel = query_triple[1]
        curr_task = self.tasks[curr_rel]
        if self.step == 'test':
            curr_cand = self.rel2can[str(curr_rel)]
        else:
            curr_cand = self.rel2candidates[str(curr_rel)]

        # get support triples
        support_triples = curr_task[:self.few]

        # construct support negative
        support_negative_triples = []
        shift = 0
        for triple in support_triples:
            e1, rel, e2, t = triple
            while True:
                for negative in curr_cand:
                    # if (negative not in self.e1rel_e2[str(e1) + str(rel) + str(t)]) \
                    #         and negative != e2:
                    if negative != e2:
                        break
                    else:
                        shift += 1
                break
            support_negative_triples.append([e1, rel, negative, t])
        # construct negative triples
        negative_triples = []
        ls = [query_triple]
        for triple in ls:
            e1, rel, e2, t = triple
            while True:
                for negative in curr_cand:
                    # if (negative not in self.e1rel_e2[str(e1) + str(rel) + str(t)]) \
                    #         and negative != e2:
                    if negative != e2:
                        break
                break
            negative_triples.append([e1, rel, negative, t])
        support_triples = [support_triples]
        support_negative_triples = [support_negative_triples]
        query_triple = [[query_triple]]
        negative_triples = [negative_triples]

        return [support_triples, support_negative_triples, query_triple, negative_triples], curr_rel

    def next_one_on_eval_by_relation(self, curr_rel):
        if self.curr_tri_idx == len(self.tasks[curr_rel][self.few:]):
            self.curr_tri_idx = 0
            return "EOT", "EOT"

        # get current triple
        query_triple = self.tasks[curr_rel][self.few:][self.curr_tri_idx]
        self.curr_tri_idx += 1
        # curr_rel = query_triple[1]
        curr_cand = self.rel2candidates[str(curr_rel)]
        curr_task = self.tasks[curr_rel]

        # get support triples
        support_triples = curr_task[:self.few]

        # construct support negative
        support_negative_triples = []
        shift = 0
        for triple in support_triples:
            e1, rel, e2, t = triple
            while True:
                for negative in curr_cand:
                    # if (negative not in self.e1rel_e2[str(e1) + str(rel) + str(t)]) \
                    #         and negative != e2:
                    if negative != e2:
                        break
                    else:
                        shift += 1
                break
            support_negative_triples.append([e1, rel, negative, t])
        # construct negative triples
        negative_triples = []
        for triple in query_triple:
            e1, rel, e2, t = triple
            while True:
                for negative in curr_cand:
                    # if (negative not in self.e1rel_e2[str(e1) + str(rel) + str(t)]) \
                    #         and negative != e2:
                    if negative != e2:
                        break
                break
            negative_triples.append([e1, rel, negative, t])
        support_triples = [support_triples]
        support_negative_triples = [support_negative_triples]
        query_triple = [[query_triple]]
        negative_triples = [negative_triples]

        return [support_triples, support_negative_triples, query_triple, negative_triples], curr_rel

