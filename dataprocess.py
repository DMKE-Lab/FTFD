import os

import numpy as np

import pandas as pd

time_dict = {}
objects = []

def up(d1, d2):
	c = d1.copy()
	for key in d2:
		if key in d1:
			c[key].update(d2[key])
		else:
			c[key] = d2[key]
	return c


def read_quadruples(file_path, entity2id=None, relation2id=None, time2id=None):
	quadruples = []
	quadruples_dict = dict()
	e1relt_e2 = dict()
	rel2candidates = dict()
	
	with open(file_path, encoding='utf-8') as f:
		for line in f:
			re = []
			qe = []
			time_id = None
			l = line.strip().split('\t')
			if l[3] in time_dict:
				time_id = time_dict[l[3]]
			else:
				time_id = len(time_dict)
				time_dict[l[3]] = time_id
			if len(l) == 4:
				head, rel, tail, t = l
			elif len(l) == 5:
				head, rel, tail, t, _ = l
			if entity2id and relation2id and time2id:
				if relation2id[rel] in quadruples_dict:
					qe = quadruples_dict.get(relation2id[rel])
				qe.append((entity2id[head], relation2id[rel], entity2id[tail], time_id))
				quadruples_dict[relation2id[rel]] = qe
				e1relt_e2[int(str(entity2id[head]) + str(relation2id[rel]) + str(time_id))] = entity2id[tail]
				if relation2id[rel] in rel2candidates:
					re = rel2candidates.get(relation2id[rel])
				re.append(entity2id[head])
				re.append(entity2id[tail])
				rel2candidates[relation2id[rel]] = re
			else:
				if int(rel) in quadruples_dict:
					qe = quadruples_dict.get(int(rel))
				qe.append((int(head), int(rel), int(tail), int(time_id)))
				quadruples_dict[int(rel)] = qe
				e1relt_e2[str(head) + str(rel) + str(time_id)] = int(tail)
				if int(rel) in rel2candidates:
					re = rel2candidates.get(int(rel))
				re.append(int(head))
				re.append(int(tail))
				rel2candidates[int(rel)] = re
	return quadruples_dict, e1relt_e2, rel2candidates


if __name__ == '__main__':
	m = 'ICEWS05-15'
	all_rels = []
	
	entity2id = dict()
with open(os.path.join('ICEWS05-15/entity2id.txt'), encoding='utf-8') as f:
	for line in f:
		entity, eid = line.strip().split('\t')
		entity2id[entity] = int(eid)

with open(os.path.join('ICEWS05-15/relation2id.txt'), encoding='utf-8') as f:
	relation2id = dict()
	for line in f:
		relation, rid = line.strip().split('\t')
		relation2id[relation] = int(rid)
		all_rels.extend(relation)

time_file = os.path.join('ICEWS05-15/time2id.txt')

t2id = dict()
if os.path.exists(time_file):
	with open(time_file, 'r', encoding='utf-8') as f:
		for line in f:
			time_str, tid = line.strip().split('\t')
			t2id[time_str] = int(tid)
train_quadruples_dict, train_e1relt_e2, train_rel2candidates = read_quadruples(os.path.join('ICEWS05-15/train_tasks.txt'),
                                                                               entity2id, relation2id)
valid_quadruples_dict, valid_e1relt_e2, valid_rel2candidates = read_quadruples(os.path.join('ICEWS05-15/dev_tasks.txt'),
                                                                               entity2id, relation2id)
test_quadruples_dict, test_e1relt_e2, test_rel2candidates = read_quadruples(os.path.join('ICEWS05-15/test_tasks.txt'),
                                                                            entity2id, relation2id)
all_quadruples = dict()
# print(len(train_quadruples_dict.keys()))
# print(len(valid_quadruples_dict.keys()))
# print(len(test_quadruples_dict.keys()))
ls = []
for i in train_quadruples_dict.keys():
	if i in all_quadruples.keys():
		ls = all_quadruples.get(i)
		ls.extend(train_quadruples_dict.get(i))
	else:
		ls = train_quadruples_dict[i]
	all_quadruples[i] = ls
for i in valid_quadruples_dict.keys():
	if i in all_quadruples.keys():
		ls = all_quadruples.get(i)
		ls.extend(valid_quadruples_dict.get(i))
	else:
		ls = valid_quadruples_dict[i]
	all_quadruples[i] = ls
for i in test_quadruples_dict.keys():
	if i in all_quadruples.keys():
		ls = all_quadruples.get(i)
		ls.extend(test_quadruples_dict.get(i))
	else:
		ls = test_quadruples_dict[i]
	all_quadruples[i] = ls
task = []
rest_task = []
x = 0
y = 0
train = []
valid = []
test = []
train_in_train_dic = dict()
train_dic = dict()
valid_dic = dict()
test_dic = dict()
relation2ids = dict()
path_graph = dict()
e1relt_e2 = dict()
rel2candidates = dict()

num_e = set()
num_r = set()
num_t = set()
can = None
for i in all_quadruples.keys():
	if len(all_quadruples[i]) >= 50 and len(all_quadruples[i]) <= 500:
		can = []
		if i in train_rel2candidates.keys():
			can.extend(train_rel2candidates[i])
		if i in valid_rel2candidates.keys():
			can.extend(valid_rel2candidates[i])
		if i in test_rel2candidates.keys():
			can.extend(test_rel2candidates[i])
		x = x + 1
		
		if x <= 67:
			train_in_train_dic[i] = all_quadruples[i]
			train.extend(all_quadruples[i])
			train_dic[i] = all_quadruples[i]
		else:
			if x <= 74:
				valid.extend(all_quadruples[i])
				valid_dic[i] = all_quadruples[i]
			else:
				test.extend(all_quadruples[i])
				test_dic[i] = all_quadruples[i]
	else:
		train_in_train_dic[i] = all_quadruples[i]
		path_graph[i] = all_quadruples[i]
	for item in all_quadruples[i]:
		y = y + 1
		e1, r, e2, t = item
		num_e.add(e1)
		num_e.add(e2)
		num_r.add(r)
		num_t.add(t)
eset = set()
ls = set()
rel2candidates_in_train = dict()
e1relt_e2_in_train = dict()
num_tvt = 0
for i in train_in_train_dic.keys():
	for m in train_in_train_dic[i]:
		eset.clear()
		ls.clear()
		e1, r, e2, t = m
		tempstr = str(r) + str(t)
		eset.add(e1)
		eset.add(e2)
		if tempstr in rel2candidates_in_train.keys():
			eset = eset | rel2candidates_in_train[tempstr]
		rel2candidates_in_train[tempstr] = eset
		tempstr = str(e1) + str(r) + str(t)
		ls.add(e2)
		if tempstr in e1relt_e2_in_train.keys():
			ls = ls | e1relt_e2_in_train[tempstr]
		e1relt_e2_in_train[tempstr] = ls
for i in train_dic.keys():
	for m in train_dic[i]:
		num_tvt = num_tvt + 1
		eset.clear()
		ls.clear()
		e1, r, e2, t = m
		tempstr = str(r) + str(t)
		eset.add(e1)
		eset.add(e2)
		if tempstr in rel2candidates.keys():
			eset = eset | rel2candidates[tempstr]
		rel2candidates[tempstr] = eset
		tempstr = str(e1) + str(r) + str(t)
		ls.add(e2)
		if tempstr in e1relt_e2.keys():
			ls = ls | e1relt_e2[tempstr]
		e1relt_e2[tempstr] = ls
for i in valid_dic.keys():
	for m in valid_dic[i]:
		num_tvt = num_tvt + 1
		eset.clear()
		ls.clear()
		e1, r, e2, t = m
		tempstr = str(r) + str(t)
		eset.add(e1)
		eset.add(e2)
		if tempstr in rel2candidates.keys():
			eset = eset | rel2candidates[tempstr]
		rel2candidates[tempstr] = eset
		tempstr = str(e1) + str(r) + str(t)
		ls.add(e2)
		if tempstr in e1relt_e2.keys():
			ls = ls | e1relt_e2[tempstr]
		e1relt_e2[tempstr] = ls
for i in test_dic.keys():
	for m in test_dic[i]:
		num_tvt = num_tvt +1
		eset.clear()
		ls.clear()
		e1, r, e2, t = m
		tempstr = str(r) + str(t)
		eset.add(e1)
		eset.add(e2)
		if tempstr in rel2candidates.keys():
			eset = eset | rel2candidates[tempstr]
		rel2candidates[tempstr] = eset
		tempstr = str(e1) + str(r) + str(t)
		ls.add(e2)
		if tempstr in e1relt_e2.keys():
			ls = ls | e1relt_e2[tempstr]
		e1relt_e2[tempstr] = ls

SSS = set()
OOO = set()
trainvalidtestSSS = dict()
trainvalidtestOOO = dict()
for i in train_dic.keys():
	for m in train_dic[i]:
		e1, r, e2, t = m
		SSS.add(e1)
		OOO.add(e2)
for i in valid_dic.keys():
	for m in valid_dic[i]:
		e1, r, e2, t = m
		SSS.add(e1)
		OOO.add(e2)
for i in test_dic.keys():
	for m in test_dic[i]:
		e1, r, e2, t = m
		SSS.add(e1)
		OOO.add(e2)
print('**********************')
print(len(SSS))
print(len(OOO))
print('**********************')
# print('**********************')
# print(y)
# b=list(num_t)
# b.sort()
# print(len(b))
# b=list(num_e)
# b.sort()
# print(len(b))
# b=list(num_r)
# b.sort()
# print(len(b))
# print('**********************')
# print(len(train_dic.keys()))
# print(len(valid_dic.keys()))
# print(len(test_dic.keys()))
# print(len(train_in_train_dic.keys()))
# print(x)
# np.save('ICEWS18/data/train_in_train_dic.npy', train_in_train_dic)
# fw_stat = open('ICEWS18/data/train.txt', "w")
# for i in train:
#     e1, r, e2, t = i
#     fw_stat.write(str(e1) + '\t' + str(r) + '\t' + str(e2) + '\t' + str(t) + '\t' + "0" + '\n')
# fw_stat.close()
# fw_stat = open('ICEWS18/data/valid.txt', "w")
# for i in valid:
#     e1, r, e2, t = i
#     fw_stat.write(str(e1) + '\t' + str(r) + '\t' + str(e2) + '\t' + str(t) + '\t' + "0" + '\n')
# fw_stat.close()
# fw_stat = open('ICEWS18/data/test.txt', "w")
# for i in test:
#     e1, r, e2, t = i
#     fw_stat.write(str(e1) + '\t' + str(r) + '\t' + str(e2) + '\t' + str(t) + '\t' + "0" + '\n')
# fw_stat.close()


# print(train)
# 保存文件
# np.save('ICEWS18/data/e1relt_e2_in_train.npy', e1relt_e2_in_train)
# np.save('ICEWS18/data/rel2candidates_in_train.npy', rel2candidates_in_train)
# np.save('ICEWS18/data/train_dic.npy', train_dic)
# np.save('ICEWS18/data/valid_dic.npy', valid_dic)
# np.save('ICEWS18/data/test_dic.npy', test_dic)
# np.save('ICEWS18/data/path_graph.npy', path_graph)
# np.save('ICEWS18/data/e1relt_e2.npy', e1relt_e2)
# np.save('ICEWS18/data/rel2candidates.npy', rel2candidates)
# 读取文件
# new_dict = np.load('ICEWS05/train_dic.npy', allow_pickle='TRUE').item()
# print(len(new_dict.keys()))
#
# m = './ICEWS05-15'
# print("load data from {}".format(m))
# dataset = dict()
# with open(os.path.join(m, 'ent2ids.txt'), encoding='utf-8') as f:
#     entity2id = dict()
#     for line in f:
#         entity, eid = line.strip().split('\t')
#         entity2id[entity] = int(eid)
#
# with open(os.path.join(m, 'rel2ids.txt'), encoding='utf-8') as f:
#     relation2id = dict()
#     for line in f:
#         relation, rid = line.strip().split('\t')
#         relation2id[relation] = int(rid)
#
#
# dataset['time2id'] = open(data_dir['time2id.txt'])
# time_file = os.path.join(m, 'time2id.txt')
# time2id = None
# if os.path.exists(time_file):
#     with open(time_file, 'r', encoding='utf-8') as f:
#         time2id = dict()
#         for line in f:
#             time_str, tid = line.strip().split('\t')
#             time2id[time_str] = int(tid)
#
# def read_quadruples(file_path, entity2id=None, relation2id=None, time2id=None):
#     quadruples = []
#     with open(file_path, encoding='utf-8') as f:
#         for line in f:
#             l = line.strip().split('\t')
#             if len(l) == 4:
#                 head, rel, tail, t = l
#             elif len(l) == 5:
#                 head, rel, tail, t, _ = l
#             if entity2id and relation2id and time2id:
#                 quadruples.append((entity2id[head], relation2id[rel], entity2id[tail], time2id[t]))
#             else:
#                 quadruples.append((int(head), int(rel), int(tail), int(t)))
#     return np.array(quadruples)
# train_quadruples = read_quadruples(os.path.join(m, 'train_tasks.txt'), entity2id, relation2id)
# valid_quadruples = read_quadruples(os.path.join(m, 'dev_tasks.txt'), entity2id, relation2id)
# test_quadruples = read_quadruples(os.path.join(m, 'test_tasks.txt'), entity2id, relation2id)
# all_quadruples = np.concatenate([train_quadruples, valid_quadruples, test_quadruples], axis=0)
#
# print('num_entity: {}'.format(len(entity2id)))
# print('num_relation: {}'.format(len(relation2id)))
# print('num_train_quads: {}'.format(len(train_quadruples)))
# print('num_valid_quads: {}'.format(len(valid_quadruples)))
# print('num_test_quads: {}'.format(len(test_quadruples)))
# print('num_all_quads: {}'.format(len(all_quadruples)))
# print("finish loading raw data!\n")
#
# # if time2id is None:
# #     return entity2id, relation2id, None, train_quadruples, valid_quadruples, test_quadruples, all_quadruples
# # else:
# #     return entity2id, relation2id, time2id, train_quadruples, valid_quadruples, test_quadruples, all_quadruples
#
# # print(entity2id)
# # print(relation2id)
# # print(time2id)
# # print(train_quadruples)
# # print(valid_quadruples)
# # print(test_quadruples)
# print(all_quadruples)
# # print(all_quadruples[50][0])
#
#
