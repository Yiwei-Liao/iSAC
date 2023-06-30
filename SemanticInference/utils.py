from __future__ import division
from __future__ import print_function
import random
from collections import namedtuple, Counter
import numpy as np

from BFS.KB import KB
from BFS.BFS import BFS

# hyperparameters
state_dim = 200
action_space = 400
eps_start = 1
eps_end = 0.1
epe_decay = 1000
replay_memory_size = 10000
batch_size = 128
embedding_dim = 100
gamma = 0.99
target_update_freq = 1000
max_steps = 50
max_steps_test = 50

dataPath = './NELL-995/'

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


def teacher(e1, e2, num_paths, env, path=None):
    f = open(path)
    content = f.readlines()
    f.close()
    kb = KB()  # 存储graph.txt中的所有三元组
    for line in content:
        ent1, rel, ent2 = line.rsplit()
        kb.addRelation(ent1, rel, ent2)
    # kb.removePath(e1, e2)
    intermediates = kb.pickRandomIntermediatesBetween(e1, e2, num_paths)  # num_paths=5
    res_entity_lists = []
    res_path_lists = []
    for i in range(num_paths):
        suc1, entity_list1, path_list1 = BFS(kb, e1, intermediates[i])
        suc2, entity_list2, path_list2 = BFS(kb, intermediates[i], e2)
        if suc1 and suc2:
            res_entity_lists.append(entity_list1 + entity_list2[1:])
            res_path_lists.append(path_list1 + path_list2)
    print('BFS found paths:', len(res_path_lists))
    # 随机选取实体并双向搜索至头节点和尾结点
    # ---------- clean the path --------
    res_entity_lists_new = []
    res_path_lists_new = []
    for entities, relations in zip(res_entity_lists, res_path_lists):
        rel_ents = []
        for i in range(len(entities) + len(relations)):
            if i % 2 == 0:
                rel_ents.append(entities[int(i / 2)])
            else:
                rel_ents.append(relations[int(i / 2)])

        entity_stats = Counter(entities).items()
        duplicate_ents = [item for item in entity_stats if item[1] != 1]
        duplicate_ents.sort(key=lambda x: x[1], reverse=True)
        for item in duplicate_ents:
            ent = item[0]
            ent_idx = [i for i, x in enumerate(rel_ents) if x == ent]
            if len(ent_idx) != 0:
                min_idx = min(ent_idx)
                max_idx = max(ent_idx)
                if min_idx != max_idx:
                    rel_ents = rel_ents[:min_idx] + rel_ents[max_idx:]
        entities_new = []
        relations_new = []
        for idx, item in enumerate(rel_ents):
            if idx % 2 == 0:
                entities_new.append(item)
            else:
                relations_new.append(item)
        res_entity_lists_new.append(entities_new)
        res_path_lists_new.append(relations_new)

    # print(res_entity_lists_new)
    # print(res_path_lists_new)

    good_episodes = []
    targetID = env.entity2id_[e2]
    for path in zip(res_entity_lists_new, res_path_lists_new):
        good_episode = []
        for i in range(len(path[0]) - 1):
            currID = env.entity2id_[path[0][i]]
            nextID = env.entity2id_[path[0][i + 1]]
            state_curr = [currID, targetID, 0]
            state_next = [nextID, targetID, 0]
            actionID = env.relation2id_[path[1][i]]
            good_episode.append(
                Transition(state=env.idx_state(state_curr), action=actionID, next_state=env.idx_state(state_next),
                           reward=1))
        good_episodes.append(good_episode)
    print("good_episodes:", len(good_episodes))
    # print("good_episodes:", good_episodes)
    return good_episodes


def path_clean(path):
    rel_ents = path.split(' -> ')
    relations = []
    entities = []
    for idx, item in enumerate(rel_ents):
        if idx % 2 == 0:
            relations.append(item)
        else:
            entities.append(item)
    entity_stats = Counter(entities).items()
    duplicate_ents = [item for item in entity_stats if item[1] != 1]
    duplicate_ents.sort(key=lambda x: x[1], reverse=True)
    for item in duplicate_ents:
        ent = item[0]
        ent_idx = [i for i, x in enumerate(rel_ents) if x == ent]
        if len(ent_idx) != 0:
            min_idx = min(ent_idx)
            max_idx = max(ent_idx)
            if min_idx != max_idx:
                rel_ents = rel_ents[:min_idx] + rel_ents[max_idx:]
    return ' -> '.join(rel_ents)


def bfs_two(e1, e2, path, kb, kb_inv):
    start = 0
    end = len(path)
    left = set()
    right = set()
    left.add(e1)
    right.add(e2)

    left_path = []
    right_path = []
    while (start < end):
        left_step = path[start]
        left_next = set()
        right_step = path[end - 1]
        right_next = set()

        if len(left) < len(right):
            left_path.append(left_step)
            start += 1
            for entity in left:
                try:
                    for path_ in kb.getPathsFrom(entity):
                        if path_.relation == left_step:
                            left_next.add(path_.connected_entity)
                except Exception as e:
                    print('left', len(left))
                    print(left)
                    print('not such entity')
                    return False
            left = left_next

        else:
            right_path.append(right_step)
            end -= 1
            for entity in right:
                try:
                    for path_ in kb_inv.getPathsFrom(entity):
                        if path_.relation == right_step:
                            right_next.add(path_.connected_entity)
                except Exception as e:
                    print('right', len(right))
                    print('no such entity')
                    return False
            right = right_next

    if len(right & left) != 0:
        return True
    return False

'''
    Parameters:
        feature_stats: retrain()输出的可行路径，文件名为path_stats.txt
        featurePath: test()输出的可行路径，文件名为path_to_use.txt
        relationId_path: 将relation字符串转换为id编号的文件路径
    
    
'''
def get_features(feature_stats, featurePath, relationId_path):
    stats = {}
    f = open(feature_stats)
    path_freq = f.readlines()
    f.close()
    for line in path_freq:
        path = line.split('\t')[0]
        num = int(line.split('\t')[1])
        stats[path] = num
    max_freq = max(stats.values())

    relation2id = {}
    f = open(relationId_path)
    content = f.readlines()
    f.close()
    for line in content:
        relation2id[line.split()[0]] = int(line.split()[1])

    # 读取test()输出的路径
    useful_paths = []
    named_paths = []
    f = open(featurePath)
    paths = f.readlines()
    f.close()

    for line in paths:
        path = line.rstrip()
        # print(max_freq)
        if path not in stats:
            continue
        elif max_freq > 1 and stats[path] < 2:
            continue

        length = len(path.split(' -> '))

        if length <= 10:
            pathIndex = []
            pathName = []
            relations = path.split(' -> ')

            for rel in relations:
                pathName.append(rel)
                rel_id = relation2id[rel]
                pathIndex.append(rel_id)
            useful_paths.append(pathIndex)
            named_paths.append(pathName)

    print('How many paths used: ', len(useful_paths))
    return useful_paths, named_paths
