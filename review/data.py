import random
import torch
from torch import nn
import numpy as np
import json
import pickle
import os

# external tokens
PAD, UNK, SOS, EOS = "<pad>", "<unk>", "<sos>", "<eos>"

# relation
pad_path, unk_path, self_path = "<pad>", "<unk>", "<self>"
pad_path_idx, self_path_idx = 0, 2

# node types
user_node, item_node, ent_node, word_node, pad_node = "<user>", "<item>", "<entity>", "<word>", "<pad>"


class Vocab(object):
    def __init__(self, filename):
        self._token2idx = pickle.load(open(filename, "rb"))
        self._idx2token = {v: k for k, v in self._token2idx.items()}
        self._padding_idx = self._token2idx[PAD]
        self._unk_idx = self._token2idx[UNK]

    @property
    def size(self):
        return len(self._idx2token)

    @property
    def unk_idx(self):
        return self._unk_idx

    @property
    def padding_idx(self):
        return self._padding_idx

    def idx2token(self, x):
        if isinstance(x, list):
            return [self.idx2token(i) for i in x]
        return self._idx2token[x]

    def token2idx(self, x):
        if isinstance(x, list):
            return [self.token2idx(i) for i in x]
        return self._token2idx.get(x, self.unk_idx)


def ListsofListsToLongTensor(xs, vocab=None):
    max_len = max(len(x) for x in xs)
    ys = []
    for x in xs:
        y = [vocab.token2idx(w) for w in x]
        y += [vocab.padding_idx] * (max_len - len(x))
        ys.append(y)
    data = torch.LongTensor(ys)
    return data


def ListsofArraysToLongTensor(xs):
    x = np.array([list(x.shape) for x in xs])
    shape = [len(xs)] + list(x.max(axis=0))
    data = np.zeros(shape, dtype=np.int_)  # padding_idx == 0
    for i, x in enumerate(xs):
        slicing_shape = list(x.shape)  # rel_len
        slices = tuple([slice(i, i + 1)] + [slice(0, x) for x in slicing_shape])
        data[slices] = x
    tensor = torch.LongTensor(data)
    return tensor


def ListsofMasksToLongTensor(xs):
    x = np.array([list(x.shape) for x in xs])
    shape = [len(xs)] + list(x.max(axis=0))
    data = np.ones(shape, dtype=np.int_)
    for i, x in enumerate(xs):
        slicing_shape = list(x.shape)  # rel_len
        slices = tuple([slice(i, i + 1)] + [slice(0, x) for x in slicing_shape])
        data[slices] = x
    tensor = torch.LongTensor(data)
    tensor[:, :, 0] = tensor[:, :, 1] = 0.
    return tensor


def ListsofLabelsToLongTensor(xs):
    max_len = max(len(x) for x in xs)
    ys = []
    for x in xs:
        y = x + [0] * (max_len - len(x))  # 0 for generation
        ys.append(y)
    data = torch.LongTensor(ys)
    return data


def batchify(data, vocabs):

    nodes = ListsofListsToLongTensor([x["nodes"] for x in data], vocabs["node"])
    types = ListsofListsToLongTensor([x["types"] for x in data], vocabs["type"])

    all_relation_paths = dict()  # tuple(relation_path) -> idx, list object cannot be used as key of dict
    all_relation_paths[tuple([pad_path_idx])] = 0  #
    all_relation_paths[tuple([self_path_idx])] = 1

    relations = []  # bsz * n_nodes * n_nodes
    for bidx, x in enumerate(data):
        n_nodes = len(x["nodes"])
        brs = []
        for i in range(n_nodes):
            rps = []
            for j in range(n_nodes):
                path = x["relation"][str(i) + " " + str(j)]
                path = tuple(vocabs["relation"].token2idx(path))
                path_idx = all_relation_paths.get(path, len(all_relation_paths))  # add path into dict
                if path_idx == len(all_relation_paths):
                    all_relation_paths[path] = len(all_relation_paths)
                rps.append(path_idx)
            brs.append(rps)
        brs = np.stack(brs)  # n_nodes * n_nodes
        relations.append(brs)  # bsz * n_nodes * n_nodes
    relations = ListsofArraysToLongTensor(relations)  # bsz * n_nodes * n_nodes

    # relation bank in the current batch
    relation_bank = dict()
    relation_length = dict()
    for k, v in all_relation_paths.items():
        relation_bank[v] = np.array(k, dtype=np.int)
        relation_length[v] = len(k)
    relation_bank = [relation_bank[i] for i in range(len(all_relation_paths))]  # used through relation encoder
    relation_length = [relation_length[i] for i in range(len(all_relation_paths))]

    relation_bank = ListsofArraysToLongTensor(relation_bank)  # n_rel * rel_len
    relation_length = torch.LongTensor(relation_length)  # n_rel

    tokens = [x["token"] for x in data]
    tokens = ListsofListsToLongTensor(tokens, vocabs["token"])

    # bsz * n_tokens * n_concepts
    sub_mask = [x["sub_mask"] for x in data]
    sub_mask = ListsofMasksToLongTensor([np.array(x) for x in sub_mask])

    sub_graph = []
    for x in data:
        text = x["token"]
        graph = x["sub_mask"]
        assert len(text) == len(graph), "The length of text and graph is not equal!"
        sub_graph.append([graph[idx] for idx in range(len(graph)) if text[idx] == "<sos>"])

    # batch_size * (n_nodes - 1)
    cp_tokens = [x["nodes"][1:] for x in data]
    cp_tokens = ListsofListsToLongTensor(cp_tokens, vocabs["token"])  # exclude user

    cp_labels = [x["cp_label"] for x in data]  # 0 for generation
    cp_labels = ListsofLabelsToLongTensor(cp_labels)

    attribute = [x["attribute"] for x in data]
    review = [x["token"] for x in data]

    ret = {
        "nodes": nodes,
        "types": types,
        "relations": relations,
        "relation_bank": relation_bank,
        "relation_length": relation_length,
        "tokens_in": tokens[:, :-1],
        "tokens_out": tokens[:, 1:],
        "sub_masks": sub_mask[:, :-1, :],
        "sub_graphs": sub_graph,
        "cp_tokens": cp_tokens,
        "cp_labels": cp_labels[:, 1:],
        "attribute": attribute,
        "review": review
    }
    return ret


def read_file(vocabs, config, batch_size, usage):
    tar_filename = os.path.join(config["data_dir"], config["dataset"], '{}_{}.tar'.format(usage, batch_size))
    try:
        data = torch.load(tar_filename)
    except FileNotFoundError:
        batches, data = [], []
        filename = os.path.join(config["data_dir"], config["dataset"], '{}.json'.format(usage))
        with open(filename, 'r') as f:
            for l in f.readlines():
                data.append(json.loads(l))
                if len(data) >= batch_size:
                    batches.append(data)
                    data = []

        if len(data) > 0:
            batches.append(data)

        data = []
        for bch in batches:
            ret = batchify(bch, vocabs)
            data.append(ret)
        torch.save(data, tar_filename)
    return data


class DataLoader(object):
    def __init__(self, vocabs, config, batch_size, usage):
        self.data = read_file(vocabs, config, batch_size, usage)
        self.vocabs = vocabs
        self.batch_size = batch_size
        self.usage = usage

    def __iter__(self):

        if self.usage == "train":
            random.shuffle(self.data)

        for batch in self.data:
            yield batch
