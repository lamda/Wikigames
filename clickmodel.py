# -*- coding: utf-8 -*-

from __future__ import division, print_function

import collections
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pdb
import random
import scipy.stats


pd.options.mode.chained_assignment = None
pd.set_option('display.width', 400)


# load dataframe, group by page
#     what fraction of pages are visited in our games?
# compute ground truth model
#     resulting in a distribution to page visits excluding the start pages
# compute uniform click model
# compute linear link pos click model
# compare distributions with KLD
# compute pskip model (or something similar)


class ClickModel(object):
    def __init__(self, df):
        self.df = df
        self.data = collections.defaultdict(float)

    def normalize(self):
        total = sum(self.data.values())
        self.data = [self.data[k] / total for k in sorted(self.data)]



class GroundTruthModel(ClickModel):
    def __init__(self, df):
        self.label = 'GroundTruth'
        super(GroundTruthModel, self).__init__(df)
        self.data = {k: v
                     for k, v in df['node_next'].value_counts().iteritems()}
        self.normalize()
        print('click data for %d pages' % len(self.data))

    def compare(self, mdl):
        kl = np.abs(scipy.stats.entropy(self.data, mdl.data, base=2))
        lab = self.label + '\t' + mdl.label
        print('\t%.2f\t%s' % (kl, lab))


class UniformModel(ClickModel):
    def __init__(self, df):
        super(UniformModel, self).__init__(df)
        self.label = 'Uniform'
        keys = df['node'].value_counts().index
        grouped = df.groupby('node')
        for kidx, key in enumerate(keys):
            print(kidx, '/', len(keys), end='\r')
            df_sub = grouped.get_group(key)
            targets = set(df_sub['node_next'])
            for t in targets:
                self.data[t] += len(df_sub) / len(targets)
        self.normalize()


class IndirectProportionModel(ClickModel):
    def __init__(self, df):
        super(IndirectProportionModel, self).__init__(df)
        self.label = 'Indirect Proportional'
        keys = df['node'].value_counts().index
        grouped = df.groupby('node')
        for kidx, key in enumerate(keys):
            print(kidx, '/', len(keys), end='\r')
            df_sub = grouped.get_group(key)
            node2p = collections.defaultdict(float)
            for ridx, row in df_sub.iterrows():
                node2p[row['node_next']] += 1 / (row['linkpos_first'] + 1)
                if row['ambiguous']:
                    node2p[row['node_next']] += 1 / (row['linkpos_last'] + 1)
            total = sum(node2p.values())
            for k, v in node2p.items():
                self.data[k] += (v * len(df_sub)) / total

        self.normalize()


class InverseRankModel(ClickModel):
    def __init__(self, df):
        super(InverseRankModel, self).__init__(df)
        self.label = 'Inverse Rank'
        keys = df['node'].value_counts().index
        grouped = df.groupby('node')
        for kidx, key in enumerate(keys):
            print(kidx, '/', len(keys), end='\r')
            df_sub = grouped.get_group(key)
            nodepos = set()
            for ridx, row in df_sub.iterrows():
                nodepos.add((row['node_next'], row['linkpos_first'] + 1))
                if row['ambiguous']:
                    nodepos.add((row['node_next'], row['linkpos_last'] + 1))
            total = sum(range(1, len(nodepos)+1)) + len(nodepos)
            for idx, e in enumerate(sorted(nodepos)):
                self.data[e[0]] += ((idx+1) * len(df_sub)) / total
        self.normalize()


class InverseModel(ClickModel):
    def __init__(self, df):
        super(InverseModel, self).__init__(df)
        self.label = 'Test'
        keys = df['node'].value_counts().index
        grouped = df.groupby('node')
        for kidx, key in enumerate(keys):
            print(kidx, '/', len(keys), end='\r')
            df_sub = grouped.get_group(key)
            node2p = collections.defaultdict(float)
            for ridx, row in df_sub.iterrows():
                node2p[row['node_next']] += row['word_count'] - row['linkpos_first']
                if row['ambiguous']:
                    node2p[row['node_next']] += row['word_count'] - row['linkpos_last']
            total = sum(node2p.values())
            for k, v in node2p.items():
                self.data[k] += (v * len(df_sub)) / total
        self.normalize()


def get_df():
    path = os.path.join('data', 'WIKTI', 'data.obj')
    # path = os.path.join('data', 'Wikispeedia', 'data.obj')
    df = pd.read_pickle(path)
    df = df[~df['backtrack']]
    # includes both successful and unsuccessful games, all for spl = 3
    features = [
        'degree_out',
        'degree_in',
        'ngram',
        'view_count',
    ]
    base = ['node', 'linkpos_first', 'linkpos_last', 'subject', 'word_count']
    if 'linkpos_actual' in df.columns:
        base += ['linkpos_actual']
    df = df[base + features]
    df['ambiguous'] = df['linkpos_first'] != df['linkpos_last']
    # for f in features:
    #     df[f + '_next'] = df[f].shift(-1)
    # df = df.dropna()

    df['node_next'] = df['node'].shift(-1)
    df = df.dropna()
    return df

if __name__ == '__main__':
    df = get_df()
    gt = GroundTruthModel(df)
    models = [
        UniformModel,
        # IndirectProportionModel,
        # InverseRankModel,
        InverseModel,
    ]
    models = [m(df) for m in models]
    for m in models:
        gt.compare(m)
