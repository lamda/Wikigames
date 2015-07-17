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


class ClickModel(object):
    def __init__(self, df, unambig=False):
        self.df = df
        if unambig:
            pdb.set_trace()
        self.data = collections.defaultdict(float)
        self.keys = set(df['source'])
        self.grouped = df.groupby('source')
        self.total_clicks = {k: self.grouped.get_group(k)['amount'].sum()
                             for k in self.keys}

    def normalize(self):
        total = sum(self.data.values())
        self.data = [self.data[k] / total for k in sorted(self.data)]


class GroundTruthModel(ClickModel):
    def __init__(self, df, unambig=False):
        self.label = 'GroundTruth'
        super(GroundTruthModel, self).__init__(df, unambig)
        self.data = {k: v for
                     k, v in df.groupby('target')['amount'].sum().iteritems()}
        self.normalize()
        print('click data for %d pages' % len(self.data))

    def compare(self, mdl):
        kl = np.abs(scipy.stats.entropy(self.data, mdl.data, base=2))
        lab = self.label + '\t' + mdl.label
        print('\t%.4f\t%s' % (kl, lab))


class UniformModel(ClickModel):
    def __init__(self, df):
        super(UniformModel, self).__init__(df)
        self.label = 'Uniform'
        for kidx, key in enumerate(self.keys):
            print(kidx+1, '/', len(self.keys), end='\r')
            df_sub = self.grouped.get_group(key)
            targets = len(df_sub['target'])
            for ridx, row in df_sub.iterrows():
                self.data[row['target']] += self.total_clicks[key]/targets
        self.normalize()


class IndirectProportionModel(ClickModel):
    def __init__(self, df):
        super(IndirectProportionModel, self).__init__(df)
        self.label = 'Indirect Proportional'
        for kidx, key in enumerate(self.keys):
            print(kidx+1, '/', len(self.keys), end='\r')
            df_sub = self.grouped.get_group(key)
            node2p = collections.defaultdict(float)
            for ridx, row in df_sub.iterrows():
                node2p[row['target']] += 1 / (row['linkpos_first'] + 1)
                if row['linkpos_ambig']:
                    node2p[row['target']] += 1 / (row['linkpos_last'] + 1)
            total = sum(node2p.values())
            for k, v in node2p.items():
                self.data[k] += self.total_clicks[key] * v / total
        self.normalize()


class InverseRankModel(ClickModel):
    def __init__(self, df):
        super(InverseRankModel, self).__init__(df)
        self.label = 'Inverse Rank'  # similar to Zipf's law
        for kidx, key in enumerate(self.keys):
            print(kidx+1, '/', len(self.keys), end='\r')
            df_sub = self.grouped.get_group(key)
            nodepos = set()
            for ridx, row in df_sub.iterrows():
                nodepos.add((row['linkpos_first'] + 1, row['target']))
                if row['linkpos_ambig']:
                    nodepos.add((row['linkpos_last'] + 1, row['target']))
            total = sum(1/(idx+1) for idx in range(len(nodepos)))
            for idx, e in enumerate(sorted(nodepos)):
                self.data[e[1]] += self.total_clicks[key] * (1/(idx+1)) / total
        self.normalize()


class InverseModel(ClickModel):
    def __init__(self, df):
        super(InverseModel, self).__init__(df)
        self.label = 'InverseModel'
        for kidx, key in enumerate(self.keys):
            print(kidx, '/', len(self.keys), end='\r')
            df_sub = self.grouped.get_group(key)
            node2p = collections.defaultdict(float)
            for ridx, row in df_sub.iterrows():
                rt = row['target']
                node2p[rt] += row['word_count'] - row['linkpos_first']
                if row['linkpos_ambig']:
                    node2p[rt] += row['word_count'] - row['linkpos_last']
            total = sum(node2p.values())
            for k, v in node2p.items():
                self.data[k] += v * self.total_clicks[key] / total
        self.normalize()


class PSkipModel(ClickModel):
    def __init__(self):
        # introduce a parameter to only run models on unambiguous clicks
        # estimate on all unambiguous clicks and apply to all clicks
        # for each click: get r=number of skipped possible clicks before click
        # learn alpha = (\sum_i (r_i p(x_i))) / (\sum_i (1+r_i) p(x_i))
        pass


def get_df_wikigame(wikti=False):
    # includes both successful and unsuccessful games, all for spl = 3
    if wikti:
        path = os.path.join('data', 'WIKTI', 'data.obj')
    else:
        path = os.path.join('data', 'Wikispeedia', 'data.obj')
    df = pd.read_pickle(path)
    df = df[['node', 'node_id', 'linkpos_first', 'linkpos_last', 'linkpos_all',
             'backtrack', 'subject', 'word_count']]
    df['target'] = df['node'].shift(-1)
    df['target_id'] = df['node_id'].shift(-1)
    df.columns = ['source', 'source_id'] + df.columns[2:].tolist()
    df = df[~df['backtrack']]
    df = df[['source', 'source_id', 'linkpos_first', 'linkpos_last',
             'linkpos_all', 'word_count', 'target', 'target_id']]
    df['linkpos_ambig'] = df['linkpos_first'] != df['linkpos_last']
    df = df.dropna()
    df['amount'] = df.groupby(['source', 'target']).transform('count')['source_id']
    df = df.drop_duplicates(subset=['source', 'target'])
    return df


def get_df_wikipedia(smoothed=False):
    # includes all paths of length > 1
    fname = 'DataHandler_aggregate_clicks' + ('_smoothed' if smoothed else '')
    path = os.path.join('clickstream', fname + '.obj')
    df = pd.read_pickle(path).values()[0]
    if smoothed:
        df.ix[df['amount'] == 0, 'amount'] = 10
    return df


if __name__ == '__main__':
    unambig = True
    df = get_df_wikigame(wikti=True)
    df = get_df_wikipedia()
    gt = GroundTruthModel(df, unambig=unambig)

    models = [
        UniformModel,
        IndirectProportionModel,
        InverseRankModel,
        InverseModel,
    ]
    models = [m(df, unambig=unambig) for m in models]
    for m in models:
        gt.compare(m)
