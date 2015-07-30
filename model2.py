# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

import collections
import cPickle as pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pdb
import scipy.stats
import seaborn as sns


pd.options.mode.chained_assignment = None
pd.set_option('display.width', 400)


class ClickModel(object):
    def __init__(self, df):
        self.df = df
        self.data = collections.defaultdict(float)
        self.keys = set(df['source'])
        self.grouped = df.groupby('source')
        self.total_clicks = {k: self.grouped.get_group(k)['amount'].sum()
                             for k in self.keys}

    def normalize(self):
        total = sum(self.data.values())
        self.data = [self.data[k] / total for k in sorted(self.data)]

    def compare(self, mdl):
        kl = np.abs(scipy.stats.entropy(self.data, mdl.data, base=2))
        lab = self.label + '\t' + mdl.label
        print('\t%.4f\t%s' % (kl, lab))


class GroundTruthModel(ClickModel):
    def __init__(self, df):
        self.label = 'GroundTruth'
        super(GroundTruthModel, self).__init__(df)
        self.data = {k: v for
                     k, v in df.groupby('target')['amount'].sum().iteritems()}
        self.normalize()
        print('click data for %d pages' % len(self.data))


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


class LeadModel(ClickModel):
    def __init__(self, df, leadp=70):
        super(LeadModel, self).__init__(df)
        self.label = 'Uniform'
        self.leadp = 70
        for kidx, key in enumerate(self.keys):
            print(kidx+1, '/', len(self.keys), end='\r')
            df_sub = self.grouped.get_group(key)
            targets = len(df_sub['target'])
            pdb.set_trace()
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


def get_df_wikigame(smoothed=False):
    path = os.path.join('data', 'cache', 'Wikispeedia_get_model_df.obj')
    with open(path, 'rb') as infile:
        df, ib_length, lead_length = pickle.load(infile)
    if not smoothed:
        df = df[df['amount']]
    return df, ib_length, lead_length


def get_df_wikipedia(smoothed=False):
    fname = 'DataHandler_aggregate_clicks' + ('_smoothed' if smoothed else '')
    fpath = os.path.join('clickstream', fname + '.obj')
    df = pd.read_pickle(fpath).values()[0]

    fpath = os.path.join('clickstream', 'DataHandler_get_ib_lengths.obj')
    ib_length = pd.read_pickle(fpath).values()[0]

    fpath = os.path.join('clickstream', 'DataHandler_get_lead_lengths.obj')
    lead_length = pd.read_pickle(fpath).values()[0]

    return df, ib_length, lead_length


if __name__ == '__main__':
    df = get_df_wikigame()[0]
    # df = get_df_wikipedia(smoothed=True)[0]
    pdb.set_trace()
    # gt = GroundTruthModel(df)
    #
    models = [
        UniformModel,
    #     IndirectProportionModel,
    #     InverseRankModel,
    #     InverseModel,
        LeadModel,
    ]
    models = [m(df) for m in models]
    for m in models:
        gt.compare(m)
