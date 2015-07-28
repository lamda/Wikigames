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


class GroundTruthModel(ClickModel):
    def __init__(self, df):
        self.label = 'GroundTruth'
        super(GroundTruthModel, self).__init__(df)
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


def get_df_wikigame():
    # path = os.path.join('data', 'Wikispeedia', 'data.obj')
    # df = pd.read_pickle(path)
    # df = df[['node', 'node_id', 'linkpos_first', 'linkpos_last', 'linkpos_all',
    #          'backtrack', 'subject', 'word_count']]
    # df['target'] = df['node'].shift(-1)
    # df['target_id'] = df['node_id'].shift(-1)
    # df.columns = ['source', 'source_id'] + df.columns[2:].tolist()
    # df = df[~df['backtrack']]
    # df = df[['source', 'source_id', 'linkpos_first', 'linkpos_last',
    #          'linkpos_all', 'word_count', 'target', 'target_id']]
    # df['linkpos_ambig'] = df['linkpos_first'] != df['linkpos_last']
    # df = df.dropna()
    # df['amount'] = df.groupby(['source', 'target']).transform('count')['source_id']
    # df = df.drop_duplicates(subset=['source', 'target'])
    # df.to_pickle(os.path.join('data', 'Wikispeedia', 'data_clicks.obj'))

    df = pd.read_pickle(os.path.join('data', 'Wikispeedia', 'data_clicks.obj'))
    path = os.path.join('data', 'Wikispeedia', 'link_positions.obj')
    with open(path, 'rb') as infile:
        link2pos_first, link2pos_last, length, pos2link, pos2linklength,\
        ib_length, lead_length = pickle.load(infile)
    return df, ib_length, lead_length


def get_df_wikipedia(smoothed=False):
    # includes all paths of length > 1
    fname = 'DataHandler_aggregate_clicks' + ('_smoothed' if smoothed else '')
    fpath = os.path.join('clickstream', fname + '.obj')
    df = pd.read_pickle(fpath).values()[0]
    if smoothed:
        df.ix[df['amount'] == 0, 'amount'] = 10

    fpath = os.path.join('clickstream', 'DataHandler_get_ib_lengths.obj')
    ib_length = pd.read_pickle(fpath).values()[0]

    fpath = os.path.join('clickstream', 'DataHandler_get_lead_lengths.obj')
    lead_length = pd.read_pickle(fpath).values()[0]
    df['linkpos_ambig'] = df['linkpos_first'] != df['linkpos_last']
    return df, ib_length, lead_length


def plot_click_pos():
    # for func, label in [
    #     (get_df_wikigame, 'Wikispeedia'),
    #     (get_df_wikipedia, 'Wikipedia'),
    #     # (get_df_wikipedia(smoothed=True), 'Wikipedia (smoothed)'),
    # ]:
    #     print(label)
    #     df, ib_length, lead_length = func()
    #     # df = df.iloc[:250]
    #
    #     first_a, uniform_a, last_a = [], [], []
    #     first_r, uniform_r, last_r = [], [], []
    #     for ridx, row in enumerate(df.iterrows()):
    #         print ('   ', ridx+1, '/', df.shape[0], end='\r')
    #         row = row[1]
    #         first_a += [row['linkpos_first']] * row['amount']
    #         first_r += [row['linkpos_first']/row['word_count']] * row['amount']
    #
    #         last_a += [row['linkpos_last']] * row['amount']
    #         last_r += [row['linkpos_last']/row['word_count']] * row['amount']
    #
    #         for i in range(row['amount']):
    #             pos = np.random.choice(row['linkpos_all'])
    #             uniform_a.append(pos)
    #             uniform_r.append(pos/row['word_count'])
    #
    #     for data, suffix, ylim in [
    #         ([first_a, uniform_a, last_a], 'absolute', (-400, 16400)),
    #         ([first_r, uniform_r, last_r], 'relative', (-0.025, 1.025)),
    #     ]:
    #         df = pd.DataFrame(data=zip(*data),
    #                           columns=['first', 'uniform', 'last'])
    #         fpath = os.path.join('data', 'clickmodels')
    #         if not os.path.exists(fpath):
    #             os.makedirs(fpath)
    #         df.to_pickle(os.path.join(fpath, suffix + '_' + label + '.obj'))

    for label in [
        'Wikispeedia',
        'Wikipedia',
        # 'Wikipedia (smoothed)',
    ]:
        print(label)
        for suffix, ylim in [
            # ('absolute', (-400, 40000)),
            ('relative', (-0.025, 1.025)),
        ]:
            print('   ', suffix)
            fpath = os.path.join('data', 'clickmodels',
                suffix + '_' + label + '.obj')
            df = pd.read_pickle(fpath)
            print('       loaded')
            plt.clf()
            sns.boxplot(vals=df)
            plt.title(label + ' (' + suffix + ')')
            fname = 'clicks_' + suffix + '_' + label + '.png'
            plt.ylim(ylim)
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(os.path.join('plots', fname))


def plot_ib_lead_clicks():
    # for func, label in [
    #     (get_df_wikigame, 'Wikispeedia'),
    #     (get_df_wikipedia, 'Wikipedia'),
    #     # (get_df_wikipedia(smoothed=True), 'Wikipedia (smoothed)'),
    # ]:
    #     print(label)
    #     df, ib_length, lead_length = func()
    #     # df = df.iloc[:25]
    #     try:
    #         df['ib_length'] = map(lambda x: ib_length[x], df['source'])
    #         df['lead_length'] = map(lambda x: lead_length[x], df['source'])
    #     except KeyError as e:
    #         print(e)
    #         pdb.set_trace()
    #
    #     first_ib, first_lead = 0, 0
    #     last_ib, last_lead = 0, 0
    #     uniform_ib, uniform_lead = 0, 0
    #     for ridx, row in enumerate(df.iterrows()):
    #         print ('   ', ridx+1, '/', df.shape[0], end='\r')
    #         row = row[1]
    #
    #         # first
    #         if row['linkpos_first'] < row['ib_length']:
    #             first_ib += row['amount']
    #         if row['ib_length'] < row['linkpos_first'] < row['lead_length']:
    #             first_lead += row['amount']
    #
    #         # last
    #         if row['linkpos_last'] < row['ib_length']:
    #             last_ib += row['amount']
    #         if row['ib_length'] < row['linkpos_last'] < row['lead_length']:
    #             last_lead += row['amount']
    #
    #         # uniform
    #         ib = len([l for l in row['linkpos_all'] if l < row['ib_length']])
    #         ib /= len(row['linkpos_all'])
    #         uniform_ib += ib * row['amount']
    #
    #         lead = len([l for l in row['linkpos_all']
    #                     if row['ib_length'] < l < row['lead_length']])
    #         lead /= len(row['linkpos_all'])
    #         uniform_lead += lead * row['amount']
    #
    #     data = [first_ib, uniform_ib, last_ib,
    #             first_lead, uniform_lead, last_lead]
    #     total = df['amount'].sum()
    #     data = [d / total for d in data]
    #     columns = ['ib_first', 'ib_uniform', 'ib_last',
    #                'lead_first', 'lead_uniform', 'lead_last']
    #     df = pd.Series(data=data, index=columns)
    #     fpath = os.path.join('data', 'clickmodels')
    #     if not os.path.exists(fpath):
    #         os.makedirs(fpath)
    #     df.to_pickle(os.path.join(fpath, label + '_ib_lead.obj'))

    colors = ["#8DA0CB", "#66C2A5", "#FC8D62", "#E78AC3"]
    columns = ['ib_first', 'ib_uniform', 'ib_last',
               'lead_first', 'lead_uniform', 'lead_last']
    labels = ['IB (first)', 'IB (uniform)', 'IB (last)',
              'Lead (first)', 'Lead (uniform)', 'Lead (last)']
    fig, ax = plt.subplots(figsize=(9, 4))
    ind = np.arange(float(len(columns)))
    ind[len(ind)/2:] += 1
    width = 0.25
    rects = []
    hatch = ''

    for label in [
        'Wikipedia',
        'Wikispeedia',
    ]:
        fpath = os.path.join('data', 'clickmodels',
                             label + '_ib_lead.obj')
        df = pd.read_pickle(fpath)
        for fix, feature in enumerate(columns):
            val = [100 * df[feature]]
            rect = ax.bar(ind[fix], val, width, hatch=hatch,
                          facecolor=colors[fix % 3], edgecolor='white')
            rects.append(rect)
        ind += 0.25
        hatch = '///'

    ax.set_ylabel('Percent')
    ax.set_xticks(ind-0.25)
    ax.set_xticklabels(labels)
    dummy = matplotlib.patches.Rectangle((0, 0), 1, 1, fill=False,
                                         edgecolor='none', visible=False)
    ax.legend((rects[0], rects[1], rects[2],
               dummy, rects[6],
               rects[7], rects[8]),
              ('First (Wikipedia)', 'Uniform (Wikipedia)', 'Last (Wikipedia)',
               '', 'First (Wikispeedia)',
               'Uniform (Wikispeedia)', 'Last (Wikispeedia)'),
              loc=0)
    plt.ylim(0, 50)
    # plt.show()
    # plt.title('Clicks to Infobox and Lead')
    fname = 'clicks_ib_lead.png'
    plt.savefig(os.path.join('plots', fname))


def get_ambiguous_links():
    for func, label in [
        # (get_df_wikigame, 'Wikispeedia'),
        (get_df_wikipedia, 'Wikipedia'),
        # (get_df_wikipedia(smoothed=True), 'Wikipedia (smoothed)'),
    ]:
        print(label)
        df, ib_length, lead_length = func()
        pdb.set_trace()
        total_links = df.shape[0]
        total_ambig = df[df['linkpos_ambig']].shape[0]
        total_amount = df['amount'].sum()
        total_ambig_amount = df[df['linkpos_ambig']]['amount'].sum()
        print('%.2f%% ambiguous links, %.2f%% ambiguous clicks\n' %
              (100 * total_ambig / total_links,
               100 * total_ambig_amount / total_amount))

if __name__ == '__main__':
    # unambig = True
    # df = get_df_wikigame()[0]
    # # df = get_df_wikipedia()[0]
    # gt = GroundTruthModel(df)
    #
    # models = [
    #     UniformModel,
    #     IndirectProportionModel,
    #     InverseRankModel,
    #     InverseModel,
    # ]
    # models = [m(df) for m in models]
    # for m in models:
    #     gt.compare(m)

    # plot_click_pos()
    # plot_ib_lead_clicks()
    get_ambiguous_links()
