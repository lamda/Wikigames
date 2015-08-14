# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

import collections
import cPickle as pickle
import matplotlib.pyplot as plt
plt.style.use('ggplot')  # TODO
import numpy as np
import os
import pandas as pd
import pdb
import scipy.stats
# import seaborn as sns

import decorators

pd.options.mode.chained_assignment = None
pd.set_option('display.width', 400)


class ClickModel(object):
    def __init__(self, dataset, kind=None, step=None, spl=None, pl=None):
        self.dataset = dataset
        self.kind = kind
        self.step = step
        self.spl = spl
        self.pl = pl
        self.suffix = self.get_suffix()
        if dataset == 'wikispeedia':
            fpath = 'data/clickmodels/wikispeedia_' + kind + self.suffix +\
                    '.obj'
        elif dataset == 'wikipedia':
            fpath = 'data/clickmodels/wikipedia_all.obj'
        else:
            print('unrecognized parameter')
            raise NotImplemented
        self.df = pd.read_pickle(fpath)
        with open('data/clickmodels/' + dataset + '_stats.obj', 'rb') as infile:
            stats_orig = pickle.load(infile)
        self.stats = {}
        for key, val in stats_orig.iteritems():
            self.stats[key] = {k: v for k, v in val.iteritems()}
            # self.stats[key] = {k: np.log(v+1) for k, v in val.iteritems()}
            # self.stats[key] = {k: 1/(v+1) for k, v in val.iteritems()}
        self.sources = set(self.df['source'])
        self.targets = set(self.df['target'])
        self.keys = sorted(self.sources | self.targets)
        func_dict = lambda: {k: 0.01 for k in self.keys}
        self.data = collections.defaultdict(func_dict)
        self.df_source = self.df.groupby('source')
        self.df_target = self.df.groupby('target')
        self.clicks = {key: self.df_source.get_group(key)['amount'].sum()
                       for key in self.sources}
        self.clicks = {k: v for k, v in self.clicks.items() if v > 0}
        self.wg = None
        self.linkpos_type = 'linkpos_first'

    def get_suffix(self):
        suffix = ''
        if self.step is not None:
            suffix += '_step_' + unicode(self.step)
        if self.spl is not None:
            suffix += '_spl_' + unicode(self.spl)
        if self.pl is not None:
            suffix += '_pl_' + unicode(self.pl)
        return suffix

    def update_data(self, label, data2):
        for k, v in data2.iteritems():
            self.data[label][k] += v

    def normalize(self):
        for label in self.data:
            total = sum(self.data[label].values())
            self.data[label] = [self.data[label][k] / total for k in self.keys]

    def compare(self, m2):
        m1 = 'Ground Truth'
        kl = np.abs(scipy.stats.entropy(self.data[m1], self.data[m2], base=2))
        print('        %.3f\t%s' % (kl, m2))
        return kl

    def compare_all(self):
        self.normalize()
        for key in sorted(self.data):
            self.compare(key)

    def ground_truth(self):
        iterable = self.df_target['amount'].sum().iteritems()
        self.update_data('Ground Truth', {k: v for k, v in iterable})

    def uniform(self):
        for key, clicks_total in self.clicks.items():
            df_sub = self.df_source.get_group(key)
            lps = df_sub[self.linkpos_type].apply(len)
            lp_total = lps.sum()
            for k, v in zip(df_sub['target'], lps):
                self.data['Uniform'][k] += clicks_total * v / lp_total

    def area(self, area, leadp=0.5):
        assert leadp <= 1
        for key, clicks_total in self.clicks.items():
            df_sub = self.df_source.get_group(key)
            dkey = area + '_' + unicode(leadp)

            # links within area of interest
            lps = df_sub['linkpos_' + area].apply(len)
            lp_total = lps.sum()
            if lp_total != 0:
                for k, v in zip(df_sub['target'], lps):
                    self.data[dkey][k] += leadp * clicks_total * v / lp_total

            # links outside area of interest
            not_area = 'not_' + area if 'not' not in area else area[4:]
            lps = df_sub['linkpos_' + not_area].apply(len)
            lp_total = lps.sum()
            if lp_total != 0:
                for k, v in zip(df_sub['target'], lps):
                    self.data[dkey][k] += (1-leadp) * clicks_total * v / lp_total

    def proportional(self, stat, stat_label):
        for key, clicks_total in self.clicks.items():
            df_sub = self.df_source.get_group(key)
            lps = df_sub[self.linkpos_type].apply(len)
            prop = df_sub['target'].apply(lambda x: self.stats[stat][x])
            total = np.dot(lps, prop)
            for k, l, d in zip(df_sub['target'], lps, prop):
                self.data[stat_label][k] += clicks_total * l * d / total

    def tfidf(self):
        assert self.dataset == 'wikispeedia'
        from main import Wikispeedia
        self.wg = Wikispeedia()
        self.wg.load_data()
        df = self.wg.data
        df = df[~df['backtrack']]
        df = df[df['successful'] == (self.kind == 'successful')]
        if self.step is not None:
            df = df[df['step'] == self.step]
        if self.spl is not None:
            df = df[df['spl'] == self.spl]
        if self.pl is not None:
            df = df[df['pl'] == self.pl]
        df = df[['node', 'node_id', 'target', 'target_id']]

        for idx, row in enumerate(df.iterrows()):
            print(idx+1, '/', df.shape[0], end='\r')
            row = row[1]
            weights = self.get_tfidf_prob(row['node'], row['node_id'],
                                          row['target_id'])
            for k, v in weights.iteritems():
                self.data['TF-IDF'][k] += v

    def get_tfidf_prob(self, node, node_id, target_id):
        df_sub = self.df_source.get_group(node)
        lps = df_sub[self.linkpos_type].apply(len)
        lmbd = lambda x: self.wg.get_tfidf_similarity(node_id, x)
        prop = df_sub['target_id'].apply(lmbd)
        total = np.dot(lps, prop)
        dct = {}
        for k, l, d in zip(df_sub['target'], lps, prop):
            dct[k] = l * d / total
        return dct

    def run(self):
        # print('getting Ground Truth...')
        self.ground_truth()
        # print('getting Uniform...')
        self.uniform()
        # print('getting degree...')
        self.proportional('deg_in', 'In-Degree')
        # print('getting N-Gram...')
        self.proportional('ngram', 'N-Gram')
        # print('getting View Count...')
        self.proportional('view_count', 'View Count')
        # print('getting TF-IDF...')
        self.tfidf()
        # print('getting areas...')
        # for area in [
        #     'lead',
        #     'ib',
        #     # 'ib_lead',
        # ]:
        #     print('    ', area, '\n')
        #     for areap in np.arange(0, 1, 0.01):
        #     # for areap in np.arange(0, 1, 0.25):
        #         print('        ', areap, end='\r')
        #         self.area(area, areap)
        # print()
        self.normalize()
        columns = sorted(self.data)
        columns.remove('Ground Truth')
        data = [self.compare(key) for key in columns]
        se = pd.Series(data=data, index=columns)
        se.to_pickle(
            'data/clickmodels/' + self.dataset + '_results' +
            ('_' + self.kind if self.kind is not None else '') + self.suffix +
            '.obj'
        )


def plot_results(dataset, kind=None, other=True, normalized=False,
                 step=None, spl=None, pl=None):
    suffix = ''
    if step is not None:
        suffix += '_step_' + unicode(step)
    if spl is not None:
        suffix += '_spl_' + unicode(spl)
    if pl is not None:
        suffix += '_pl_' + unicode(pl)
    se_full = pd.read_pickle(
        'data/clickmodels/' + dataset + '_results' +
        ('_' + kind if kind is not None else '') + suffix + '.obj'
    )
    se_filtered = se_full[[c for c in se_full.index if '.' not in c]]
    pdb.set_trace()
    keys = [
        'ib',
        'lead',
        # 'ib_lead',
    ]
    if other:
        # data, columns = se_filtered.values.tolist(), se_filtered.index.tolist()
        # make sure View Count comes before Uniform
        columns = ['Indegree', 'View Count', 'Bing N-Grams', 'Uniform']
        data = se_filtered[columns].values.tolist()
    else:
        data, columns = [], []
    for key in keys:
        data.append(se_full.filter(regex=r'^' + key + '_\d').min())
        columns.append(se_full.filter(regex=r'^' + key + '_\d').idxmin())
    se = pd.Series(data=data, index=columns)
    if normalized:
        # via http://math.stackexchange.com/questions/51482
        se = se.apply(lambda x: 1 - np.exp(-x))
    print('\n\n', dataset, kind, '\n', se)
    ax = plt.subplot(111)
    se.plot(ax=ax, kind='bar', legend=False, width=0.6, rot=70, fontsize=18)
    plt.tight_layout()
    plt.ylim(0, max(se) * 1.075)
    label_offset = max(se) * 0.01
    for p in ax.patches:
        ax.annotate(
            '%.2f' % p.get_height(),
            (p.get_x() + p.get_width() / 2., p.get_height() + label_offset),
            ha='center',
            fontsize=14,
        )
    ofname = 'plots/clickmodels_' + dataset +\
             ('_normalized' if normalized else '') +\
             ('_' + kind if kind is not None else '') + suffix
    plt.savefig(ofname + '.pdf')
    plt.savefig(ofname + '.png')
    plt.close()


def get_area_importance():
    for dataset in ['wikispeedia', 'wikipedia']:
        df = pd.read_pickle('data/clickmodels/' + dataset + '_all.obj')
        lp_ib = df['linkpos_ib'].apply(len).sum()
        lp_lead = df['linkpos_lead'].apply(len).sum()
        lp_all = df['linkpos_all'].apply(len).sum()
        print('%.2f (IB) %.2f (LEAD)' % (lp_ib/lp_all, lp_lead/lp_all))


if __name__ == '__main__':
    # get_area_importance()

    # cm = ClickModel('wikipedia'); cm.run_all()
    # plot_results('wikipedia', normalized=False)
    #
    # for kind in [
    #     # 'all',
    #     # 'successful',
    #     'unsuccessful'
    # ]:
    #     cm = ClickModel('wikispeedia', kind); cm.run()
    #     # plot_results('wikispeedia', kind, normalized=False)

    print('SPL = 3')
    for step in range(3):
        print('    STEP =', step)
        cm = ClickModel('wikispeedia', 'successful', step=step, spl=3, pl=4)
        cm.run()
        print('\n\n')

    # print('SPL = 4')
    # for step in range(4):
    #     print('    STEP =', step)
    #     cm = ClickModel('wikispeedia', 'successful', step=step, spl=3, pl=4)
    #     cm.run()
    #     print('\n\n')

    # cm.run()
    # plot_results('wikispeedia', 'successful', step=0, spl=3, pl=4)
