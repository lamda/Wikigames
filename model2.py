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

pd.options.mode.chained_assignment = None
pd.set_option('display.width', 400)


class ClickModel(object):
    def __init__(self, dataset, kind=None):
        self.dataset = dataset
        self.kind = kind
        if dataset == 'wikispeedia':
            fpath = 'data/clickmodels/wikispeedia_' + kind + '.obj'
        elif dataset == 'wikipedia':
            fpath = 'clickstream/DataHandler_aggregate_clicks_smoothed.obj'
        else:
            print('unrecognized parameter')
            raise NotImplemented
        self.df = pd.read_pickle(fpath)
        with open('data/clickmodels/' + dataset + '_stats.obj', 'rb') as infile:
            self.stats = pickle.load(infile)
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

    def get_df_wikigame(self, kind):
        df = pd.read_pickle('data/clickmodels/wikispeedia_' + kind + '.obj')
        return df

    def get_df_wikipedia(self):
        fpath = os.path.join('clickstream',
                             'DataHandler_aggregate_clicks_smoothed.obj')
        df = pd.read_pickle(fpath).values()[0]
        return df

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
        print('\t%.4f\t%s' % (kl, m2))
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
            lps = df_sub['linkpos_all'].apply(len)
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
            lps = df_sub['linkpos_all'].apply(len)
            prop = df_sub['target'].apply(lambda x: self.stats[stat][x])
            total = np.dot(lps, prop)
            for k, l, d in zip(df_sub['target'], lps, prop):
                self.data[stat_label][k] += clicks_total * l * d / total

    def run(self):
        self.ground_truth()
        self.uniform()
        self.proportional('ngram', 'N-Gram')
        self.proportional('deg_in', 'In-Degree')
        self.proportional('view_count', 'View Count')

        self.compare_all()

    def run_all(self):
        self.ground_truth()
        self.uniform()
        self.proportional('ngram', 'N-Gram')
        self.proportional('deg_in', 'In-Degree')
        self.proportional('view_count', 'View Count')
        for area in [
            'lead',
            'ib',
            'ib_lead',
        ]:
            print('    ', area)
            for areap in np.arange(0, 1, 0.01):
                print('        ', areap)
                self.area(area, areap)
        print()
        self.normalize()
        columns = sorted(self.data)
        columns.remove('Ground Truth')
        data = [self.compare(key) for key in columns]
        se = pd.Series(data=data, index=columns)
        se.to_pickle(
            'data/clickmodels/' + self.dataset + '_results' +
            ('_' + self.kind if self.kind is not None else '') + '.obj'
        )


def plot(dataset, kind=None, other=True):
    se_full = pd.read_pickle(
        'data/clickmodels/' + dataset + '_results' +
        ('_' + kind if kind is not None else '') + '.obj'
    )
    se_filtered = se_full[[c for c in se_full.index if '.' not in c]]
    keys = ['ib', 'lead', 'ib_lead']
    if other:
        data, columns = se_filtered.values.tolist(), se_filtered.index.tolist()
    else:
        data, columns = [], []
    for key in keys:
        data.append(se_full.filter(regex=key + '_\d').min())
        columns.append(se_full.filter(regex=key + '_\d').idxmin())
    se = pd.Series(data=data, index=columns)
    ax = plt.subplot(111)
    se.plot(ax=ax, kind='bar', legend=False, width=0.5, rot=70)
    plt.tight_layout()
    # plt.show()
    plt.savefig(
        'plots/clickmodels_' + dataset +
        ('_' + kind if kind is not None else '') + '.png'
    )

if __name__ == '__main__':
    # cm = ClickModel('wikipedia'); cm.run_all()
    for kind in [
        # 'all',
        # 'successful',
        'unsuccessful'
    ]:
        cm = ClickModel('wikispeedia', kind); cm.run_all()
        plot('wikispeedia', kind)
