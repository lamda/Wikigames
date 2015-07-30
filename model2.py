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
    def __init__(self, df, keys):
        self.label = ''
        self.df = df
        self.data = {k: 0 for k in keys}
        self.keys = set(df['source'])
        self.grouped = df.groupby('source')
        self.total_clicks = {k: self.grouped.get_group(k)['amount'].sum()
                             for k in self.keys}

    def normalize(self):
        pdb.set_trace()
        total = sum(self.data.values())
        self.data = [self.data[k] / total for k in sorted(self.data)]

    def compare(self, mdl):
        kl = np.abs(scipy.stats.entropy(self.data, mdl.data, base=2))
        lab = self.label + '\t' + mdl.label
        print('\t%.4f\t%s' % (kl, lab))


class GroundTruthModel(ClickModel):
    def __init__(self, df, keys):
        self.label = 'GroundTruth'
        print(self.label)
        super(GroundTruthModel, self).__init__(df, keys)
        iterable = df.groupby('target')['amount'].sum().iteritems()
        self.data.update({k: v for k, v in iterable})
        self.normalize()


class UniformModel(ClickModel):
    def __init__(self, df, keys):
        self.label = 'Uniform'
        print(self.label)
        super(UniformModel, self).__init__(df, keys)
        vals = pd.DataFrame(
            data=zip(df['target'], df['linkpos_all'].apply(len)),
            columns=['node', 'amount']
        )
        groupby = vals.groupby('node').sum()
        result = {k: v for k, v in zip(groupby.index, groupby['amount'])}
        self.data.update(result)
        self.normalize()


class LeadModel(ClickModel):
    def __init__(self, df, keys, leadp=60):
        super(LeadModel, self).__init__(df, keys)
        self.label = 'Lead (' + unicode(leadp) + '% weight on lead links)'

        # Lead links
        vals = pd.DataFrame(
            data=zip(df['target'], df['linkpos_lead'].apply(len)),
            columns=['node', 'amount']
        )
        groupby = vals.groupby('node').sum()
        result = {k: v * leadp
                  for k, v in zip(groupby.index, groupby['amount'])}
        self.data.update(result)

        # Not-Lead links
        vals = pd.DataFrame(
            data=zip(df['target'], df['linkpos_not_lead'].apply(len)),
            columns=['node', 'amount']
        )
        groupby = vals.groupby('node').sum()
        result = {k: v * (100-leadp)
                  for k, v in zip(groupby.index, groupby['amount'])}
        for r in result:
            self.data[r] += result[r]
        self.normalize()


def get_df_wikigame(smoothed=True):
    path = os.path.join('cache', 'Wikispeedia_get_model_df.obj')
    with open(path, 'rb') as infile:
        df, keys = pickle.load(infile).values()[0]
    if not smoothed:
        df = df[df['amount'] > 0]
    return df, keys


def get_df_wikipedia(smoothed=True):
    fpath = os.path.join('clickstream',
                         'DataHandler_aggregate_clicks_smoothed.obj')
    df = pd.read_pickle(fpath).values()[0]

    fpath = os.path.join('clickstream', 'DataHandler_get_keys.obj')
    keys = pd.read_pickle(fpath).values()[0]
    if not smoothed:
        df = df[df['amount'] > 0]
    return df, keys


if __name__ == '__main__':
    # df, keys = get_df_wikigame(smoothed=True)
    print(1)
    df, keys = get_df_wikipedia(smoothed=True)
    # TODO: Wikipedia DataFrame has targets outside the Wikispeedia set
    # FIXME: ?
    print(2)

    gt = GroundTruthModel(df.iloc[950000:955000], keys)
    pdb.set_trace()
    # # df = df.iloc[950000:955000]
    # lm = LeadModel(df.iloc[950000:955000], keys, 90); gt.compare(lm)
    # gt.compare(lm)

    gt = GroundTruthModel(df, keys)
    # print(3)
    # um = UniformModel(df, keys)
    # gt.compare(um)
    for leadp in [
        90,
        50,
        10,
    ]:
        # print(leadp)
        lm = LeadModel(df, keys, leadp)
        gt.compare(lm)
    pdb.set_trace()



    # models = [
    #     UniformModel,
    # #     IndirectProportionModel,
    # #     InverseRankModel,
    # #     InverseModel,
    #     LeadModel,
    # ]
    # models = [m(df) for m in models]
    # for m in models:
    #     gt.compare(m)
