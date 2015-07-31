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

from main import Wikispeedia

pd.options.mode.chained_assignment = None
pd.set_option('display.width', 400)


class ClickModel(object):
    def __init__(self, df):
        self.label = ''
        self.df = df
        self.data = {k: 0.01 for k in set(df['target'])}

    def update_data(self, data2):
        for k, v in data2.iteritems():
            self.data[k] += v

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
        print(self.label)
        super(GroundTruthModel, self).__init__(df)
        iterable = df.groupby('target')['amount'].sum().iteritems()
        self.update_data({k: v for k, v in iterable})
        self.normalize()


class UniformModel(ClickModel):
    def __init__(self, df):
        self.label = 'Uniform'
        print(self.label)
        super(UniformModel, self).__init__(df)
        vals = pd.DataFrame(data=zip(df['target'], df['linkpos_all'].apply(len)), columns=['node', 'amount'])
        groupby = vals.groupby('node').sum()
        result = {k: v for k, v in zip(groupby.index, groupby['amount'])}
        self.update_data(result)
        self.normalize()


class LeadModel(ClickModel):
    def __init__(self, df, leadp=60):
        super(LeadModel, self).__init__(df)
        self.label = 'Lead (' + unicode(leadp) + '% weight on lead links)'

        # Lead links
        vals = pd.DataFrame(
            data=zip(df['target'], df['linkpos_ib_lead'].apply(len)),
            columns=['node', 'amount']
        )
        groupby = vals.groupby('node').sum()
        result = {k: v * leadp
                  for k, v in zip(groupby.index, groupby['amount'])}
        self.update_data(result)

        # Not-Lead links
        vals = pd.DataFrame(
            data=zip(df['target'], df['linkpos_not_ib_lead'].apply(len)),
            columns=['node', 'amount']
        )
        groupby = vals.groupby('node').sum()
        result = {k: v * (100-leadp)
                  for k, v in zip(groupby.index, groupby['amount'])}
        self.update_data(result)
        self.normalize()


def get_df_wikigame(kind):
    wg = Wikispeedia()
    df = wg.get_model_df(kind)
    return df


def get_df_wikipedia():
    fpath = os.path.join('clickstream',
                         'DataHandler_aggregate_clicks_smoothed.obj')
    df = pd.read_pickle(fpath).values()[0]
    return df


if __name__ == '__main__':
    # df = get_df_wikipedia()
    for kind in [
        'all',
        'successful',
        'unsuccessful'
    ]:
        print('\n\n' + kind)
        df = get_df_wikigame(kind)
        gt = GroundTruthModel(df)
        um = UniformModel(df)
        gt.compare(um)
        for leadp in [
            90,
            88,
            86,
            84,
            82,
            80,
            70,
            60,
            50,
            40,
        ]:
            lm = LeadModel(df, leadp)
            gt.compare(lm)

