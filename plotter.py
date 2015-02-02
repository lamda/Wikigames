# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import pdb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# set a few options
pd.options.mode.chained_assignment = None
colors = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
markers = ['o', 'h', 'd', 'v', 's', 'x']
sns.set_palette(sns.color_palette(colors))


class Plotter(object):
    def __init__(self, label):
        print(label)
        self.label = label
        print('loading data...')
        self.data = pd.read_pickle(os.path.join('data', self.label, 'data.pd'))
        print('loaded\n')
        self.plot_folder = os.path.join('data', self.label, 'plots')
        if not os.path.exists(self.plot_folder):
            os.makedirs(self.plot_folder)

    def plot(self):
        for feature, title in [
            ('spl_target', 'Shortest Path Length to Target'),
            ('tfidf_target', 'TF-IDF similarity to Target'),
            ('degree_out', 'Out-degree'),
            ('degree_in', 'In-degree'),
            ('ngram', 'N-Gram Occurrence Frequency (=~ familiarity)'),
            ('category_depth', 'Category Depth (1...most general)'),
            ('category_target', 'Category Distance to target'),
            ('exploration', 'Explored Percentage of Page'),
        ]:
            print(feature)
            try:
                self.data.iloc[0]['data'][feature]
            except KeyError, e:
                print('    Feature not present')
                continue
            p = Plot(title, 'Distance to Target')

            for k, m, c in zip([4, 5, 6, 7], markers, colors):
                subj = 0
                result = []
                df = self.data[(self.data.pl == k) & (self.data.spl == 3) &
                               self.data.successful]
                data = [d[feature].tolist() for d in df['data']]
                data = [d for d in data if '' not in d and not np.isnan(sum(d))]
                for d in data:
                    distance = range(k)
                    distance.reverse()
                    result.append(pd.DataFrame({
                        'condition': ['PL %d' % k] * len(d),
                        'subj': [str(subj)] * len(d),
                        'distance': distance,
                        'path': d,
                    }, dtype=np.float))
                    subj += 1
                result = pd.concat(result)
                p.add_tsplot(result, time='distance', unit='subj',
                             condition='condition', value='path',
                             marker=m, color=c)
            p.finish(os.path.join(self.plot_folder, feature + '.png'))

    def plot_linkpos(self):
        print('linkpos')
        p = Plot('Link Position', 'Distance to Target')
        for k, c in zip([4, 5, 6, 7], colors):
        # for k, c in zip([5], colors):
            for feature, label, m, ls in [
                ('linkpos_last', 'last', 'v', 'solid'),
                ('linkpos_actual', 'actual', 'o', 'dashed'),
                ('linkpos_first', 'first', '^', 'solid'),
            ]:
                try:
                    self.data.iloc[0]['data'][feature]
                except KeyError, e:
                    print('    Feature not present')
                    continue
                subj = 0
                result = []
                df_raw = self.data[(self.data.pl == k) & (self.data.spl == 3) &
                                   self.data.successful]
                data = [d[feature].tolist() for d in df_raw['data']]
                data = [d for d in data if '' not in d]
                for d in data:
                    distance = range(k)
                    distance.reverse()
                    df = pd.DataFrame({
                        'condition': ['PL %d (%s)' % (k, label)] * len(d),
                        'subj': [str(subj)] * len(d),
                        'distance': distance,
                        'path': d,
                    }, dtype=np.float)
                    df = df[~np.isnan(df['path'])]
                    result.append(df)
                    subj += 1
                result = pd.concat(result)
                p.add_tsplot(result, time='distance', unit='subj',
                             condition='condition', value='path',
                             marker=m, color=c, linestyle=ls, ci=0)
        p.finish(os.path.join(self.plot_folder, 'linkpos.png'))


def estimator(data, **kwargs):
    """compute the mean of a list and skip np.NaN values"""
    if kwargs['axis'] != 0:
        print('not implemented yet')
        pdb.set_trace()
    data = [np.mean([x for x in col if not np.isnan(x)]) for col in data.T]
    return data


class Plot(object):
    def __init__(self, title, xlabel):
        """create the plot"""
        self.fig, self.ax = plt.subplots(1, figsize=(8, 5))
        self.title = title
        self.xlabel = xlabel

    def add_tsplot(self, data, time, unit, condition, value,
                   marker='o', color='black', linestyle='solid', ci=68):
            # TODO 68 is the standard error?
            self.ax.invert_xaxis()
            sns.tsplot(data, time=time, unit=unit, condition=condition,
                       value=value, ci=ci, estimator=estimator,
                       marker=marker, color=color, linestyle=linestyle)

    def finish(self, fname):
        """perform some beautification"""
        plt.legend(loc=0)
        offset = np.abs(0.05 * plt.xlim()[1])
        plt.xlim((plt.xlim()[0] - offset, plt.xlim()[1] + offset))
        offset = np.abs(0.05 * plt.ylim()[1])
        plt.ylim((plt.ylim()[0] - offset, plt.ylim()[1] + offset))
        self.ax.invert_xaxis()
        plt.title(self.title)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.title)
        plt.savefig(fname)


if __name__ == '__main__':
    pt = Plotter('WIKTI')
    # pt = Plotter('Wikispeedia')
    # pt.plot()
    pt.plot_linkpos()

