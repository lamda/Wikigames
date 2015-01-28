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
sns.set_palette(sns.color_palette(["#9b59b6", "#3498db", "#95a5a6",
                                   "#e74c3c", "#34495e", "#2ecc71"]))


class Plotter(object):
    def __init__(self, label):
        self.label = label
        print('loading data...')
        self.data = pd.read_pickle(os.path.join('data', self.label, 'data.pd'))
        print('loaded')
        self.plot_folder = os.path.join('data', self.label, 'plots')
        if not os.path.exists(self.plot_folder):
            os.makedirs(self.plot_folder)

    def plot(self):
        markers = ['o', 'h', 'd', 'v']
        for feature, title in [
            ('spl_target', 'Shortest Path Length to Target'),
            ('tfidf_target', 'TF-IDF similarity to Target'),
            ('degree_out', 'Out-degree'),
            ('degree_in', 'In-degree'),
            ('pagerank', 'PageRank'),
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
            p = Plot(title, title, 'Distance to Target')

            result = []
            subj = 0
            for k in [4, 5, 6, 7]:
                df = self.data[(self.data.pl == k) & (self.data.spl == 3) &
                               self.data.successful]
                data = [d[feature].tolist() for d in df['data']]
                data = [d for d in data if '' not in d and np.NaN not in d]
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
                         condition='condition', value='path', markers=markers)
            p.finish(os.path.join(self.plot_folder, feature + '.png'))


class Plot(object):
    def __init__(self, title, xlabel, ylabel):
        """create the plot"""
        self.fig, self.ax = plt.subplots(1, figsize=(8, 5))
        plt.title(title)
        plt.xlabel('distance to-go to target')
        plt.ylabel(xlabel)

    def add_tsplot(self, data, time, unit, condition, value, markers, ci=68):
            self.ax.invert_xaxis()
            sns.tsplot(data, time=time, unit=unit, condition=condition,
                       value=value, ci=68)  # TODO 68 is the standard error?
            for m, a in zip(markers, reversed(self.ax.lines)):
                a.set_marker(m)

    def finish(self, fname):
        """perform some beautification"""
        plt.legend()
        offset = np.abs(0.05 * plt.xlim()[1])
        plt.xlim((plt.xlim()[0] - offset, plt.xlim()[1] + offset))
        offset = np.abs(0.05 * plt.ylim()[1])
        plt.ylim((plt.ylim()[0] - offset, plt.ylim()[1] + offset))
        self.ax.invert_xaxis()
        plt.savefig(fname)


if __name__ == '__main__':
    pt = Plotter('WIKTI')
    pt.plot()

    # pt = Plotter('Wikispeedia')
    # pt.plot()
