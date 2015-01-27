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
            # ('exploration', 'Explored Percentage of Page'),
        ]:
            print(feature)
            try:
                self.data.iloc[0]['data'][feature]
            except KeyError, e:
                print('Feature not present')
                continue

            fig, ax = plt.subplots(1, figsize=(8, 5))
            use_tsplot = True
            result = []
            subj = 0
            for k, m in zip([4, 5, 6, 7], markers):
                df = self.data[(self.data.pl == k) &
                               (self.data.spl == 3) &
                               self.data.successful]
                data_raw = [[d[feature].iloc[i] for d in df['data']]
                            for i in range(k)]
                if use_tsplot:
                    data = [[d[k1] for d in data_raw]
                            for k1 in range(len(data_raw[0]))]
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
                else:
                    data = [[e for e in d if e != '' and not np.isnan(e)]
                            for d in data_raw]
                    data = [np.mean(d) for d in data]
                    print(k, data)
                    data.reverse()
                    plt.plot(data, label=str(k), marker=m)

            if use_tsplot:
                result = pd.concat(result)
                ax = plt.gca()
                ax.invert_xaxis()
                sns.tsplot(result, time='distance', unit='subj',
                           condition='condition', value='path',
                           marker='o', ci=68)  # TODO 68 is the standard error?

            print()

            # Beautification
            # for i, m in enumerate(markers):
            #     ax.lines[-i].set_marker(m)
            for m, a in zip(markers, reversed(ax.lines)):
                a.set_marker(m)
            plt.legend()
            plt.title(title)
            plt.xlabel('distance to-go to target')
            plt.ylabel(feature)
            offset = np.abs(0.05 * plt.xlim()[1])
            plt.xlim((plt.xlim()[0] - offset, plt.xlim()[1] + offset))
            offset = np.abs(0.05 * plt.ylim()[1])
            plt.ylim((plt.ylim()[0] - offset, plt.ylim()[1] + offset))
            plt.gca().invert_xaxis()
            alt = '' if use_tsplot else '_alt'
            fname = os.path.join(self.plot_folder, feature + alt + '.png')
            plt.savefig(fname)


if __name__ == '__main__':
    p = Plotter('WIKTI')
    p.plot()

    # p = Plotter('Wikispeedia')
    # p.plot()
