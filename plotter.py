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
        for feature, title in [
            # ('spl_target', 'Shortest Path Length to target'),
            # ('tfidf_target', 'TF-IDF similarity to target'),
            # ('degree_out', 'Out-degree'),
            # ('degree_in', 'In-degree'),
            # ('pagerank', 'PageRank'),
            # ('ngram', 'N-Gram occurrence frequency (=~ familiarity)'),
            ('category_depth', 'Category depth (1...most general)'),
            # ('category_target', 'Category distance to target'),
            # ('exploration', 'Explored percentage of page'),
        ]:
            print(feature)
            try:
                self.data.iloc[0]['data'][feature]
            except KeyError, e:
                print('Feature not present')
                continue

            fig, ax = plt.subplots(1, figsize=(10, 5))
            for k, m in zip([4, 5, 6, 7], ['o', 'h', 'd', 'v']):
                df = self.data[(self.data.pl == k) &
                               (self.data.spl == 3) &
                               self.data.successful]
                data = [[d[feature].iloc[i] for d in df['data']]
                        for i in range(k)]
                data = [np.mean([e for e in d if e != '']) for d in data]
                print(k, data)
                data.reverse()
                plt.plot(data, label=str(k), marker=m)

            print()
            plt.legend()

            # Beautification
            plt.title(title)
            plt.xlabel('distance to-go to target')
            plt.ylabel(feature)
            offset = 0.1 * plt.xlim()[1]
            plt.xlim((plt.xlim()[0] - offset, plt.xlim()[1] + offset))
            offset = 0.1 * plt.ylim()[1]
            plt.ylim((plt.ylim()[0] - offset, plt.ylim()[1] + offset))
            plt.gca().invert_xaxis()
            fname = os.path.join(self.plot_folder, feature + '.png')
            plt.savefig(fname)


if __name__ == '__main__':
    # p = Plotter('WIKTI')
    # p.plot()

    p = Plotter('Wikispeedia')
    p.plot()
