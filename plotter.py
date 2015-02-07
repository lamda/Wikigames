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

# import warnings
# warnings.simplefilter("error")


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
        for feature, title, ylim in [
            # ('spl_target', 'Shortest Path Length to Target'),
            # ('tfidf_target', 'TF-IDF similarity to Target'),
            # ('degree_out', 'Out-degree'),
            # ('degree_in', 'In-degree'),
            # ('ngram', 'N-Gram Occurrence Frequency (=~ familiarity)'),
            # ('category_depth', 'Category Depth (1...most general)'),
            # ('category_target', 'Category Distance to target'),
            # ('exploration', 'Explored Percentage of Page'),
            # ('linkpos_intro', 'Fraction of Links in Introduction', (0, 1)),
            ('time', 'Time per article', (0, 11000)),
            ('time_word', 'Time per article (per word)', (0, 11000)),
            ('time_link', 'Time per article (per link)', (0, 11000)),
            ('time_normalized', 'Time per article (normalized)', (0, 11000)),
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
                data = [d for d in data if '' not in d]
                for d in data:
                    distance = range(k)
                    distance.reverse()
                    result.append(pd.DataFrame({
                        'condition': ['GL %d' % k] * len(d),
                        'subj': [str(subj)] * len(d),
                        'distance': distance,
                        'path': d,
                    }, dtype=np.float))
                    subj += 1
                result = pd.concat(result)
                p.add_tsplot(result, time='distance', unit='subj',
                             condition='condition', value='path',
                             marker=m, color=c)
            fname = feature + '_' + self.label.lower() + '.png'
            p.finish(os.path.join(self.plot_folder, fname), ylim=ylim)

    def plot_linkpos(self):
        print('linkpos')
        if self.label == 'WIKTI' and False:  # unnecessary to alway print this
            df = self.data[(self.data.spl == 3) & self.data.successful &
                           (self.data.pl < 9)]
            df = pd.concat([d for d in df['data']])
            df['linkpos_diff'] = df['linkpos_first'] - df['linkpos_last']
            df = df[~np.isnan(df['linkpos_diff'])]
            diff = df[df['linkpos_diff'] != 0]
            print('multiple link positions for %.2f of %d clicked links' %
                  (diff.shape[0] / df.shape[0], df.shape[0]))

            first = diff[diff['linkpos_first'] == diff['linkpos_actual']]
            first = first.shape[0]
            last = diff[diff['linkpos_last'] == diff['linkpos_actual']]
            last = last.shape[0]
            between = diff[(diff['linkpos_last'] != diff['linkpos_actual']) &\
                           (diff['linkpos_first'] != diff['linkpos_actual'])]
            entire = diff.shape[0]
            print('%.2f first, %.2f last, out of %d total' %
                  (first/entire, last/entire, entire))
            stats = between[['linkpos_first', 'linkpos_actual', 'linkpos_last']]
            first = stats['linkpos_actual'] - stats['linkpos_first'].tolist()
            last = stats['linkpos_last'] - stats['linkpos_actual'].tolist()
            ff, ll = 0, 0
            for f, l in zip(first, last):
                if f < l:
                    ff += 1
                else:
                    ll += 1
            total = ff + ll
            print(ff/total, ll/total, total)
            pdb.set_trace()

        p = Plot('word', 'Distance to Target')
        for k, c in zip([4, 5, 6], colors):
            for feature, label, m, ls in [
                ('linkpos_last', 'last occurrence', 'v', 'solid'),
                ('linkpos_actual', 'click position', 'o', 'dashed'),
                ('linkpos_first', 'first occurrence', '^', 'solid'),
                ('word_count', 'article length', '', 'dotted')
            ]:
                try:
                    self.data.iloc[0]['data'][feature]
                except KeyError, e:
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
                        'condition': ['GL %d (%s)' % (k, label)] * len(d),
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
        fname = 'linkpos_' + self.label.lower() + '.png'
        p.finish(os.path.join(self.plot_folder, fname), ylim=(0, 12000))


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
                       value=value, ci=ci, estimator=np.nanmean,
                       marker=marker, color=color, linestyle=linestyle)

    def finish(self, fname, ylim=None):
        """perform some beautification"""
        plt.legend(loc=0)
        offset = np.abs(0.05 * plt.xlim()[1])
        plt.xlim((plt.xlim()[0] - offset, plt.xlim()[1] + offset))
        if ylim:
            plt.ylim(ylim)
        else:
            offset = np.abs(0.05 * plt.ylim()[1])
            plt.ylim((plt.ylim()[0] - offset, plt.ylim()[1] + offset))
        self.ax.invert_xaxis()
        plt.title(self.title)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.title)
        plt.savefig(fname)


if __name__ == '__main__':
    for pt in [
        Plotter('WIKTI'),
        Plotter('Wikispeedia'),
    ]:
        pt.plot()
        pt.plot_linkpos()

