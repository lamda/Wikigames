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
    def __init__(self, labels):
        self.data = {}
        self.labels = labels
        for label in self.labels:
            print('loading', label, 'data...')
            path = os.path.join('data', label, 'data.obj')
            self.data[label] = pd.read_pickle(path)
            print('loaded\n')
        self.plot_folder = os.path.join('data', 'plots')
        if not os.path.exists(self.plot_folder):
            os.makedirs(self.plot_folder)

    def plot_all(self):
        """configure and call the plotter"""
        xlabel = 'Distance to-go to target'

        for feature, title in [
            ('spl_target', 'Shortest Path Length to Target'),
            ('tfidf_target', 'TF-IDF similarity to Target'),
            ('degree_out', 'Out-degree'),
            ('degree_in', 'In-degree'),
            # ('ngram_anchor', 'N-Gram Frequency (Anchor)'),
            # ('ngram_body', 'N-Gram Frequency (Body)'),
            ('ngram_query', 'N-Gram Frequency (Query)'),
            # ('ngram_title', 'N-Gram Frequency (Title)'),
            ('category_depth', 'Category Depth (1...most general)'),
            ('category_target', 'Category Distance to target'),
            # ('exploration', 'Explored Percentage of Page'),
            # ('linkpos_ib', 'Fraction of Links in Infobox'),
            # ('linkpos_lead', 'Fraction of Links in Lead'),
            # ('time', 'Time per article', 'seconds'),
            # ('time_word', 'Time per article (per word)', 'seconds'),
            # ('time_link', 'Time per article (per link)', 'seconds'),
            # ('time_normalized', 'Time per article (normalized)', 'seconds')
        ]:
            print(feature)
            p = Plot(nrows=1, ncols=2)
            for dataset_name, dataset in self.data.items():
                x = self.labels.index(dataset_name)
                for k, m, c in zip([4, 5, 6, 7], markers, colors):
                    df = dataset[(dataset['pl'] == k) & (dataset['spl'] == 3) &
                                 dataset['successful']]
                    df = df[['distance-to-go', 'subject', 'pl', feature]]
                    df.rename(columns={'pl': 'Game length'}, inplace=True)
                    p.add_tsplot(df, col=x, time='distance-to-go',
                                 unit='subject', condition='Game length',
                                 value=feature, marker=m, color=c)
            fname = feature + '_' + feature.lower() + '.png'
            titles = np.array([self.labels])
            p.finish(os.path.join(self.plot_folder, fname), suptitle=title,
                     titles=titles, xlabel=xlabel)

    def plot_linkpos(self):
        print('linkpos')
        if self.label == 'WIKTI' and False:  # only print this when needed
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
        p.finish(os.path.join(self.plot_folder, fname), ylim=(12000, 0))

    def correlation(self):
        for feature1, feature2 in [
            ('degree_in', 'category_depth'),
        ]:
            print(feature1, feature2)
            try:
                self.data.iloc[0]['data'][feature1]
                self.data.iloc[0]['data'][feature2]
            except KeyError, e:
                print('    Feature not present')
                continue
            df = self.data[(self.data.pl < 9 ) & (self.data.spl == 3) &
                           self.data.successful]
            data1 = [d[feature1].tolist() for d in df['data']]
            data2 = [d[feature2].tolist() for d in df['data']]
            data1 = [d for d in data1 if '' not in d]
            data2 = [d for d in data2 if '' not in d]
            sns.corrplot()


class Plot(object):
    def __init__(self, nrows=1, ncols=1):
        """create the plot"""
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=(14, 5),
                                           squeeze=False)

    def add_tsplot(self, data, time, unit, condition, value, **kwargs):
            row = kwargs.pop('row', 0)
            col = kwargs.pop('col', 0)
            ax = self.axes[row, col]
            if not ax.xaxis_inverted():
                ax.invert_xaxis()
            sns.tsplot(data, ax=ax, time=time, unit=unit, condition=condition,
                       value=value, estimator=np.nanmean, **kwargs)

    def match_ylim(self):
        for row in range(self.axes.shape[0]):
            ylim_lower = min(a.get_ylim()[0] for a in self.axes[row])
            ylim_upper = max(a.get_ylim()[1] for a in self.axes[row])
            for col in range(self.axes.shape[1]):
                ax = self.axes[row, col]
                ax.set_ylim(ylim_lower, ylim_upper)

    def add_margin(self, margin=0.05):
        for row in range(self.axes.shape[0]):
            for col in range(self.axes.shape[1]):
                ax = self.axes[row, col]
                ylim = ax.get_ylim()
                length = ylim[1] - ylim[0]
                ax.set_ylim(ylim[0] - np.abs(0.05 * length),
                            ylim[1] + np.abs(0.05 * length))
                xlim = ax.get_xlim()
                length = xlim[1] - xlim[0]
                ax.set_xlim(xlim[0] - np.abs(0.05 * length),
                            xlim[1] + np.abs(0.05 * length))

    def finish(self, fname, **kwargs):
        """perform some beautification"""
        titles = kwargs.pop('titles', '')
        xlabel = kwargs.pop('xlabel', '')
        ylabel = kwargs.pop('ylabel', '')
        self.match_ylim()
        self.add_margin()
        for row in range(self.axes.shape[0]):
            for col in range(self.axes.shape[1]):
                ax = self.axes[row, col]
                if not ax.xaxis_inverted():
                    ax.invert_xaxis()
                try:
                    ax.set_title(titles[row, col])
                except (IndexError, TypeError):
                    ax.set_title('')
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)

        plt.suptitle(kwargs.pop('suptitle', ''), size='xx-large')
        self.fig.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9,
                                 wspace=0.15, hspace=0.15)
        plt.savefig(fname)


if __name__ == '__main__':
    for pt in [
        Plotter(['WIKTI', 'Wikispeedia']),
        # Plotter(['WIKTI', 'WIKTI2']),
        # Plotter('Wikispeedia'),
    ]:
        pt.plot_all()
        # pt.plot_linkpos()
        # pt.correlation()

