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
        self.data = pd.read_pickle(os.path.join('data', self.label, 'data.obj'))
        print('loaded\n')
        self.plot_folder = os.path.join('data', 'plots')
        if not os.path.exists(self.plot_folder):
            os.makedirs(self.plot_folder)

    def plot(self):
        """configure and call the plotter"""
        # add subjects for tsplot
        self.data['subject'] = self.data['user'] + '_' + \
            self.data['mission'].astype('str')
        xlabel = 'Distance to-go to target'

        for feature, title, ylabel in [
            # ('spl_target', 'Shortest Path Length to Target', None),
            # ('tfidf_target', 'TF-IDF similarity to Target', None),
            # ('degree_out', 'Out-degree', None),
            # ('degree_in', 'In-degree', None),
            # ('ngram_anchor', 'N-Gram Frequency (Anchor)'),
            # ('ngram_body', 'N-Gram Frequency (Body)'),
            # ('ngram_query', 'N-Gram Frequency (Query)'),
            # ('ngram_title', 'N-Gram Frequency (Title)'),
            # ('category_depth', 'Category Depth (1...most general)'),
            # ('category_target', 'Category Distance to target'),
            # ('exploration', 'Explored Percentage of Page'),
            # ('linkpos_ib', 'Fraction of Links in Infobox'),
            # ('linkpos_lead', 'Fraction of Links in Lead'),
            ('time', 'Time per article', 'seconds'),
            # ('time_word', 'Time per article (per word)', 'seconds'),
            # ('time_link', 'Time per article (per link)', 'seconds'),
            # ('time_normalized', 'Time per article (normalized)', 'seconds')
        ]:
            print(feature)
            if feature not in self.data:
                print('    Feature not present')
                continue
            p = Plot(title)

            for k, m, c in zip([4, 5, 6, 7], markers, colors):
                df = self.data[(self.data.pl == k) & (self.data.spl == 3) &
                               self.data.successful]
                df = df[['distance-to-go', 'subject', 'pl', feature]]
                p.add_tsplot(df, time='distance-to-go', unit='subject',
                             condition='pl', value=feature, marker=m, color=c,
                             xlabel=xlabel)
            fname = feature + '_' + self.label.lower() + '.png'
            p.finish(os.path.join(self.plot_folder, fname))
        # drop subjects for tsplot
        self.data.drop('subject', axis=1, inplace=True)

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
    def __init__(self, x=1, y=1):
        """create the plot"""
        self.fig, self.ax = plt.subplots(x, y, figsize=(8, 5))

    def add_tsplot(self, data, time, unit, condition, value, **kwargs):
            self.ax.invert_xaxis()
            sns.tsplot(data, time=time, unit=unit, condition=condition,
                       value=value, estimator=np.nanmean, **kwargs)

    def finish(self, fname, **kwargs):
        """perform some beautification"""
        plt.legend(loc=0)
        offset = np.abs(0.05 * plt.xlim()[1])
        plt.xlim((plt.xlim()[0] - offset, plt.xlim()[1] + offset))
        offset = np.abs(0.05 * plt.ylim()[1])
        plt.ylim((plt.ylim()[0] - offset, plt.ylim()[1] + offset))
        self.ax.invert_xaxis()
        title = kwargs.pop('title', '')
        plt.title(title)
        plt.ylabel(kwargs.pop('ylabel', title))
        plt.xlabel(kwargs.pop('xlabel', ''))
        plt.savefig(fname)


if __name__ == '__main__':
    for pt in [
        Plotter('WIKTI'),
        # Plotter('Wikispeedia'),
    ]:
        pt.plot()
        # pt.plot_linkpos()
        # pt.correlation()

