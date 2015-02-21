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
sns.set_palette(sns.color_palette(colors))

# import warnings
# warnings.simplefilter("error")


class Plotter(object):
    def __init__(self, labels):
        self.markers = ['o', 'h', 'd', 'v', 's', 'x']
        self.colors = colors
        self.data = {}
        self.labels = labels
        for label in self.labels:
            # load datasets
            print('loading', label, 'data...')
            path = os.path.join('data', label, 'data.obj')
            self.data[label] = pd.read_pickle(path)
            # filter to include only games with shortest possible solutions of 3
            self.data[label] = self.data[label][(self.data[label]['spl'] == 3)]
            print('loaded\n')
        self.plot_folder = os.path.join('plots')
        if not os.path.exists(self.plot_folder):
            os.makedirs(self.plot_folder)

    def plot_comparison(self):
        """draw comparison plots for multiple datasets"""
        xlabel = 'Distance to-go to target'
        titles = np.array([self.labels])
        for feature, title, ylabel in [
            ('spl_target', 'Shortest Path Length to Target', ''),
            ('tfidf_target', 'TF-IDF similarity to Target', ''),
            ('degree_out', 'Out-degree', ''),
            ('degree_in', 'In-degree', ''),
            ('ngram', 'N-Gram Frequency (Query)', ''),
            ('category_depth', 'Category Depth', ''),
            ('category_target', 'Category Distance to target', ''),
            ('linkpos_ib', 'Fraction of Links in Infobox', 'Fraction of links'),
            ('linkpos_lead', 'Fraction of Links in Lead', 'Fraction of links'),
            ('link_context', 'Number of Links +/- 10 words from clicked link', 'Number of links')
        ]:
            print(feature)
            p = Plot(nrows=1, ncols=len(self.data))
            for label, dataset in self.data.items():
                x = self.labels.index(label)
                for k, m, c in zip([4, 5, 6, 7], self.markers, self.colors):
                    # filter the dataset
                    df = dataset[(dataset['pl'] == k) & dataset['successful']]
                    df = df[['distance-to-go', 'subject', 'pl', feature]]
                    df.rename(columns={'pl': 'Game length'}, inplace=True)
                    p.add_tsplot(df, col=x, time='distance-to-go',
                                 unit='subject', condition='Game length',
                                 value=feature, marker=m, color=c)
            p.finish(os.path.join(self.plot_folder, feature + '.png'),
                     suptitle=title, titles=titles,
                     xlabel=xlabel, ylabel=ylabel, invert_xaxis=True)

    def plot_wikti(self):
        """draw plots for features within the WIKTI dataset"""
        xlabel = 'Distance to-go to target'
        dataset = self.data['WIKTI']
        for features, titles, suptitle, ylabel in [
            [
                ['linkpos_ib', 'linkpos_lead'],
                ['Fraction of Links in Infobox', 'Fraction of Links in Lead'],
                'Fractions of links in Infobox and Lead',
                'Fraction of links'
            ],
            # [
            #     ['exploration'],
            #     ['Explored Percentage of Page'],
            #     ['']
            # ],
            [
                ['time', 'time_word', 'time_link'],
                ['Time per article', 'Time per word', 'Time per link'],
                'Time spent',
                'seconds'
            ]
        ]:
            p = Plot(nrows=1, ncols=len(features))
            for idx, feature in enumerate(features):
                for k, m, c in zip([4, 5, 6, 7], self.markers, self.colors):
                    df = dataset[(dataset['pl'] == k) & (dataset['spl'] == 3) &
                                 dataset['successful']]
                    df = df[['distance-to-go', 'subject', 'pl', feature]]
                    df.rename(columns={'pl': 'Game length'}, inplace=True)
                    p.add_tsplot(df, col=idx, time='distance-to-go',
                                 unit='subject', condition='Game length',
                                 value=feature, marker=m, color=c)
            titles = np.array([titles])
            path = os.path.join(self.plot_folder, '_'.join(features) + '.png')
            p.finish(path, titles=titles, xlabel=xlabel, ylabel=ylabel,
                     invert_xaxis=True, suptitle=suptitle)

    def print_stats(self):
        print('Statistics for WIKTI')
        dataset = self.data['WIKTI']
        df = dataset[dataset['successful'] & (dataset['pl'] < 9)]
        df = df[['linkpos_first', 'linkpos_last', 'linkpos_actual']]
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

    def plot_linkpos(self):
        print('linkpos')
        xlabel = 'Distance to-go to target'
        titles = np.array([self.labels])
        p = Plot(1, len(self.data))
        for k, c in zip([4, 5, 6], self.colors):
            for feature, ylabel, m, ls in [
                ('linkpos_last', 'last occurrence', 'v', 'solid'),
                ('linkpos_actual', 'click position', 'o', 'dashed'),
                ('linkpos_first', 'first occurrence', '^', 'solid'),
                ('word_count', 'article length', '', 'dotted')
            ]:
                for label, dataset in self.data.items():
                    if feature not in dataset:
                        print(feature, 'not present')
                        continue
                    x = self.labels.index(label)
                    df = dataset[(dataset['pl'] == k) & dataset['successful']]
                    df = df[['distance-to-go', 'subject', 'pl', feature]]
                    df['pl'] = df['pl'].apply(lambda l: str(l) +
                                              ' (' + ylabel + ')')
                    df.rename(columns={'pl': 'Game length'}, inplace=True)
                    p.add_tsplot(df, col=x, time='distance-to-go',
                                 unit='subject', condition='Game length', ci=0,
                                 value=feature, marker=m, color=c, linestyle=ls,
                                 legend=True)
        p.finish(os.path.join(self.plot_folder, 'linkpos.png'),
                 suptitle='Link Position', titles=titles, xlabel=xlabel,
                 ylabel='word', invert_xaxis=True)

    def correlation(self):
        for features in [
            ['degree_in', 'degree_out'],
            ['degree_in', 'category_depth'],
            ['degree_in', 'ngram'],
            ['category_depth', 'ngram'],
        ]:
            for label, dataset in self.data.items():
                df = dataset[(dataset['pl'] < 9) & dataset['successful']]
                df = df[features]
                sns.jointplot(features[0], features[1], df)
                fname = 'corr_' + features[0] + '_' + features[1] + '_'\
                        + label + '.png'
                plt.title(label)
                plt.savefig(os.path.join(self.plot_folder, fname))


class Plot(object):
    def __init__(self, nrows=1, ncols=1):
        """create the plot"""
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=(2+6*ncols, 6),
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
        suptitle = kwargs.pop('suptitle', '')
        xlabel = kwargs.pop('xlabel', '')
        ylabel = kwargs.pop('ylabel', suptitle)
        invert_xaxis = kwargs.pop('invert_xaxis', False)
        self.match_ylim()
        self.add_margin()
        for row in range(self.axes.shape[0]):
            for col in range(self.axes.shape[1]):
                ax = self.axes[row, col]
                if invert_xaxis:
                    ax.invert_xaxis()
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                try:
                    ax.set_title(titles[row, col])
                except (IndexError, TypeError):
                    ax.set_title('')
        plt.suptitle(suptitle, size='xx-large')
        self.fig.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.9,
                                 wspace=0.15, hspace=0.15)
        plt.savefig(fname)


if __name__ == '__main__':
    for pt in [
        Plotter(['WIKTI', 'Wikispeedia']),
        # Plotter(['WIKTI']),
        # Plotter(['WIKTI', 'WIKTI2']),
        # Plotter(['WIKTI', 'WIKTI2', 'WIKTI3']),
        # Plotter(['Wikispeedia']),
    ]:
        pt.plot_comparison()
        # pt.plot_wikti()
        # pt.print_stats()
        # pt.plot_linkpos()
        # pt.correlation()

