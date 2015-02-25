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
pd.set_option('display.width', 1000)
colors = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
sns.set_palette(sns.color_palette(colors))
sns.set_style("white")
# sns.set_style("ticks")
sns.set_context("notebook", font_scale=1.125, rc={"lines.linewidth": 1.5})

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
            print('loading', label, 'data...', end='\r')
            path = os.path.join('data', label, 'data.obj')
            self.data[label] = pd.read_pickle(path)
            # filter to include only games with shortest possible solutions of 3
            self.data[label] = self.data[label][(self.data[label]['spl'] == 3)]
            print(label, 'data loaded\n')
        self.plot_folder = os.path.join('plots')
        if not os.path.exists(self.plot_folder):
            os.makedirs(self.plot_folder)

    def plot_linkpos(self, data=None, labels=None, fname_suffix=''):
        fontsize_old = plt.rcParams['legend.fontsize']
        plt.rcParams['legend.fontsize'] = 7.5
        if data is None:
            data = self.data
        if labels is None:
            labels = self.labels
        print('linkpos()')
        xlabel = 'Distance to-go to target'
        titles = np.array([labels])
        p = Plot(1, len(data), rowsize=6, colsize=6)
        for k, c in zip([4, 5, 6], self.colors):
            for feature, ylabel, m, ls in [
                ('linkpos_first', 'first occurrence', 'v', 'solid'),
                ('linkpos_actual', 'click position', 'o', 'dashed'),
                ('linkpos_last', 'last occurrence', '^', 'solid'),
                ('word_count', 'article length', '', 'dotted')
            ]:
                for label, dataset in data.items():
                    if feature not in dataset:
                        print(feature, 'not present')
                        continue
                    x = labels.index(label)
                    df = dataset[(dataset['pl'] == k) & dataset['successful']]
                    df = df[['distance-to-go', 'subject', 'pl', feature]]
                    df['pl'] = df['pl'].apply(lambda l: str(l) +
                                              ' (' + ylabel + ')')
                    df.rename(columns={'pl': 'Game length'}, inplace=True)
                    p.add_tsplot(df, col=x, time='distance-to-go',
                                 unit='subject', condition='Game length', ci=0,
                                 value=feature, marker=m, color=c, linestyle=ls,
                                 legend=True)
        path = os.path.join(self.plot_folder, 'linkpos'+fname_suffix+'.png')
        p.finish(path, suptitle='Link Position', titles=titles, xlabel=xlabel,
                 ylabel='word', invert_xaxis=True, invert_yaxis=True)
        plt.rcParams['legend.fontsize'] = fontsize_old

    def plot_comparison(self, data=None, labels=None, fname_suffix=''):
        """draw comparison plots for multiple datasets"""
        if data is None:
            data = self.data
        if labels is None:
            labels = self.labels
        xlabel = 'Distance to-go to target'
        titles = np.array([labels])
        for feature, title, ylabel in [
            ('spl_target', 'Shortest Path Length to Target', ''),
            ('tfidf_target', 'TF-IDF similarity to Target', ''),
            ('degree_out', 'Out-degree', ''),
            ('degree_in', 'In-degree', ''),
            ('ngram', 'N-Gram Frequency (Query)', ''),
            ('category_depth', 'Category Depth', ''),
            ('category_target', 'Category Distance to target', ''),
            ('linkpos_ib', 'Fraction of clicked Links in Infobox', 'Fraction of links'),
            ('linkpos_lead', 'Fraction of clicked Links in Lead', 'Fraction of links'),
            ('link_context', 'Number of Links +/- 10 words from clicked link', 'Number of links')
        ]:
            print(feature)
            p = Plot(nrows=1, ncols=len(data))
            for label, dataset in data.items():
                x = labels.index(label)
                for k, m, c in zip([4, 5, 6, 7], self.markers, self.colors):
                    # filter the dataset
                    df = dataset[(dataset['pl'] == k) & dataset['successful']]
                    df = df[['distance-to-go', 'subject', 'pl', feature]]
                    df.rename(columns={'pl': 'Game length'}, inplace=True)
                    p.add_tsplot(df, col=x, time='distance-to-go',
                                 unit='subject', condition='Game length',
                                 value=feature, marker=m, color=c)
            path = os.path.join(self.plot_folder, feature+fname_suffix+'.png')
            p.finish(path, suptitle=title, titles=titles, xlabel=xlabel,
                     ylabel=ylabel, invert_xaxis=True)

    def plot_wikti(self):
        """draw plots for features within the WIKTI dataset"""
        xlabel = 'Distance to-go to target'
        dataset = self.data['WIKTI']
        for features, titles, suptitle, ylabel in [
            # [
            #     ['linkpos_ib', 'linkpos_lead'],
            #     ['Fraction of Links in Infobox', 'Fraction of Links in Lead'],
            #     'Fractions of links in Infobox and Lead',
            #     'Fraction of links'
            # ],
            # [
            #     ['exploration'],
            #     ['Explored Percentage of Page'],
            #     ['']
            # ],
            [
                ['time'],
                ['Time per article'],
                '',
                'seconds'
            ],
            [
                ['time_word'],
                ['Time per word'],
                '',
                'seconds'
            ],
            [
                ['time_link'],
                ['Time per link'],
                '',
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

    def print_click_stats(self):
        print('Statistics for WIKTI')
        dataset = self.data['WIKTI']
        df = dataset[dataset['successful'] & (dataset['pl'] < 9)]
        df = df[['linkpos_first', 'linkpos_last', 'linkpos_actual']]
        df['linkpos_diff'] = df['linkpos_first'] - df['linkpos_last']
        df = df[~np.isnan(df['linkpos_diff'])]
        diff = df[df['linkpos_diff'] != 0]
        print('multiple link positions for %.2f%% of %d clicked links' %
              (100 * diff.shape[0] / df.shape[0], df.shape[0]))

        first = diff[diff['linkpos_first'] == diff['linkpos_actual']]
        first = first.shape[0]
        last = diff[diff['linkpos_last'] == diff['linkpos_actual']]
        last = last.shape[0]
        between = diff[(diff['linkpos_last'] != diff['linkpos_actual']) &
                       (diff['linkpos_first'] != diff['linkpos_actual'])]
        entire = diff.shape[0]
        print('of those with multiple positions, %.2f%% first, %.2f%% last, %.2f%% inbetween out of %d total' %
              (100 * first/entire, 100 * last/entire, 100 - 100 * (first + last) / entire, entire))
        stats = between[['linkpos_first', 'linkpos_actual', 'linkpos_last']]
        first = stats['linkpos_actual'] - stats['linkpos_first'].tolist()
        last = stats['linkpos_last'] - stats['linkpos_actual'].tolist()
        ff, ll = 0, 0
        for f, l in zip(first, last):
            if f <= l:
                ff += 1
            else:
                ll += 1
        total = ff + ll
        print('of those inbetween, %.2f%% closer to first, %.2f%% closer to last out of %d total' %
              (100 * ff/total, 100 * ll/total, total))

    def print_game_stats(self):
        for label, dataset in self.data.items():
            df = dataset[(dataset['spl'] == 3) & (dataset['successful']) &
                         (dataset['distance-to-go'] == 0)]
            df['mission'] = df['start'] + '-' + df['target']
            print(df['mission'].value_counts(), df.shape)

    def plot_games_users(self):
        print('plot_games_users()')
        df = self.data['Wikispeedia']
        data = {
            'novice users': df[df['above_pl_user_mean']],
            'proficient users': df[~df['above_pl_user_mean']],
            'easy games': df[df['above_pl_mission_mean']],
            'hard games': df[~df['above_pl_mission_mean']],
            }
        labels = ['novice users', 'proficient users',
                  'easy games', 'hard games']
        for label, dataset in data.items():
            print(label, dataset.shape[0])
        self.plot_linkpos(data, labels, fname_suffix='_split')
        self.plot_comparison(data, labels, fname_suffix='_split')

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
    def __init__(self, nrows=1, ncols=1, rowsize=4.25, colsize=4.5):
        """create the plot"""
        if ncols == 1:
            rowsize += 0.5
        self.fig, self.axes = plt.subplots(nrows, ncols, squeeze=False,
                                           figsize=(0.5 + rowsize * ncols,
                                                    colsize))

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
        invert_yaxis = kwargs.pop('invert_yaxis', False)
        self.match_ylim()
        self.add_margin()
        for row in range(self.axes.shape[0]):
            for col in range(self.axes.shape[1]):
                ax = self.axes[row, col]
                if invert_xaxis:
                    ax.invert_xaxis()
                if invert_yaxis:
                    ax.invert_yaxis()
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                try:
                    ax.set_title(titles[row, col])
                except (IndexError, TypeError):
                    ax.set_title('')
        plt.suptitle(suptitle, size='xx-large')
        sns.despine(fig=self.fig)
        self.fig.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.85,
                                 wspace=0.3, hspace=0.2)
        if self.axes.shape[1] == 1:
            self.fig.subplots_adjust(left=0.15)
        # plt.show()
        plt.savefig(fname)


if __name__ == '__main__':
    for pt in [
        # Plotter(['Wikispeedia']),
        # Plotter(['WIKTI']),
        Plotter(['WIKTI', 'Wikispeedia']),
        # Plotter(['WIKTI', 'WIKTI2']),
        # Plotter(['WIKTI', 'WIKTI2', 'WIKTI3']),
    ]:
        pt.plot_linkpos()
        pt.plot_comparison()
        pt.plot_wikti()
        # pt.print_game_stats()
        # pt.plot_games_users()
        # pt.correlation()

