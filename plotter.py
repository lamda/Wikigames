# -*- coding: utf-8 -*-

from __future__ import division, print_function

import operator
import os
import pdb

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns

# set a few options
pd.options.mode.chained_assignment = None
pd.set_option('display.width', 1000)
# colors = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71",
colors = [ "#2ecc71","#3498db", "#e74c3c", "#9b59b6",  "#34495e", "#95a5a6",
          "#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]
sns.set_palette(sns.color_palette(colors))

sns.set_style("white")
# sns.set_style("ticks")
sns.set_context("notebook", font_scale=1.125, rc={"lines.linewidth": 1.5})


class Plotter(object):
    def __init__(self, labels, spl=3):
        self.markers = ['o', 's', '^', 'd', 'v', 'h']
        self.colors = colors
        self.data = {}
        self.labels = labels
        for label in self.labels:
            # load dataset
            print('loading', label, 'data...', end='\r')
            path = os.path.join('data', label, 'data.obj')
            self.data[label] = pd.read_pickle(path)

            # filter dataset
            self.data[label] = self.data[label][
                (self.data[label]['spl'] == spl) &
                (self.data[label]['successful']) &
                (self.data[label]['pl'] < 9)
            ]
            print(label, 'data loaded\n')

        self.plot_folder = os.path.join('plots')
        if not os.path.exists(self.plot_folder):
            os.makedirs(self.plot_folder)

    def plot_linkpos_fill_between(self, data=None, labels=None, fname_suffix='',
                                  full=True):
        fontsize_old = plt.rcParams['legend.fontsize']
        plt.rcParams['legend.fontsize'] = 7.5
        if data is None:
            data = self.data
        if labels is None:
            labels = self.labels
        print('linkpos()')
        xlabel = 'Distance to-go to target'
        titles = np.array([labels])
        p = Plot(labels, len(data))
        for k, c, m in zip([4, 5, 6], self.colors, self.markers):
            for label, dataset in data.items():
                col = labels.index(label)
                df = dataset[dataset['pl'] == k]
                df = df.dropna()
                x = sorted(df['distance-to-go'].unique().tolist())
                first = [df[df['distance-to-go'] == dtg]['linkpos_first'].mean()
                         for dtg in range(1, k)]
                last = [df[df['distance-to-go'] == dtg]['linkpos_last'].mean()
                        for dtg in range(1, k)]
                length = [df[df['distance-to-go'] == dtg]['word_count'].mean()
                          for dtg in range(1, k)]

                # normalization
                first = [e/l for e, l in zip(first, length)]
                last = [e/l for e, l in zip(last, length)]
                # pdb.set_trace()
                if full:
                    p.add_fill_between(x, first, last, color=c, col=col, gl=k,
                                       label='link occurrence')
                    p.add_plot(x, first, color=c, col=col, lw=0.5)
                    p.add_plot(x, last, color=c, col=col, lw=0.5)
                    if 'linkpos_actual' in df.columns:
                        actual = [df[df['distance-to-go'] == dtg]['linkpos_actual'].mean()
                                  for dtg in range(1, k)]
                        # normalization
                        actual = [e/l for e, l in zip(actual, length)]
                        p.add_plot(x, actual, color=c, col=col,
                                   label='actual link position', ls='dashed')
                else:
                    p.add_plot(x, first, color=c, marker=m, col=col,
                               label='link position')
                # p.add_plot(x, length, color=c, col=col, label='article length',
                #            ls='dotted')

        path = os.path.join(self.plot_folder, 'linkpos' + fname_suffix)
        p.finish(path, suptitle='Clicked Link Position', titles=titles,
                 xlabel=xlabel, ylabel='Fraction of article length', legend='all',
                 invert_yaxis=True)
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
            # ('spl_target', 'Shortest Path Length to Target', ''),
            # ('tfidf_target', 'TF-IDF similarity to Target', ''),
            # ('degree_out', 'Outdegree', ''),
            ('degree_in', 'Indegree', 'indegree'),
            # ('ngram', 'N-Gram Occurrences', 'occurrences (log)'),
            # ('view_count', 'View Count', 'view count'),
            # ('category_depth', 'Category Specificity', 'category depth'),
            # ('category_target', 'Category Distance to target', ''),
            # ('linkpos_ib', 'Fraction of clicked Links in Infobox', 'Fraction of links'),
            # ('linkpos_lead', 'Fraction of clicked Links in Lead', 'Fraction of links'),
            # ('link_context', 'Number of Links +/- 10 words from clicked link', 'Number of links'),

            # ('perc_deg_in', 'Indegree Percentage', ''),
            # ('perc_ngram', 'Ngram Percentage', ''),
            # ('perc_view_count', 'Ngram Percentage', ''),
            #
            # ('dev_av_deg_in', 'Indegree Deviation from Average', ''),
            # ('dev_av_ngram', 'Ngram Deviation from Average', ''),
            # ('dev_av_view_count', 'View Count Deviation from Average', ''),
            #
            # ('dev_md_deg_in', 'Indegree Deviation from Median', ''),
            # ('dev_md_ngram', 'Ngram Deviation from Median', ''),
            # ('dev_md_view_count', 'View Count Deviation from Median', ''),
        ]:
            print(feature)
            p = Plot(labels, len(data))
            for label, dataset in data.items():
                x = labels.index(label)
                for k, m, c in zip([4, 5, 6, 7, 8, 9], self.markers, self.colors):
                    # filter the dataset
                    df = dataset[dataset['pl'] == k]
                    if not df.shape[0]:
                        continue
                    df['ngram'] = df['ngram'].apply(np.log)
                    df = df[['distance-to-go', 'subject', 'pl', feature]]
                    df.rename(columns={'pl': 'Game length'}, inplace=True)
                    p.add_tsplot(df, col=x, time='distance-to-go',
                                 unit='subject', condition='Game length',
                                 value=feature, marker=m, color=c)
            yinv = True if feature == 'category_depth' else False
            if 'perc' in feature:
                ylim = (0.2, 0.7)
            elif 'linkpos' in feature:
                ylim = (0, 0.5)
            else:
                ylim = None
            path = os.path.join(self.plot_folder, feature + fname_suffix)
            p.finish(path, suptitle=title, titles=titles, xlabel=xlabel,
                     ylabel=ylabel, invert_xaxis=True, invert_yaxis=yinv,
                     ylim=ylim, ylabeltok=True)

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
                    df = dataset[dataset['pl'] == k]
                    df = df[['distance-to-go', 'subject', 'pl', feature]]
                    df.rename(columns={'pl': 'Game length'}, inplace=True)
                    p.add_tsplot(df, col=idx, time='distance-to-go',
                                 unit='subject', condition='Game length',
                                 value=feature, marker=m, color=c)
            titles = np.array([titles])
            path = os.path.join(self.plot_folder, '_'.join(features))
            p.finish(path, titles=titles, xlabel=xlabel, ylabel=ylabel,
                     invert_xaxis=True, suptitle=suptitle)

    def print_click_stats(self):
        print('Statistics for WIKTI')
        df = self.data['WIKTI']
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
        print('of those with multiple positions,',
              '%.2f%% first, %.2f%% last, %.2f%% inbetween out of %d total' %
              (100 * first/entire, 100 * last/entire,
               100 - 100 * (first + last) / entire, entire))
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
        print('of those inbetween,',
              '%.2f%% closer to first, %.2f%% closer to last out of %d total' %
              (100 * ff/total, 100 * ll/total, total))

    def print_game_stats(self):
        for label, dataset in self.data.items():
            df = dataset[dataset['distance-to-go'] == 0]
            df['mission'] = df['start'] + '-' + df['target']
            print(label, df['mission'].value_counts(), df.shape)

    def plot_split(self):
        print('plot_split()')
        df = self.data['Wikispeedia']
        data = [
            {
                'all': df,
                'easy': df[~df['above_pl_mission_mean']],
                'hard': df[df['above_pl_mission_mean']],
            },
            # {
            #     'all': df,
            #     'slow users': df[~df['above_pl_user_mean']],
            #     'fast users': df[df['above_pl_user_mean']],
            # },
        ]
        labels = [
            [
                'all',
                'easy',
                'hard'
            ],
            # ['all', 'fast users', 'slow users'],
        ]
        suffices = [
            '_missions',
            # '_users',
        ]
        for dataset, label, suffix in zip(data, labels, suffices):
            # self.plot_comparison(dataset, label, fname_suffix=suffix)

            del data[0]['all']
            del labels[0][0]
            self.plot_linkpos_fill_between(dataset, label, fname_suffix=suffix,
                                           full=False)

    def feature_combinations(self, features):
        for ai, a in enumerate(features):
            for bi, b in enumerate(features):
                if ai < bi:
                    yield (a, b)

    def correlation_clicked(self):
        for label, dataset in self.data.items():
            print(label)
            for f1, f2 in self.feature_combinations([
                'degree_in',
                'category_depth',
                'ngram',
            ]):
                print('   ', f1, '|', f2)
                df = dataset[[f1, f2]]
                df = df[(df[f1] != 0) & (df[f2] != 0)]
                # df[f1] = df[f1].apply(np.log)
                # df[f2] = df[f2].apply(lambda x: np.log(x * -1))
                r = scipy.stats.pearsonr(df[f1], df[f2])[0]
                rho = scipy.stats.spearmanr(df[f1], df[f2])[0]
                tau = scipy.stats.kendalltau(df[f1], df[f2])[0]

                print('    r = %.2f, rho = %.2f, tau = %.2f\n' % (r, rho, tau))

                sns.jointplot(f1, f2, df, kind='reg', color='#4CB391')
                fname = 'corr_' + f1 + '_' + f2 + '_' + label + '.png'
                plt.title(label)
                # plt.show()
                plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95,
                                    top=0.95, wspace=0.3, hspace=0.3)
                plt.savefig(os.path.join(self.plot_folder,
                                         'correlation', fname))

    def correlation_all(self):
        path = os.path.join('data', 'Wikispeedia', 'data_correlation.obj')
        dataset = pd.read_pickle(path)
        for f1, f2 in self.feature_combinations([
            'degree_in',
            'category_depth',
            'view_count',
            'ngram',
        ]):
            print('   ', f1, '|', f2)
            df = dataset[[f1, f2]]
            df = df[(df[f1] != 0) & (df[f2] != 0)]
            # df[f1] = df[f1].apply(np.log)
            # df[f2] = df[f2].apply(lambda x: np.log(x * -1))
            r = scipy.stats.pearsonr(df[f1], df[f2])[0]
            rho = scipy.stats.spearmanr(df[f1], df[f2])[0]
            tau = scipy.stats.kendalltau(df[f1], df[f2])[0]
            print('    r = %.2f, rho = %.2f, tau = %.2f\n' % (r, rho, tau))
            # sns.jointplot(f1, f2, df, kind='reg', color='#4CB391')
            # plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95,
            #                     top=0.95, wspace=0.3, hspace=0.3)
            # plt.show()

    def correlation_max(self):
        for label, dataset in self.data.items():
            print(label)
            dataset.index = np.arange(dataset.shape[0])
            d = dataset[(dataset['pl'] == k) & (dataset['step'] == 1)]
            sample = np.random.choice(d.index.values, 281, replace=False)
            df = [dataset.ix[sample] for k in [4, 5, 6, 7]]
            df = pd.concat(df)
            # df = dataset
            # gb = df.groupby('subject')
            # pl = gb['pl'].mean()
            # df = df[df['step'] == 1]
            pl = df['pl']
            # hugo = df.groupby('pl').mean()
            pdb.set_trace()
            for f1 in [
                'degree_in',
                'category_depth',
                'ngram',
            ]:
                print('   ', f1)
                # feature = gb[f1].max()
                feature = df[f1]
                # pdb.set_trace()
                r = scipy.stats.pearsonr(pl, feature)[0]
                rho = scipy.stats.spearmanr(pl, feature)[0]
                tau = scipy.stats.kendalltau(pl, feature)[0]

                print('    r = %.2f, rho = %.2f, tau = %.2f\n' % (r, rho, tau))

                sns.jointplot(pl, feature, kind='kde', color='#4CB391')
                plt.title(label)
                # plt.show()
                plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95,
                                    top=0.95, wspace=0.3, hspace=0.3)
                fname = 'corr_4_' + f1 + '_' + label + '.png'
                plt.savefig(os.path.join(self.plot_folder,
                                         'correlation', fname))


def plot_models():
    markers = ['o', '*', 'd', 'v', '^', 's', 'h', '8', '+', '*']
    colors = ['black', "#A03003", "#FB6023", "#235847", "#46AF8E", "#8DA0CB"]
    model_labels = [
        'Random',
        'Degree',
        'TF-IDF',
        'N-gram',
        'View Count',
        'Lead + IB',
        # 'Category',
        # 'LinkPosDegree',
        # 'LinkPosNgram',
        # 'LinkPosViewCount',
    ]
    path = os.path.join('data', 'Wikispeedia', 'models.obj')
    df_full = pd.read_pickle(path)
    df_full['model'] = df_full['model'].apply(lambda x: x.replace(' 0.40', ''))
    df_full['model'] = df_full['model'].apply(lambda x: x.replace('Ngram', 'N-gram'))
    df_full['distance-to-go'] = df_full['pl'] - 1 - df_full['step']
    label2title = {
        'all': 'All Games',
        'no usa': 'Games not passing through to the U.S. article',
    }
    for label in [
        'all',
        # 'usa',
        'no usa',
    ]:
        print(label)
        df_label = df_full[df_full['df'] == label]
        p = Plot(['gl_4', 'gl_5', 'gl_6', 'gl_7'], len(df_label['pl'].unique()))
        for col_idx, pl in enumerate(sorted(df_label['pl'].unique())):
            df = df_label[(df_label['pl'] == pl)]
            for mdl, m, c in zip(model_labels, markers, colors):
                data = df[df['model'] == mdl]['kld'].tolist()
                x = df[df['model'] == mdl]['distance-to-go'].tolist()
                ls = '--' if mdl == 'Random' else '-'
                p.add_plot(x, data, col=col_idx, label=mdl, marker=m, color=c,
                           ls=ls)
        titles = np.array([['Game length ' + str(int(l))
                            for l in sorted(df_label['pl'].unique())]])
        fpath = os.path.join('plots', 'models_' + label.replace(' ', '_'))
        p.finish(fpath, suptitle=label2title[label], titles=titles,
                 legend='single', xlabel='Distance to-go to target',
                 ylabel='KL divergence (bits)')


def print_models():
    path = os.path.join('data', 'Wikispeedia', 'models.obj')
    df_full = pd.read_pickle(path)
    df_full['model'] = df_full['model'].apply(lambda x: x.replace(' 0.40', ''))

    model_labels = [
        'Random',
        'Degree',
        'Ngram',
        'View Count',
        'TF-IDF',
        'Linkpos',
        # 'Category',
        # 'LinkPosDegree',
        # 'LinkPosNgram',
        # 'LinkPosViewCount',
    ]

    for label in [
        'all',
        # 'usa',
        'no usa',
    ]:
        print('\n', label.upper(), '------------------------------------------')
        df_label = df_full[df_full['df'] == label]
        for col_idx, pl in enumerate(sorted(df_label['pl'].unique())):
            print('    PATH LENGTH:', pl, '--------------------------')
            df = df_label[(df_label['pl'] == pl)]
            for step in range(int(pl) - 1):
                print('        STEP', step)
                df_step = df[df['step'] == step]
                results = {mdl[1]['model']: mdl[1]['kld']
                           for mdl in df_step.iterrows()}
                # pdb.set_trace()
                results = {k: v for k, v in results.items() if k in model_labels}
                for r in sorted(results.items(), key=operator.itemgetter(1)):
                    print('            %.2f\t%s' % (r[1], r[0]))


class Plot(object):
    def __init__(self, labels, ncols=1, filextension='.pdf'):
        """create the plot"""
        self.filextension = filextension
        self.figsizes = {
            2: (5, 3),
            3: (5, 3),
            4: (5, 3)
        }
        self.adjusts = {
            2: {'left': 0.15, 'bottom': 0.2, 'right': 0.97, 'top': 0.90},
            3: {'left': 0.15, 'bottom': 0.2, 'right': 0.97, 'top': 0.90},
            4: {'left': 0.15, 'bottom': 0.2, 'right': 0.97, 'top': 0.90}
        }
        self.figs = [plt.figure(figsize=self.figsizes[ncols]) for n in range(ncols)]
        self.labels = [l.lower() for l in labels]
        self.axes = [f.add_subplot(111) for f in self.figs]

    def add_tsplot(self, data, time, unit, condition, value, **kwargs):
        col = kwargs.pop('col', 0)
        fig, ax = self.figs[col], self.axes[col]
        if not ax.xaxis_inverted():
            ax.invert_xaxis()
        sns.tsplot(data, ax=ax, time=time, unit=unit, condition=condition,
                   value=value, estimator=np.nanmean, legend=False, **kwargs)

    def add_plot(self, x, y, **kwargs):
        col = kwargs.pop('col', 0)
        fig, ax = self.figs[col], self.axes[col]
        if not ax.xaxis_inverted():
            ax.invert_xaxis()
        ax.plot(x, y, **kwargs)

    def add_fill_between(self, x, first, second, **kwargs):
        col = kwargs.pop('col', 0)
        fig, ax = self.figs[col], self.axes[col]
        if not ax.xaxis_inverted():
            ax.invert_xaxis()
        gl = kwargs.pop('gl', False)
        label = kwargs.pop('label', None)
        if label:
            ax.plot(None, label=' ', lw=10, alpha=0.0, **kwargs)
            ax.plot(None, label='Game Length %s' % gl, lw=10, alpha=0.0, **kwargs)
            ax.plot(None, label=label, lw=10, alpha=0.2, **kwargs)
        ax.fill_between(x, first, second, alpha=0.2, **kwargs)

    def match_ylim(self):
        ylim_lower = min(a.get_ylim()[0] for a in self.axes)
        ylim_upper = max(a.get_ylim()[1] for a in self.axes)
        for ax in self.axes:
            ax.set_ylim(ylim_lower, ylim_upper)

    def set_ylim(self, ylim):
        for ax in self.axes:
            ax.set_ylim(ylim[0], ylim[1])

    def add_margin(self, margin=0.05):
        for ax in self.axes:
            ylim = ax.get_ylim()
            length = ylim[1] - ylim[0]
            ax.set_ylim(ylim[0] - np.abs(0.05 * length),
                        ylim[1] + np.abs(0.05 * length))
            xlim = ax.get_xlim()
            length = xlim[1] - xlim[0]
            margin = np.abs(0.05 * length)
            margin0 = margin * - 1 if xlim[0] < xlim[1] else margin
            margin1 = margin * - 1 if xlim[0] > xlim[1] else margin
            ax.set_xlim(xlim[0] + margin0,
                        xlim[1] + margin1)

    def set_only_integer_xticks(self):
        for ax in self.axes:
            xx = ax.get_xaxis()
            xx.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    def set_only_integer_yticks(self):
        for ax in self.axes:
            xx = ax.get_yaxis()
            xx.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    def add_legend(self, legend):
        if legend == 'single':
            plt.legend(loc=0)
        elif legend == 'all':
            for row in range(self.axes.shape[0]):
                for col in range(self.axes.shape[1]):
                    ax = self.axes[row, col]
                    ax.legend(loc=0)

    def ylabeltok(self):
        for fig, ax in zip(self.figs, self.axes):
            fig.canvas.draw()
            ytl = [l.get_text() for l in ax.get_yticklabels()]
            ytl = [str(int(int(l) / 1000)) + 'k' if l else '' for l in ytl]
            ax.set_yticklabels(ytl)

    def plot_legend(self, fig_data, fname, horizontal=True):
        # plot the legend in a separate plot
        fig = plt.figure()
        if horizontal:
            data = fig_data.axes[0].get_legend_handles_labels()
            lgd = plt.figlegend(*data, loc=10, ncol=6)
            fig.canvas.draw()
            bbi = lgd.get_window_extent()  # legend bounding box in display units
            bbit = bbi.transformed(fig.dpi_scale_trans.inverted())  # inches
            bbit_exp = bbit.expanded(1.1, 1.1)  # expanded
            fig.savefig(fname + '_legend' + self.filextension, bbox_inches=bbit_exp)
        else:
            lgd = plt.figlegend(*fig_data.axes[0].get_legend_handles_labels(), loc=10)
            fig.canvas.draw()
            bbi = lgd.get_window_extent()  # legend bounding box in display units
            bbit = bbi.transformed(fig.dpi_scale_trans.inverted())  # inches
            bbit_exp = bbit.expanded(1.1, 1.1)  # expanded
            fig.savefig(fname + '_legend' + self.filextension, bbox_inches=bbit_exp)
        plt.close(fig)

    def finish(self, fname, **kwargs):
        """perform some beautification"""
        suptitle = kwargs.pop('suptitle', '')
        xlabel = kwargs.pop('xlabel', '')
        ylabel = kwargs.pop('ylabel', suptitle)
        invert_xaxis = kwargs.pop('invert_xaxis', False)
        invert_yaxis = kwargs.pop('invert_yaxis', False)
        ylim = kwargs.pop('ylim', False)
        titles = kwargs.pop('titles', False)
        # pdb.set_trace()
        if titles is not None:
            for ax, title in zip(self.axes, titles[0]):
                ax.set_title(title)
        if ylim:
            self.set_ylim(ylim)
        else:
            self.match_ylim()
        self.add_margin()
        for ax in self.axes:
            if invert_xaxis:
                ax.invert_xaxis()
            if invert_yaxis:
                ax.invert_yaxis()
            ax.set_xlabel(xlabel)

        self.axes[0].set_ylabel(ylabel)
        plt.setp(self.axes[1].get_yticklabels(), visible=False)
        # plt.setp(self.axes[2].get_yticklabels(), visible=False)

        # sns.despine(fig=self.fig)
        # pdb.set_trace()
        self.plot_legend(self.figs[0], fname)
        # self.set_only_integer_xticks()
        # self.set_only_integer_yticks()
        # if kwargs.pop('ylabeltok', False):
        #     self.ylabeltok()

        for fig, label in zip(self.figs, self.labels):
            fig.subplots_adjust(**self.adjusts[len(self.figs)])
            fig.savefig(fname + '_' + label + self.filextension)
            # plt.show()
            plt.close(fig)


if __name__ == '__main__':
    for pt in [
        # Plotter(['Wikispeedia']),
        # Plotter(['Wikispeedia'], 4),
        # Plotter(['WIKTI']),
        Plotter(['WIKTI', 'Wikispeedia']),
        # Plotter(['WIKTI', 'WIKTI2']),
        # Plotter(['WIKTI', 'WIKTI2', 'WIKTI3']),
    ]:
        pt.plot_linkpos_fill_between()
        # pt.plot_split()

        # pt.plot_comparison()
        # pt.plot_wikti()
        # pt.print_game_stats()
        # pt.print_click_stats()
        # pt.correlation_clicked()
        # pt.correlation_all()
        # pt.correlation_max()
        # pt.mutual_information()

    # plot_models()
    # print_models()
