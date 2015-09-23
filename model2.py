# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

import collections
import cPickle as pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import operator
import os
import pandas as pd
import pdb
import scipy.stats

import decorators
from plotter import Plot

pd.options.mode.chained_assignment = None
pd.set_option('display.width', 400)
plt.style.use('ggplot')  # TODO


class ClickModel(object):
    def __init__(self, dataset, kind=None, step=None, spl=None, pl=None,
                 linkpos='linkpos_all'):
        self.dataset = dataset
        self.kind = kind
        self.step = step
        self.spl = spl
        self.pl = pl
        self.suffix = self.get_suffix()
        stepwise = 'stepwise/' if step is not None else ''
        if dataset == 'wikispeedia':
            fpath = 'data/clickmodels/' + stepwise + 'wikispeedia_' + kind +\
                     self.suffix + '.obj'
        elif dataset == 'wikipedia':
            fpath = 'data/clickmodels/' + stepwise + 'wikipedia_all.obj'
        else:
            print('unrecognized parameter')
            raise NotImplemented
        self.df = pd.read_pickle(fpath)
        # self.df = self.df[self.df['source'] == 'Africa']
        with open('data/clickmodels/' + dataset + '_stats.obj', 'rb') as infile:
            stats_orig = pickle.load(infile)
        self.stats = {}
        for key, val in stats_orig.iteritems():
            self.stats[key] = {k: v for k, v in val.iteritems()}
            # self.stats[key] = {k: np.log(v+1) for k, v in val.iteritems()}
            # self.stats[key] = {k: 1/(v+1) for k, v in val.iteritems()}
        self.sources = set(self.df['source'])
        self.targets = set(self.df['target'])
        self.keys = sorted(self.sources | self.targets)
        func_dict = lambda: {k: 0.0001 for k in self.keys}
        self.data = collections.defaultdict(func_dict)
        self.df_source = self.df.groupby('source')
        self.df_target = self.df.groupby('target')
        self.clicks = {key: self.df_source.get_group(key)['amount'].sum()
                       for key in self.sources}
        self.clicks = {k: v for k, v in self.clicks.items() if v > 0}
        self.wg = None
        self.linkpos_type = linkpos
        print(self.linkpos_type)

    def get_suffix(self):
        suffix = ''
        if self.step is not None:
            suffix += '_step_' + unicode(self.step)
        if self.spl is not None:
            suffix += '_spl_' + unicode(self.spl)
        if self.pl is not None:
            suffix += '_pl_' + unicode(self.pl)
        return suffix

    def update_data(self, label, data2):
        for k, v in data2.iteritems():
            self.data[label][k] += v

    def normalize(self):
        for label in self.data:
            total = sum(self.data[label].values())
            self.data[label] = [self.data[label][k] / total for k in self.keys]

    def compare(self, m2):
        m1 = 'Ground Truth'
        # why use np.abs?
        # kl = np.abs(scipy.stats.entropy(self.data[m1], self.data[m2], base=2))
        kl = scipy.stats.entropy(self.data[m1], self.data[m2], base=2)
        # print('        %.3f\t%s' % (kl, m2))
        return kl

    def ground_truth(self):
        iterable = self.df_target['amount'].sum().iteritems()
        self.update_data('Ground Truth', {k: v for k, v in iterable})

    def uniform(self):
        for key, clicks_total in self.clicks.items():
            df_sub = self.df_source.get_group(key)
            lps = df_sub[self.linkpos_type].apply(len)
            lp_total = lps.sum()
            for k, v in zip(df_sub['target'], lps):
                self.data['Uniform'][k] += clicks_total * v / lp_total

    def area(self, area, leadp=0.5):
        assert leadp <= 1
        for key, clicks_total in self.clicks.items():
            df_sub = self.df_source.get_group(key)
            dkey = area + '_' + unicode(leadp)

            # links within area of interest
            lps = df_sub['linkpos_' + area].apply(len)
            lp_total = lps.sum()
            if lp_total != 0:
                for k, v in zip(df_sub['target'], lps):
                    self.data[dkey][k] += leadp * clicks_total * v / lp_total

            # links outside area of interest
            not_area = 'not_' + area if 'not' not in area else area[4:]
            lps = df_sub['linkpos_' + not_area].apply(len)
            lp_total = lps.sum()
            if lp_total != 0:
                for k, v in zip(df_sub['target'], lps):
                    self.data[dkey][k] += (1-leadp) * clicks_total * v / lp_total

    def proportional(self, stat, stat_label):
        for key, clicks_total in self.clicks.items():
            df_sub = self.df_source.get_group(key)
            lps = df_sub[self.linkpos_type].apply(len)
            prop = df_sub['target'].apply(lambda x: self.stats[stat][x])
            total = np.dot(lps, prop)
            for k, l, d in zip(df_sub['target'], lps, prop):
                self.data[stat_label][k] += clicks_total * l * d / total

    def tfidf(self):
        assert self.dataset == 'wikispeedia'
        from main import Wikispeedia
        self.wg = Wikispeedia()
        self.wg.load_data()
        df = self.wg.data
        df = df[~df['backtrack']]
        if self.kind == 'successful':
            df = df[df['successful']]
        elif self.kind == 'successful_middle':
            df = df[(df['successful']) & (df['step'] != 0) & (df['distance-to-go'] != 1)]
        elif self.kind == 'unsuccessful':
            df = df[~df['successful']]
        if self.step is not None:
            df = df[df['step'] == self.step]
        if self.spl is not None:
            df = df[df['spl'] == self.spl]
        if self.pl is not None:
            df = df[df['pl'] == self.pl]
        df = df.dropna(subset=['node_next']) # for unsuccessful games
        df = df[['node', 'node_id', 'target', 'target_id']]

        for idx, row in enumerate(df.iterrows()):
            print(idx+1, '/', df.shape[0], end='\r')
            row = row[1]
            weights = self.get_tfidf_prob(row['node'], row['node_id'],
                                          row['target_id'])
            for k, v in weights.iteritems():
                self.data['TF-IDF'][k] += v

    def get_tfidf_prob(self, node, node_id, target_id):
        df_sub = self.df_source.get_group(node)
        lps = df_sub[self.linkpos_type].apply(len)
        lmbd = lambda x: self.wg.get_tfidf_similarity(node_id, x)
        prop = df_sub['target_id'].apply(lmbd)
        total = np.dot(lps, prop)
        dct = {}
        for k, l, d in zip(df_sub['target'], lps, prop):
            dct[k] = l * d / total
        return dct

    def run(self, tfidf=False, areas=False):
        print('getting Ground Truth...')
        self.ground_truth()
        print('getting Uniform...')
        self.uniform()
        print('getting degree...')
        self.proportional('deg_in', 'In-Degree')
        print('getting N-Gram...')
        self.proportional('ngram', 'N-Gram')
        print('getting View Count...')
        self.proportional('view_count', 'View Count')
        if tfidf:
            print('getting TF-IDF...')
            self.tfidf()
        if areas:
            print('getting areas...')
            for area in [
                'lead',
                'ib',
                # 'ib_lead',
            ]:
                print('    ', area, '\n')
                for areap in np.arange(0, 1, 0.01):
                # for areap in np.arange(0, 1, 0.25):
                    print('        ', areap, end='\r')
                    self.area(area, areap)
            print()
        self.normalize()
        columns = sorted(self.data)
        columns.remove('Ground Truth')
        data = [self.compare(key) for key in columns]
        columns, data = self.max_area(columns, data)
        se = pd.Series(data=data, index=columns)
        se.sort()
        for key, val in se.iteritems():
            print('%.2f\t%s' % (val, key))
        se.to_pickle(
            'data/clickmodels/' + self.dataset + '_results' +
            ('_' + self.kind if self.kind is not None else '') + self.suffix +
            '.obj'
        )
        return columns, data

    def max_area(self, columns, data):
        keys, vals = [], []
        ib_key, ib_val = '', 100
        lead_key, lead_val = '', 100
        ib_lead_key, ib_lead_val = '', 100
        for c, d in zip(columns, data):
            if c.startswith('ib_lead_') and d < ib_lead_val:
                ib_lead_key, ib_lead_val = c, d
            if c.startswith('ib_') and not c.startswith('ib_lead') and d < ib_val:
                ib_key, ib_val = c, d
            elif c.startswith('lead_') and d < lead_val:
                lead_key, lead_val = c, d
            if not c.startswith('ib_') and not c.startswith('lead_'):
                keys.append(c)
                vals.append(d)
        if ib_val < 100:
            keys.append(ib_key)
            vals.append(ib_val)
        if ib_lead_val < 100:
            keys.append(ib_lead_key)
            vals.append(ib_lead_val)
        if lead_val < 100:
            keys.append(lead_key)
            vals.append(lead_val)
        return keys, vals


def plot_results(dataset, kind=None, normalized=False,
                 step=None, spl=None, pl=None):
    suffix = ''
    if step is not None:
        suffix += '_step_' + unicode(step)
    if spl is not None:
        suffix += '_spl_' + unicode(spl)
    if pl is not None:
        suffix += '_pl_' + unicode(pl)
    se = pd.read_pickle(
        'data/clickmodels/' + dataset + '_results' +
        ('_' + kind if kind is not None else '') + suffix + '.obj'
    )
    se = se[se.index.map(lambda x: 'ib_lead' not in x)]
    plot_settings = [
        ('Uniform', '#000000', 'o'),
        ('In-Degree', '#4daf4a', '*'),
        ('TF-IDF', '#ff7f00', 'd'),
        ('N-Gram', '#a65628', 'v'),
        ('View Count', '#f781bf', '^'),
        ('IB', '#e41a1c', 's'),
        ('Lead', '#377eb8', 'h'),
        ('IB & Lead', '#984ea3', '8'),
    ]
    plot_settings_dict = {s[0]: (s[1], s[2]) for s in plot_settings}
    keys = []
    colors = []
    for k in se.index:
        if 'ib_lead_' in k:
            keys.append('IB & LEAD (%d%%)' %
                        (100 * float(k.rsplit('_', 1)[-1])))
            colors.append(plot_settings_dict['IB & Lead'][0])
            pass
        elif 'ib_' in k:
                keys.append('IB (%d%%)' % (100 * float(k.rsplit('_', 1)[-1])))
                colors.append(plot_settings_dict['IB'][0])
        elif 'lead_' in k:
                keys.append('LEAD (%d%%)' % (100 * float(k.rsplit('_', 1)[-1])))
                colors.append(plot_settings_dict['Lead'][0])
        else:
            keys.append(k)
            colors.append(plot_settings_dict[k][0])
    se = pd.Series(data=se.tolist(), index=keys)
    if normalized:
        # via http://math.stackexchange.com/questions/51482
        se = se.apply(lambda x: 1 - np.exp(-x))

        # alternative approach - divide by max in series
        # se /= max(se)

    print('\n\n', dataset, kind, '\n', se)
    ax = plt.subplot(111)
    b = se.plot(ax=ax, kind='bar', legend=False, width=0.6, rot=70, fontsize=18)
    bars = filter(lambda x: isinstance(x, matplotlib.patches.Rectangle),
                  b.get_children())
    for bar, c in zip(bars, colors):
        bar.set_color(c)
    # if normalized:
    #     plt.ylim(0, 1.075)
    # else:
    #     # plt.ylim(0, max(se) * 1.075)
    #     plt.ylim(0, 6.5)
    if dataset == 'wikispeedia':
        plt.ylim(0, 1.2)
    label_offset = max(se) * 0.01
    for p in ax.patches:
        ax.annotate(
            '%.2f' % p.get_height(),
            (p.get_x() + p.get_width() / 2., p.get_height() + label_offset),
            ha='center',
            fontsize=14,
        )
    plt.ylabel('KL divergence (bits)')
    plt.tight_layout()
    ofname = 'plots/clickmodels_' + dataset +\
             ('_normalized' if normalized else '') +\
             ('_' + kind if kind is not None else '') + suffix
    plt.savefig(ofname + '.pdf')
    plt.savefig(ofname + '.png')
    plt.close()


def get_area_importance():
    for dataset in ['wikispeedia', 'wikipedia']:
        print(dataset)
        df = pd.read_pickle('data/clickmodels/' + dataset + '_all.obj')
        lp_ib = df['linkpos_ib'].apply(len).sum()
        lp_lead = df['linkpos_lead'].apply(len).sum()
        lp_all = df['linkpos_all'].apply(len).sum()
        print('    %.2f (IB) %.2f (LEAD)' % (lp_ib/lp_all, lp_lead/lp_all))


def plot_area_importance():
    label2short = {
        'Indegree': 'indegree',
        'N-Gram Frequency (log)': 'ngram',
        'View Count': 'view_count'
    }
    for dataset, label, data, color in [
        # mean for ib, lead, rest
        # ('Wikipedia', 'Indegree', [16997.5278, 9994.6628, 15510.2125], '#4daf4a'),
        # ('Wikispeedia', 'Indegree', [247.5816, 213.5608, 177.8769], '#4daf4a'),
        #
        # ('Wikipedia', 'N-Gram Frequency (log)', [-6.0379, -5.9888,  -6.4606], '#a65628'),
        # ('Wikispeedia', 'N-Gram Frequency (log)', [-4.9233, -4.8421, -4.9284], '#a65628'),
        #
        # ('Wikipedia', 'View Count', [34115.3295, 41092.9816, 27211.8253], '#f781bf'),
        # ('Wikispeedia', 'View Count', [111909.0179, 129656.7252, 132233.6598], '#f781bf'),

        # median for ib, lead and rest
        ('Wikipedia', 'Indegree', [910, 845, 445], '#4daf4a'),
        ('Wikispeedia', 'Indegree', [110, 99, 76], '#4daf4a'),

        ('Wikipedia', 'N-Gram Frequency (log)', [-7.459, -7.485, -8.225], '#a65628'),
        ('Wikispeedia', 'N-Gram Frequency (log)', [-5.588, -5.261, -5.355], '#a65628'),

        ('Wikipedia', 'View Count', [11761, 13283, 4690], '#f781bf'),
        ('Wikispeedia', 'View Count', [58347, 67845, 76762], '#f781bf'),
    ]:
        ax = plt.subplot(111)
        if 'gram' in label.lower():
            data = map(np.exp, data)
        ax.bar(np.arange(len(data)), data, width=0.5, color=color, align='center')
        #  rot=70, fontsize=18,
        label_offset = max(data) * 0.01
        for p in ax.patches:
            ax.annotate(
                '%.2f' % p.get_height(),
                (p.get_x() + p.get_width() / 2., p.get_height() + label_offset),
                ha='center',
                fontsize=14,
            )
        plt.ylabel(label)
        plt.ylim(0, max(data) * 1.1)
        ax.set_xticks(np.arange(len(data)))
        ax.set_xticklabels(('Lead', 'IB', 'Remainder'), rotation=70)

        plt.tight_layout()
        ofname = 'plots/ib_lead_rest_links_' +\
                 dataset + '_' + label2short[label]
        plt.savefig(ofname + '.pdf')
        plt.savefig(ofname + '.png')
        plt.close()


def get_distribution_stats():
    cm = ClickModel('wikispeedia', 'all')
    cm.ground_truth()
    cm.normalize()
    ws = sorted(cm.data['Ground Truth'])

    cm = ClickModel('wikipedia')
    cm.ground_truth()
    cm.normalize()
    wp = sorted(cm.data['Ground Truth'])

    plt.plot(ws, label='wikispeedia')
    plt.plot(wp, label='wikipedia')
    pdb.set_trace()


def compare_models_stepwise():
    for label in [
        ''
        # '_usa',
        # '_no_usa',
    ]:
        print('+++++++++++++++++', label, '+++++++++++++++++')
        for spl in [
            3,
            # 4,
            # 5,
        ]:
            print('----------------SPL', spl, '----------------')
            # for pl in [
            #     # 4,
            #     # 5,
            #     # 6,
            #     # 7,
            #     # 8,
            #     # 9,
            #     10
            # ]:
            for pl in range(spl+1, 11):
                df_result = pd.DataFrame(columns=['df', 'pl', 'step',
                                                  'model', 'kld'])
                print('    ------------PATH LENGTH', pl, '------------    ')
                for step in range(pl-1):
                    print('\n        --------', step, '--------        ')
                    cm = ClickModel('wikispeedia', kind='successful' + label,
                                    step=step, spl=spl, pl=pl)
                    keys, klds = cm.run(tfidf=True, areas=True)
                    results = {}
                    for key, kld in zip(keys, klds):
                        results[key] = kld
                        idx = df_result.index.shape[0]
                        df_result.loc[idx] = [label, pl, step, key, kld]
                    # for r in sorted(results.items(),
                    #                 key=operator.itemgetter(1)):
                    #     print('%.2f\t%s' % (r[1], r[0]))
                df_result.to_pickle(
                    'data/clickmodels/stepwise/models_stepwise' + label +
                    '_spl_' + unicode(spl) + '_pl_' + unicode(pl) + '.obj'
                )


def plot_models():
    def convert_label(label):
        if 'ib_lead_' in label:
            return 'IB & Lead'
        if 'ib_' in label:
            return 'IB'
        if 'lead_' in label:
            return 'Lead'
        return label

    plot_settings = [
        ('Uniform', '#000000', 'o'),
        ('In-Degree', '#4daf4a', '*'),
        ('TF-IDF', '#ff7f00', 'd'),
        ('N-Gram', '#a65628', 'v'),
        ('View Count', '#f781bf', '^'),
        ('Infobox', '#e41a1c', 's'),
        ('Lead', '#377eb8', 'h'),
        ('Infobox & Lead', '#984ea3', '8'),
    ]
    pls = [
        '4',
        '5',
        '6',
        '7',
    ]
    models = {'Uniform', 'Lead', 'In-Degree', 'TF-IDF'}
    plot_labels = ['gl_' + unicode(int(gl)) for gl in pls]
    p = Plot(plot_labels, len(pls), fileextension=['.pdf', '.png'])
    for col_idx, pl in enumerate(pls):
        print('\n----------------PATH LENGTH', pl, '----------------')
        df = pd.read_pickle(
            'data/clickmodels/stepwise/models_stepwise' +
            '_spl_3_pl_' + pl + '.obj'
        )
        df['model'] = df['model'].apply(convert_label)
        df['distance-to-go'] = df['pl'] - 1 - df['step']
        df = df[df['model'] != 'IB & Lead']
        # pdb.set_trace()
        # print to console
        # for step in range(int(pl)-1):
        #     print('    --------', step, '--------')
        #     for ridx, row in enumerate(df[df['step'] == step].sort('kld').iterrows()):
        #         # if row[1]['model'] not in models:
        #         #     continue
        #         print('        %.2f %s' % (row[1]['kld'], row[1]['model']))
        #         # if ridx > 1:
        #         #     break
        # print()

        for mdl, c, m in plot_settings:
            if mdl not in models:
                continue
            data = df[df['model'] == mdl]['kld'].tolist()
            x = df[df['model'] == mdl]['distance-to-go'].tolist()
            ls = '--' if mdl == 'Random' else '-'
            p.add_plot(x, data, col=col_idx, label=mdl, marker=m, color=c,
                       ls=ls)

    fpath = os.path.join('plots', 'models')
    p.finish(fpath, xlim=(0.5, 6),
             legend='external', xlabel='Distance to-go to target',
             ylabel='KL divergence (bits)', invert_xaxis=True,
             show=False)


def percentage_models():
    def convert_label(label):
        if 'ib_lead_' in label:
            return 'IB & Lead'
        if 'ib_' in label:
            return 'IB'
        if 'lead_' in label:
            return 'Lead'
        return label

    plot_settings = [
        ('Uniform', '#000000', 'o'),
        ('In-Degree', '#4daf4a', '*'),
        ('TF-IDF', '#ff7f00', 'd'),
        ('N-Gram', '#a65628', 'v'),
        ('View Count', '#f781bf', '^'),
        ('IB', '#e41a1c', 's'),
        ('Lead', '#377eb8', 'h'),
        # ('IB & Lead', '#984ea3', '8'),
    ]
    stats = {l: 0 for l in [p[0] for p in plot_settings]}
    stats_first = {l: 0 for l in [p[0] for p in plot_settings]}
    stats_last = {l: 0 for l in [p[0] for p in plot_settings]}
    stats_middle = {l: 0 for l in [p[0] for p in plot_settings]}
    for spl in [
        3,
        4,
        5
    ]:
        for pl in range(spl+1, 11):
            df = pd.read_pickle(
                'data/clickmodels/stepwise/models_stepwise' +
                '_spl_' + unicode(spl) + '_pl_' + unicode(pl) + '.obj'
            )
            df['model'] = df['model'].apply(convert_label)
            df = df[df['model'] != 'IB & Lead']
            for step in df['step'].unique():
                idx = df[df['step'] == step]['kld'].idxmin()
                stats[df[df['step'] == step].loc[idx]['model']] += 1
                if step == 0:
                    stats_first[df[df['step'] == step].loc[idx]['model']] += 1
                elif step == pl-2:
                    stats_last[df[df['step'] == step].loc[idx]['model']] += 1
                else:
                    stats_middle[df[df['step'] == step].loc[idx]['model']] += 1

    for d, label in [
        (stats, 'all'),
        (stats_first, 'first'),
        (stats_middle, 'middle'),
        (stats_last, 'last'),
    ]:
        print(label)
        for k, v in d.items():
            print('    ', k, v)
        print()

    idx = ['Lead', 'IB', 'Uniform', 'TF-IDF',
           'N-Gram', 'View Count', 'In-Degree']
    colors = ['#377eb8', '#e41a1c', '#000000', '#ff7f00',
              '#a65628', '#f781bf', '#4daf4a']
    stats = pd.Series(data=[stats[i] for i in idx], index=idx)
    stats_first = pd.Series(data=[stats_first[i] for i in idx], index=idx)
    stats_middle = pd.Series(data=[stats_middle[i] for i in idx], index=idx)
    stats_last = pd.Series(data=[stats_last[i] for i in idx], index=idx)
    for se, label in [
        (stats, 'all'),
        (stats_first, 'first'),
        (stats_middle, 'middle'),
        (stats_last, 'last'),
    ]:
        se = 100 * se/sum(se)
        ax = plt.subplot(111)
        b = se.plot(ax=ax, kind='bar', legend=False, width=0.6, rot=70,
                    fontsize=18)
        bars = filter(lambda x: isinstance(x, matplotlib.patches.Rectangle),
                      b.get_children())
        for bar, c in zip(bars, colors):
            bar.set_color(c)
        label_offset = max(se) * 0.01
        for p in ax.patches:
            ax.annotate(
                '%.2f%%' % p.get_height(),
                (p.get_x() + p.get_width() / 2., p.get_height() + label_offset),
                ha='center',
                fontsize=14,
            )
        plt.ylabel('Percent of best fits')
        plt.ylim(0, 110)
        plt.tight_layout()
        ofname = 'plots/wikispeedia_stepwise_best_fits_' + label
        plt.savefig(ofname + '.pdf')
        plt.savefig(ofname + '.png')
        plt.close()


if __name__ == '__main__':
    # --------------------------------------------------------------------------
    # get_area_importance()
    plot_area_importance()
    # get_distribution_stats()

    # --------------------------------------------------------------------------
    # cm = ClickModel('wikipedia'); cm.run(areas=True)

    # for kind in [
        # 'all',
        # 'successful',
        # 'successful_middle',
        # 'unsuccessful'
    # ]:
    #     print(kind)
    #     cm = ClickModel('wikispeedia', kind=kind)
    #     cm.run(areas=True, tfidf=True)

    # plot aggregated
    # plot_results('wikipedia', normalized=False)
    # plot_results('wikipedia', normalized=True)
    #
    # for kind in [
    #     'all',
    #     'successful',
    #     'successful_middle',
    #     'unsuccessful'
    # ]:
    #     print('Wikispeedia (', kind, ')')
    #     plot_results('wikispeedia', kind=kind, normalized=False)
    #     plot_results('wikispeedia', kind=kind, normalized=True)

    # --------------------------------------------------------------------------
    # compare_models_stepwise()
    #
    # plot stepwise
    # plot_models()
    # percentage_models()
