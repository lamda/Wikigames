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
    def __init__(self, dataset, kind=None, step=None, spl=None, pl=None):
        self.dataset = dataset
        self.kind = kind
        self.step = step
        self.spl = spl
        self.pl = pl
        self.suffix = self.get_suffix()
        if dataset == 'wikispeedia':
            fpath = 'data/clickmodels/wikispeedia_' + kind + self.suffix +\
                    '.obj'
        elif dataset == 'wikipedia':
            fpath = 'data/clickmodels/wikipedia_all.obj'
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
        self.linkpos_type = 'linkpos_all'

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
        # pdb.set_trace()

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
        df = df[df['successful'] == (self.kind == 'successful')]
        if self.step is not None:
            df = df[df['step'] == self.step]
        if self.spl is not None:
            df = df[df['spl'] == self.spl]
        if self.pl is not None:
            df = df[df['pl'] == self.pl]
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
                'ib_lead',
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


def plot_results(dataset, kind=None, other=True, normalized=False,
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

    print('\n\n', dataset, kind, '\n', se)
    ax = plt.subplot(111)
    b = se.plot(ax=ax, kind='bar', legend=False, width=0.6, rot=70, fontsize=18)
    bars = filter(lambda x: isinstance(x, matplotlib.patches.Rectangle),
                  b.get_children())
    for bar, c in zip(bars, colors):
        bar.set_color(c)
    plt.tight_layout()
    if normalized:
        plt.ylim(0, 1.075)
    else:
        # plt.ylim(0, max(se) * 1.075)
        plt.ylim(0, 6.5)
    label_offset = max(se) * 0.01
    for p in ax.patches:
        ax.annotate(
            '%.2f' % p.get_height(),
            (p.get_x() + p.get_width() / 2., p.get_height() + label_offset),
            ha='center',
            fontsize=14,
        )
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


def compare_models_stepwise():
    df_result = pd.DataFrame(columns=['df', 'pl', 'step', 'model', 'kld'])
    for label in [
        # 'all'
        # 'usa',
        'no_usa',
    ]:
        print('+++++++++++++++++', label, '+++++++++++++++++')
        for pl in [
            4,
            5,
            6,
            7
        ]:
            print('----------------PATH LENGTH', pl, '----------------')
            for step in range(pl-1):
                print('\n--------', step, '--------')
                cm = ClickModel('wikispeedia', kind='successful_' + label,
                                step=step, spl=3, pl=pl)
                keys, klds = cm.run(tfidf=True, areas=True)
                results = {}
                for key, kld in zip(keys, klds):
                    results[key] = kld
                    idx = df_result.index.shape[0]
                    df_result.loc[idx] = [label, pl, step, key, kld]
                for r in sorted(results.items(), key=operator.itemgetter(1)):
                    print('%.2f\t%s' % (r[1], r[0]))
            df_result.to_pickle('data/clickmodels/models_stepwise_' + label +
                                '_pl_' + unicode(pl) + '.obj')


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
    label2title = {
        'all': 'All Games',
        'usa': 'Games passing through to the U.S. article',
        'no_usa': 'Games not passing through to the U.S. article',
    }
    pls = [
        '4',
        '5',
        '6',
        '7',
    ]
    plot_labels = ['gl_' + unicode(int(gl)) for gl in pls]
    for label in [
            'all',
            # 'usa',
            'no_usa',
    ]:
        print(label)
        p = Plot(plot_labels, len(pls), filextension='.png')
        for col_idx, pl in enumerate(pls):
            print('    ', pl)
            df = pd.read_pickle(
                'data/clickmodels/models_stepwise_' + label + '_pl_' + pl +
                '.obj'
            )
            df['model'] = df['model'].apply(convert_label)
            df['distance-to-go'] = df['pl'] - 1 - df['step']

            for mdl, c, m in plot_settings:
                data = df[df['model'] == mdl]['kld'].tolist()
                x = df[df['model'] == mdl]['distance-to-go'].tolist()
                ls = '--' if mdl == 'Random' else '-'
                p.add_plot(x, data, col=col_idx, label=mdl, marker=m, color=c,
                           ls=ls)
        fpath = os.path.join('plots', 'models_' + label.replace(' ', '_'))
        p.finish(fpath, suptitle=label2title[label], xlim=(0, 6),
                 legend='external', xlabel='Distance to-go to target',
                 ylabel='KL divergence (bits)', invert_xaxis=True,
                 show=False)


if __name__ == '__main__':
    get_area_importance()

    # --------------------------------------------------------------------------
    # compute stepwise
    # compare_models_stepwise()

    # plot stepwise
    # plot_models()

    # --------------------------------------------------------------------------
    # compute aggregated
    # cm = ClickModel('wikipedia'); cm.run(areas=True)

    # for kind in [
    #     'all',
    #     'successful',
    #     'unsuccessful'
    # ]:
    #     print(kind)
    #     cm = ClickModel('wikispeedia', kind); cm.run(areas=True)


    # plot aggregated
    # plot_results('wikipedia', normalized=False)
    # plot_results('wikipedia', normalized=True)

    # for kind in [
    #     'all',
    #     'successful',
    #     'unsuccessful'
    # ]:
    #     print('Wikispeedia (', kind, ')')
    #     plot_results('wikispeedia', kind, normalized=False)
    #     plot_results('wikispeedia', kind, normalized=True)