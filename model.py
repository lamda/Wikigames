# -*- coding: utf-8 -*-

from __future__ import division, print_function

import pdb

import numpy as np
import scipy.stats
import sklearn.metrics

import decorators
import ngram
import viewcounts


class NavigationModel(object):
    def __init__(self, start, pos, wikigame, label):
        self.start = start
        self.pos = pos
        self.wikigame = wikigame
        self.node2weight = {n: 0.000001 for n in self.wikigame.id2name}
        self.data = None
        self.label = label
        self.window = None
        # self.compute()

    def compute(self):
        raise NotImplementedError

    def get_neighbors(self, node, pos,
                      window_type='words', window=False, names=False):
        node = self.wikigame.id2name[node]
        if window:
            if window_type == 'words':
                neighbors = [n for p, n in self.wikigame.pos2link[node].items()
                             if (pos-window) <= p <= (pos+window)]
            elif window_type == 'links':
                positions = sorted(self.wikigame.pos2link[node])
                index = positions.index(pos)
                positions = positions[max(0, index-window):min(index+window+1, len(positions))]
                neighbors = [self.wikigame.pos2link[node][k] for k in positions]
        else:
            neighbors = [n for n in self.wikigame.pos2link[node].values()]
        if names:
            neighbors = [(n, self.wikigame.id2name[n]) for n in neighbors]
        return neighbors

    def set_data(self):
        total = sum(self.node2weight.values())
        self.data = [self.node2weight[k] / total
                     for k in sorted(self.node2weight)]

    def compare_to(self, mdl):
        def sig_stars(p):
            """
            Return a R-style significance string corresponding to p values.
            borrowed from seaborn/utils.py
            """
            if p < 0.001:
                return "***"
            elif p < 0.01:
                return "** "
            elif p < 0.05:
                return "*  "
            elif p < 0.1:
                return ".  "
            return "   "

        kl = self.get_kld(mdl)
        # ks = np.abs(scipy.stats.ks_2samp(self.data, mdl.data))
        # rmse = np.log2(sklearn.metrics.mean_squared_error(self.data, mdl.data))
        lab = self.label + '\t' + mdl.label
        # print('\t%.2f\t%.2f\t%s\t%.5f\t%s'
        #       % (kl, ks[0], sig_stars(ks[1]), rmse, lab))
        print('\t%.2f\t%s' % (kl, lab))

    def get_kld(self, mdl):
        return np.abs(scipy.stats.entropy(self.data, mdl.data, base=2))


class GroundTruthModel(NavigationModel):
    def __init__(self, start, pos, first, wikigame):
        self.first = first
        super(GroundTruthModel, self).__init__(start, pos, wikigame,
                                               'Ground Truth')

    def compute(self):
        vc = self.first.value_counts()
        for k, v in zip(vc.index, vc.values):
            self.node2weight[k] += v
        self.set_data()
        # hugo = {self.wikigame.id2title[k]: v for k, v in self.node2weight.items()}
        # pdb.set_trace()


class UniformModel(NavigationModel):
    def __init__(self, start, pos, wikigame):
        super(UniformModel, self).__init__(start, pos, wikigame, 'Uniform')

    def compute(self):
        for node, pos in zip(self.start, self.pos):
            neighbors = self.get_neighbors(node, pos, window=self.window)
            for n in neighbors:
                self.node2weight[n] += 1
        pdb.set_trace()
        self.set_data()


class DegreeModel(NavigationModel):
    def __init__(self, start, pos, wikigame):
        super(DegreeModel, self).__init__(start, pos, wikigame, 'Degree')

    def compute(self):
        for node, pos in zip(self.start, self.pos):
            neighbors = self.get_neighbors(node, pos, window=self.window)
            total = sum(self.wikigame.id2deg_in[nb] for nb in neighbors)
            for nb in neighbors:
                self.node2weight[nb] += self.wikigame.id2deg_in[nb] / total
        self.set_data()


class ViewCountModel(NavigationModel):
    def __init__(self, start, pos, wikigame):
        super(ViewCountModel, self).__init__(start, pos, wikigame, 'View Count')

    def compute(self):
        vc = viewcounts.viewcount
        for node, pos in zip(self.start, self.pos):
            neighbors = self.get_neighbors(node, pos, window=self.window,
                                           names=True)
            total = sum(vc.get_frequency(nb[1]) for nb in neighbors)
            for nid, nb in neighbors:
                self.node2weight[nid] += vc.get_frequency(nb) / total
        self.set_data()


class NgramModel(NavigationModel):
    def __init__(self, start, pos, wikigame):
        super(NgramModel, self).__init__(start, pos, wikigame,
                                               'Ngram')

    def compute(self):
        ng = ngram.ngram_frequency
        for node, pos in zip(self.start, self.pos):
            neighbors = self.get_neighbors(node, pos, window=self.window, names=True)
            total = sum(np.exp(ng.get_frequency(nb[1])) for nb in neighbors)
            for nid, neighbor in neighbors:
                self.node2weight[nid] += np.exp(ng.get_frequency(neighbor)) / total
        self.set_data()


class CategoryModel(NavigationModel):
    def __init__(self, start, pos, wikigame):
        super(CategoryModel, self).__init__(start, pos, wikigame, 'Category')

    def compute(self):
        for node, pos in zip(self.start, self.pos):
            neighbors = self.get_neighbors(node, pos, window=self.window)
            total = sum(self.wikigame.get_category_depth(n) for n in neighbors)
            for nb in neighbors:
                self.node2weight[nb] += self.wikigame.get_category_depth(nb) / total
        self.set_data()


class TfidfModel(NavigationModel):
    def __init__(self, start, pos, wikigame):
        super(TfidfModel, self).__init__(start, pos, wikigame, 'TF-IDF')

    def compute(self):
        for node, pos in zip(self.start, self.pos):
            neighbors = self.get_neighbors(node, pos, window=self.window)
            total = sum(self.wikigame.get_tfidf_similarity(node, n) for n in neighbors)
            for nb in neighbors:
                self.node2weight[nb] += self.wikigame.get_tfidf_similarity(node, nb) / total
        self.set_data()


class LinkPosModel(NavigationModel):
    def __init__(self, start, pos, wikigame, lead_weight=0.4):
        self.lead_weight = lead_weight
        super(LinkPosModel, self).__init__(start, pos, wikigame,
                                           'Lead + IB')

    def compute(self):
        for i, node in enumerate(self.start):
            node = self.wikigame.id2name[node]
            lim = self.wikigame.lead_length[node]
            lead_nodes = [v for k, v in self.wikigame.pos2link[node].items()
                          if k < lim]
            other_nodes = [v for k, v in self.wikigame.pos2link[node].items()
                           if k > lim]
            for nb in other_nodes:
                self.node2weight[nb] += (1 - self.lead_weight) / len(other_nodes)
            for nb in lead_nodes:
                self.node2weight[nb] += self.lead_weight / len(lead_nodes)

        self.set_data()


class LinkPosDegreeModel(NavigationModel):
    def __init__(self, start, pos, wikigame, lead_weight=0.4):
        self.lead_weight = lead_weight
        super(LinkPosDegreeModel, self).__init__(start, pos, wikigame,
                                                 'LinkPosDegree')
    def compute(self):
        for i, node in enumerate(self.start):
            node = self.wikigame.id2name[node]
            lim = self.wikigame.lead_length[node]
            lead_nodes = [v for k, v in self.wikigame.pos2link[node].items()
                          if k < lim]
            other_nodes = [v for k, v in self.wikigame.pos2link[node].items()
                           if k > lim]
            total_lead = sum(self.wikigame.id2deg_in[nb] for nb in lead_nodes)
            total_other = sum(self.wikigame.id2deg_in[nb] for nb in other_nodes)

            for nb in lead_nodes:
                self.node2weight[nb] += self.lead_weight * \
                    self.wikigame.id2deg_in[nb] / total_lead

            for nb in other_nodes:
                self.node2weight[nb] += (1 - self.lead_weight) * \
                    self.wikigame.id2deg_in[nb] / total_other
        self.set_data()


class LinkPosNgramModel(NavigationModel):
    def __init__(self, start, pos, wikigame, lead_weight=0.4):
        self.lead_weight = lead_weight
        super(LinkPosNgramModel, self).__init__(start, pos, wikigame,
                                                      'LinkPosNgram')

    def compute(self):
        ng = ngram.ngram_frequency
        for i, node in enumerate(self.start):
            node = self.wikigame.id2name[node]
            lim = self.wikigame.lead_length[node]
            lead_nodes = [(v, self.wikigame.id2name[v])
                          for k, v in self.wikigame.pos2link[node].items()
                          if k < lim]
            other_nodes = [(v, self.wikigame.id2name[v])
                           for k, v in self.wikigame.pos2link[node].items()
                           if k > lim]

            total_lead = sum(np.exp(ng.get_frequency(n[1])) for n in lead_nodes)
            total_other = sum(np.exp(ng.get_frequency(n[1])) for n in other_nodes)

            for nid, nb in lead_nodes:
                self.node2weight[nid] += self.lead_weight *\
                    np.exp(ng.get_frequency(nb)) / total_lead
            for nid, nb in other_nodes:
                self.node2weight[nid] += (1 - self.lead_weight) *\
                    np.exp(ng.get_frequency(nb)) / total_other

        self.set_data()


class LinkPosViewCountModel(NavigationModel):
    def __init__(self, start, pos, wikigame, lead_weight=0.4):
        self.lead_weight = lead_weight
        super(LinkPosViewCountModel, self).__init__(start, pos, wikigame,
                                                      'LinkPosViewCount')

    def compute(self):
        vc = viewcounts.viewcount
        for i, node in enumerate(self.start):
            node = self.wikigame.id2name[node]
            lim = self.wikigame.lead_length[node]
            lead_nodes = [(v, self.wikigame.id2name[v])
                          for k, v in self.wikigame.pos2link[node].items()
                          if k < lim]
            other_nodes = [(v, self.wikigame.id2name[v])
                           for k, v in self.wikigame.pos2link[node].items()
                           if k > lim]

            total_lead = sum(vc.get_frequency(n[1]) for n in lead_nodes)
            total_other = sum(vc.get_frequency(n[1]) for n in other_nodes)

            for nid, nb in lead_nodes:
                self.node2weight[nid] += self.lead_weight *\
                    vc.get_frequency(nb) / total_lead
            for nid, nb in other_nodes:
                self.node2weight[nid] += (1 - self.lead_weight) *\
                    vc.get_frequency(nb) / total_other

        self.set_data()
