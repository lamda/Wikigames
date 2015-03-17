# -*- coding: utf-8 -*-

from __future__ import division, print_function

import pdb

import numpy as np
import scipy.stats

import ngram
import viewcounts


class NavigationModel(object):
    def __init__(self, start, wikigame, label):
        self.start = start
        self.wikigame = wikigame
        self.node2weight = {n: 0.000001 for n in self.wikigame.id2name}
        self.data = None
        self.label = label

    def compute(self):
        raise NotImplementedError

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

        kl = np.abs(scipy.stats.entropy(self.data, mdl.data, base=2))
        ks = np.abs(scipy.stats.ks_2samp(self.data, mdl.data))
        lab = self.label + '\t' + mdl.label
        print('\t%.2f\t%.2f\t%s\t%s' % (kl, ks[0], sig_stars(ks[1]), lab))


class GroundTruthModel(NavigationModel):
    def __init__(self, start, first, wikigame):
        super(GroundTruthModel, self).__init__(start, wikigame, 'Ground Truth')
        self.first = first
        self.compute()

    def compute(self):
        vc = self.first.value_counts()
        for k, v in zip(vc.index, vc.values):
            self.node2weight[k] += v
        self.set_data()


class RandomModel(NavigationModel):
    def __init__(self, start, wikigame):
        super(RandomModel, self).__init__(start, wikigame, 'Random')
        self.compute()

    def compute(self):
        for i, node in enumerate(self.start):
            node = self.wikigame.id2name[node]
            for neighbor in self.wikigame.pos2link[node].values():
                self.node2weight[neighbor] += 1
        self.set_data()


class DegreeModel(NavigationModel):
    def __init__(self, start, wikigame):
        super(DegreeModel, self).__init__(start, wikigame, 'Degree')
        self.compute()

    def compute(self):
        for i, node in enumerate(self.start):
            node = self.wikigame.id2name[node]
            neighbors = self.wikigame.pos2link[node].values()
            total = sum(self.wikigame.id2deg_in[nb] for nb in neighbors)
            for nb in neighbors:
                self.node2weight[nb] += self.wikigame.id2deg_in[nb] / total
        self.set_data()


class ViewCountModel(NavigationModel):
    def __init__(self, start, wikigame):
        super(ViewCountModel, self).__init__(start, wikigame, 'View Count')
        self.compute()

    def compute(self):
        vc = viewcounts.viewcount
        for i, node in enumerate(self.start):
            node = self.wikigame.id2name[node]
            neighbors = [(n, self.wikigame.id2name[n])
                         for n in self.wikigame.pos2link[node].values()]
            total = sum(vc.get_frequency(nb[1]) for nb in neighbors)
            for nid, nb in neighbors:
                self.node2weight[nid] += vc.get_frequency(nb) / total
        self.set_data()


class FamiliarityModel(NavigationModel):
    def __init__(self, start, wikigame):
        super(FamiliarityModel, self).__init__(start, wikigame, 'Familiarity')
        self.compute()

    def compute(self):
        ng = ngram.ngram_frequency
        for i, node in enumerate(self.start):
            node = self.wikigame.id2name[node]
            neighbors = [(n, self.wikigame.id2name[n])
                         for n in self.wikigame.pos2link[node].values()]
            total = sum(np.exp(ng.get_frequency(nb[1])) for nb in neighbors)
            for nid, neighbor in neighbors:
                self.node2weight[nid] += np.exp(ng.get_frequency(neighbor)) / total
        self.set_data()


class CategoryModel(NavigationModel):
    def __init__(self, start, wikigame):
        super(CategoryModel, self).__init__(start, wikigame, 'Category')
        self.compute()

    def compute(self):
        for i, node in enumerate(self.start):
            node = self.wikigame.id2name[node]
            neighbors = self.wikigame.pos2link[node].values()
            total = sum(self.wikigame.get_category_depth(n) for n in neighbors)
            for nb in neighbors:
                self.node2weight[nb] += self.wikigame.get_category_depth(nb) / total
        self.set_data()


class LinkPosModel(NavigationModel):
    def __init__(self, start, wikigame, lead_weight=0.4):
        super(LinkPosModel, self).__init__(start, wikigame,
                                           'Linkpos %.2f' % lead_weight)
        self.lead_weight = lead_weight
        self.compute()

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
    def __init__(self, start, wikigame, lead_weight=0.4):
        super(LinkPosDegreeModel, self).__init__(start, wikigame,
                                                 'LinkPosDegree %.2f' %
                                                 lead_weight)
        self.lead_weight = lead_weight
        self.compute()

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


class LinkPosFamiliarityModel(NavigationModel):
    def __init__(self, start, wikigame, lead_weight=0.4):
        super(LinkPosFamiliarityModel, self).__init__(start, wikigame,
                                                      'LinkPosFamiliarity %.2f'
                                                      % lead_weight)
        self.lead_weight = lead_weight
        self.compute()

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
    def __init__(self, start, wikigame, lead_weight=0.4):
        super(LinkPosViewCountModel, self).__init__(start, wikigame,
                                                      'LinkPosViewCount %.2f'
                                                      % lead_weight)
        self.lead_weight = lead_weight
        self.compute()

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
