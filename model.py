# -*- coding: utf-8 -*-

from __future__ import division, print_function

import pdb

import ngram
import viewcounts


class NavigationModel(object):
    def __init__(self, start, wikigame):
        self.start = start
        self.wikigame = wikigame
        self.node2weight = {n: 1 for n in self.wikigame.id2name}
        self.data = None

    def compute(self):
        raise NotImplementedError

    def set_data(self):
        total = sum(self.node2weight.values())
        self.data = [self.node2weight[k] / total
                     for k in sorted(self.node2weight)]


class GroundTruthModel(NavigationModel):
    def __init__(self, start, first, wikigame):
        print('    GroundTruthModel')
        super(GroundTruthModel, self).__init__(start, wikigame)
        self.first = first
        self.compute()

    def compute(self):
        vc = self.first.value_counts()
        for k, v in zip(vc.index, vc.values):
            self.node2weight[k] += v
        self.set_data()


class RandomModel(NavigationModel):
    def __init__(self, start, wikigame):
        print('    RandomModel')
        super(RandomModel, self).__init__(start, wikigame)
        self.compute()

    def compute(self):
        for i, node in enumerate(self.start):
            node = self.wikigame.id2name[node]
            for neighbor in self.wikigame.pos2link[node].values():
                self.node2weight[neighbor] += 1
        self.set_data()


class DegreeModel(NavigationModel):
    def __init__(self, start, wikigame):
        print('    DegreeModel')
        super(DegreeModel, self).__init__(start, wikigame)
        self.compute()

    def compute(self):
        for i, node in enumerate(self.start):
            node = self.wikigame.id2name[node]
            for neighbor in self.wikigame.pos2link[node].values():
                self.node2weight[neighbor] += self.wikigame.id2deg_in[neighbor]
        self.set_data()


class ViewCountModel(NavigationModel):
    def __init__(self, start, wikigame):
        print('    ViewCountModel')
        super(ViewCountModel, self).__init__(start, wikigame)
        self.compute()

    def compute(self):
        vc = viewcounts.viewcount
        for i, node in enumerate(self.start):
            node = self.wikigame.id2name[node]
            neighbors = [(n, self.wikigame.id2name[n])
                         for n in self.wikigame.pos2link[node].values()]
            for nid, neighbor in neighbors:
                self.node2weight[nid] += vc.get_frequency(neighbor)
        self.set_data()


class FamiliarityModel(NavigationModel):
    def __init__(self, start, wikigame):
        print('    FamiliarityModel')
        super(FamiliarityModel, self).__init__(start, wikigame)
        self.compute()

    def compute(self):
        ng = ngram.ngram_frequency
        for i, node in enumerate(self.start):
            node = self.wikigame.id2name[node]
            neighbors = [(n, self.wikigame.id2name[n])
                         for n in self.wikigame.pos2link[node].values()]
            for nid, neighbor in neighbors:
                self.node2weight[nid] += ng.get_frequency(neighbor)
        self.set_data()
