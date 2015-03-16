# -*- coding: utf-8 -*-

from __future__ import division, print_function

import random

class NavigationModel(object):
    def __init__(self, start, wikigame):
        self.start = start
        self.wikigame = wikigame
        self.node2weight = {n: 0 for n in self.wikigame.id2name}

    def compute(self):
        pass

    def to_distribution(self):
        pass


class GroundTruthModel(NavigationModel):
    def __init__(self, start, first, wikigame):
        super(GroundTruthModel, self).__init__(start, wikigame)
        self.first = first

    def compute(self):
        vc = self.first.value_counts()
        self.node2weight = {k: v for k, v in zip(vc.index, vc.values)}


class RandomModel(NavigationModel):
    def __init__(self, start, wikigame):
        super(RandomModel, self).__init__(start, wikigame)

    def compute(self):
        for i, node in enumerate(self.start):
            print(i, '/', len(self.start), end='\r')
            node = self.wikigame.id2name[node]
            for neighbor in self.wikigame.pos2link[node].values():
                self.node2weight[neighbor] += 1


class DegreeModel(NavigationModel):
    def __init__(self, start, wikigame):
        super(DegreeModel, self).__init__(start, wikigame)

    @staticmethod
    def weighted_choice(choices):
        # via http://stackoverflow.com/questions/3679694/
        total = sum(w for c, w in choices)
        r = random.uniform(0, total)
        upto = 0
        for c, w in choices:
            upto += w
            if upto > r:
                return c
        assert False, "Shouldn't get here"

    def compute(self):
        pass


class FamiliarityModel(NavigationModel):
    def __init__(self, start, wikigame):
        super(FamiliarityModel, self).__init__(start, wikigame)

    def compute(self):
        pass
