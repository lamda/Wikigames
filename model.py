# -*- coding: utf-8 -*-

from __future__ import division, print_function


class NavigationModel(object):
    def __init__(self, df):
        self.df = df

    def compute(self):
        pass

    def to_distribution(self):
        pass


class GroundTruthModel(NavigationModel):
    def __init__(self, df):
        super(GroundTruthModel, self).__init__(df)


class RandomModel(NavigationModel):
    def __init__(self, df):
        super(RandomModel, self).__init__(df)


class DegreeModel(NavigationModel):
    def __init__(self, df):
        super(DegreeModel, self).__init__(df)


class FamiliarityModel(NavigationModel):
    def __init__(self, df):
        super(FamiliarityModel, self).__init__(df)
