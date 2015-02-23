# -*- coding: utf-8 -*-

from __future__ import division, print_function

import urllib2

import credentials
from decorators import Cached


class NgramFrequency(object):
    def __init__(self):
        self.token = credentials.microsoft_token
        self.ngram = {}
        self.url = 'http://weblm.research.microsoft.com/rest.svc/' +\
                   'bing-query/2013-12/5/jp?u=' + self.token + '&p='

    @Cached
    def get_frequency(self, title):
        return self.retrieve_frequency(title)

    def retrieve_frequency(self, title):
        title = title.replace(' ', '+').replace('_', '+')
        url = self.url + title
        return float(urllib2.urlopen(url).read())

ngram_frequency = NgramFrequency()
