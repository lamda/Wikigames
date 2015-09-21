# -*- coding: utf-8 -*-

from __future__ import division, print_function

import urllib2

import credentials
from decorators import Cached


class NgramFrequency(object):
    """
    queries http://weblm.research.microsoft.com/info/rest.html
    more information: http://blogs.msdn.com/b/webngram/
    """
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
        title = urllib2.quote(title.encode('utf-8'))
        url = self.url + title
        trials = 0
        data = None
        while not data:
            try:
                data = float(urllib2.urlopen(url).read())
            except (urllib2.HTTPError, urllib2.URLError) as e:
                if trials > 5:
                    print(title, e)

        return data

ngram_frequency = NgramFrequency()
