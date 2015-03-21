# -*- coding: utf-8 -*-

from __future__ import division, print_function

import pdb
import re
import urllib
import urllib2

from decorators import Cached


class WikipediaViewCounts(object):
    def __init__(self):
        self.count = {}
        self.url = 'http://stats.grok.se/en/2014'

    @Cached
    def get_frequency(self, title):
        return self.retrieve_frequency(title)

    def retrieve_frequency(self, title):
        title = title.replace(' ', '_').replace('%2F', '/')
        # months = [unicode(i).zfill(2) for i in range(1, 13)]
        months = [unicode(i).zfill(2) for i in [11]]
        views = 0
        for month in months:
            url = self.url + month + '/' + title
            try:
                data = urllib2.urlopen(url).read()
            except urllib2.HTTPError, e:
                print(title, e)
                pdb.set_trace()
            views += int(re.findall(r'has been viewed (\d+)', data)[0])
        return views

viewcount = WikipediaViewCounts()
