# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

import io
import json
import os
import pdb
import random
import requests
import sys
from twisted.internet import reactor, defer, task
from twisted.python import log
from twisted.web import client


class Crawler(object):
    def __init__(self, limit=None):
        self.data_dir = os.path.join('data')
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        titles_all = os.listdir('../data/Wikispeedia/wpcd/plaintext')
        titles_all = set(t[:-4] for t in titles_all)
        titles_wrong = {
            # not relevant
            'Newshounds', 'Directdebit', 'Wowpurchase', 'Friend_Directdebit',
            'Sponsorship_Directdebit',
            'Wikipedia_Text_of_the_GNU_Free_Documentation_License',
            # missing
            'Bionicle__Mask_of_Light',
            'Star_Wars_Episode_IV__A_New_Hope',
            'X-Men__The_Last_Stand',
            # ambiguous
            'Aggregator',
            'Amur',
            'Andrew_Robinson',
            'Anne_of_Great_Britain',
            'Bantu',
            'Battle_of_Amiens',
            'Battle_of_Normandy',
            'Bj%C3%B8rn%C3%B8ya',
            'Blackbird',
            'Boa',
            'Capital',
            'Chaffinch',
            'Cocoa',
            'Conflict',
            'Dark_Ages',
            'Defaka',
            'Doom',
            'Durham',
            'Extinct_birds',
            'Firecrest',
            'Forth',
            'Gallery_of_the_Kings_and_Queens_of_England',
            'Garage_%28dance_music%29',
            'Green_Woodpecker',
            'Helen',
            'Herring_Gull',
            'Income_disparity',
            'Lake_Albert',
            'Market',
            'Newmarket',
            'Pochard',
            'Prehistoric_man',
            'Race',
            'Recorder',
            'RER',
            'Salford',
            'Sandur',
            'Scent_of_a_Woman',
            'Sequoia',
            'Set',
            'Sparrowhawk',
            'Sputnik_program',
            'Weymouth',
            'William_Gilbert',
            'Winfield_Scott_%2528ship%2529',
            'Woodruff',
            'Wood_Pigeon',
            'Zulu',
        }
        titles_right = {
            'Mask_of_Light',
            'Star_Wars_(film)',
            'X-Men_3',
            'News_aggregator',
            'Amur_River',
            'Andrew_Robinson_(actor)',
            'Anne,_Queen_of_Great_Britain',
            'Bantu_peoples',
            'Battle_of_Amiens_(1918)',
            'Bear_Island_(Norway)',
            'Common_blackbird',
            'Boidae',
            'Capital_city',
            'Common_chaffinch',
            'Cocoa_bean',
            'Conflict_(process)',
            'Dark_Ages_(historiography)',
            'Defaka_people',
            'Doom_(1993_video_game)',
            'Durham,_England',
            'List_of_recently_extinct_birds',
            'Common_firecrest',
            'Forth_(programming_language)',
            'UK_garage',
            'European_green_woodpecker',
            'Helen_of_Troy',
            'List_of_English_monarchs',
            'European_herring_gull',
            'Lake_Albert_(Africa)',
            'Market_(economics)',
            'Newmarket,_Suffolk',
            'Common_pochard',
            'Prehistory',
            'Race_(human_classification)',
            'Recorder_(musical_instrument)',
            'Reseau_Express_Regional',
            'City_of_Salford',
            'Operation_Overlord',
            'Outwash_plain',
            'Scent_of_a_Woman_(1992_film)',
            'Sequoia_(genus)',
            'Set_(mathematics)',
            'Eurasian_sparrowhawk',
            'List_of_spacecraft_called_Sputnik',
            'Weymouth,_Dorset',
            'William_Gilbert_(astronomer)',
            'Galium_odoratum',
            'Common_wood_pigeon',
            'SS_Winfield_Scott',
            'Zulu_people',
        }
        titles_all = (titles_all - titles_wrong) | titles_right
        with io.open('../data/Wikispeedia/wpcd/wp_titles.txt', 'w',
                     encoding='utf-8') as outfile:
            for t in sorted(titles_all):
                outfile.write(t + '\n')
        print(len(titles_all), 'titles total')
        file_titles = set(f[:-4] for f in os.listdir(self.data_dir))
        self.titles = titles_all - file_titles
        self.titles = random.sample(self.titles, limit)\
            if limit else self.titles
        self.no_files = len(self.titles)
        self.no_crawled_files = len(titles_all) - len(self.titles)
        self.api_url = 'https://en.wikipedia.org/w/api.php'
        self.rvids_temp = []

    def page_finished(self, results):
        reactor.stop()

    def crawl_twisted(self):

        def parallel(iterable, count, callable, *args, **kwargs):
            coop = task.Cooperator()
            work = (callable(elem, *args, **kwargs) for elem in iterable)
            return defer.DeferredList([coop.coiterate(work) for i in xrange(count)])

        def download(title):
            url = 'https://en.wikipedia.org/w/api.php?format=json'\
                  '&rvstart=20150228235959&prop=revisions|categories&continue' \
                  '&titles=%s&action=query&rvprop=content&rvparse' \
                  '&cllimit=500&clshow=!hidden&redirects=True'
            path = os.path.join(self.data_dir, title + '.txt')
            return client.downloadPage(str(url % title), path)

        print('downloading', len(self.titles), 'files...')
        log.startLogging(Logger(), setStdout=0)
        finished = parallel(self.titles, 50, download)
        # finished = parallel(['Terik'], 50, download)
        finished.addErrback(log.err)
        finished.addCallback(lambda ign: reactor.stop())
        reactor.run()


class Logger(object):
    def __init__(self, counter=0):
        self.counter = counter

    def write(self, text):
        if 'Starting factory' in text:
            pass
        elif 'Stopping factory' in text:
            self.counter += 1
            print(self.counter, '/', 4604 - self.no_files, end='\r')
        else:
            print(text)

    def flush(self):
        pass


if __name__ == '__main__':
    c = Crawler()
    c.crawl_twisted()
