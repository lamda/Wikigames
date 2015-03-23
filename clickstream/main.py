# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

import collections
import pdb
import cPickle as pickle
import HTMLParser
import io
import operator
import os
import random
import re
import urllib2

import numpy as np
import pandas as pd


# set a few options
pd.options.mode.chained_assignment = None
pd.set_option('display.width', 1000)
url_opener = urllib2.build_opener()


def convert_clickstream_to_pandas():
    fname = '2015_02_clickstream.tsv'
    df = pd.read_csv(fname, sep='\t', header=0)
    df = df[['prev_id', 'curr_id', 'n', 'type']]
    df = df.dropna()
    df = df[df['type'] == 'link']
    df = df[['prev_id', 'curr_id', 'n']]
    df.columns = ['prev', 'curr', 'n']
    df.to_pickle('clickstream.obj')


def get_id_dict():
    id2title = {}
    fname = 'enwiki-20150205-page.sql'
    # fname = 'test.txt'
    with io.open(fname, encoding='utf-8') as infile:
        lidx = 1
        for line in infile:
            print(lidx, '/ 3714', end='\r')
            lidx += 1
            if not line.startswith('INSERT'):
                continue
            for page_id, page_namespace, page_title in re.findall(r"\((\d+),(\d+),'([^\']+)", line):
                if page_namespace != '0':
                    continue
                id2title[int(page_id)] = page_title

        with open('id2title.obj', 'wb') as outfile:
            pickle.dump(id2title, outfile, -1)


def get_redirect_dict():
    id2redirect = {}
    fname = 'enwiki-20150205-redirect.sql'
    # fname = 'test.txt'
    with io.open(fname, encoding='utf-8') as infile:
        lidx = 1
        for line in infile:
            print(lidx, '/ 396', end='\r')
            lidx += 1
            if not line.startswith('INSERT'):
                continue
            for page_id, page_namespace, page_title in re.findall(r"\((\d+),(\d+),'([^\']+)", line):
                if page_namespace != '0':
                    continue
                id2redirect[int(page_id)] = page_title

        with open('id2redirect.obj', 'wb') as outfile:
            pickle.dump(id2redirect, outfile, -1)


def resolve_redirects():
    id2title = pd.read_pickle('id2title.obj')
    title2id = {v: k for k, v in id2title.iteritems()}
    id2redirect = pd.read_pickle('id2redirect.obj')

    title2redirect = {}
    idx = 1
    length = len(id2redirect)
    for k, v in id2redirect.iteritems():
        print(idx, '/', length, end='\r')
        idx += 1
        try:
            title2redirect[id2title[k]] = title2id[v]
        except KeyError:
            pass

    with open('title2redirect.obj', 'wb') as outfile:
        pickle.dump(title2redirect, outfile, -1)


def get_wiki_pages(titles):
    for i, title in enumerate(titles):
        print(i+1, '/', len(titles))
        path = os.path.join('wp', title[0].lower())
        if not os.path.exists(path):
                os.makedirs(path)
        if os.path.isfile(os.path.join(path, title + '.htm')):
            print('present')
            continue
        url = 'http://en.wikipedia.org/wiki/' + title
        data = ''
        try:
            request = urllib2.Request(url)
            data = url_opener.open(request).read()
            data = data.decode('utf-8', 'ignore')
        except (urllib2.HTTPError, urllib2.URLError) as e:
            print('!+!+!+!+!+!+!+!+ URLLIB ERROR !+!+!+!+!+!+!+!+')
            print('URLError', e, title)
            continue
        with io.open(os.path.join(path, title + '.htm'), 'w', encoding='utf-8') \
                as outfile:
            outfile.write(data)


def compute_link_positions(titles):
    print('computing link positions...')

    class MLStripper(HTMLParser.HTMLParser):
        def __init__(self):
            HTMLParser.HTMLParser.__init__(self)
            self.reset()
            self.fed = []

        def handle_data(self, d):
            self.fed.append(d)

        def get_data(self):
            return ''.join(self.fed)

        def reset(self):
            self.fed = []
            HTMLParser.HTMLParser.reset(self)

    parser = MLStripper()
    link_regex = re.compile(('(<a href="/wiki/(.+?)" title="[^"]+?[^>]+?">.+?</a>)'))
    folder = 'wp'
    link2pos_first, link2pos_last, pos2link, pos2linklength = {}, {}, {}, {}
    length, ib_length, lead_length = {}, {}, {}
    for i, a in enumerate(titles):
        print(unicode(i+1), '/', unicode(len(titles)), end='\r')
        lpos_first, lpos_last, posl, posll = {}, {}, {}, {}
        fname = os.path.join(folder, a[0].lower(), a + '.htm')
        try:
            with io.open(fname, encoding='utf-8') as infile:
                data = infile.read()
        except UnicodeDecodeError:
            # there exist decoding errors for a few irrelevant pages
            print(fname)
            continue
        data = data.split('<div id="mw-content-text" lang="en" dir="ltr" class="mw-content-ltr">')[1]
        data = data.split('<div id="mw-navigation">')[0]
        regex_results = link_regex.findall(data)
        regex_results = [(r[0], r[1]) for r in regex_results]
        for link in regex_results:
            link = [l for l in link if l]
            data = data.replace(link[0], ' [['+link[1]+']] ')

        # find infobox
        # if '<table' in data[:500]:
        #     idx = data.find('</table>')
        #     data = data[:idx] + ' [[[ENDIB]]] ' + data[idx:]
        # else:
        #     data = ' [[[ENDIB]]] ' + data

        # find lead
        idx = data.find('<span class="mw-headline"')
        if idx == -1:
            data += ' [[[ENDLEAD]]] '
        else:
            data = data[:idx] + ' [[[ENDLEAD]]] ' + data[idx:]

        data = [d.strip() for d in data.splitlines()]
        data = [d for d in data if d]
        text = []
        for d in data:
            parser.reset()
            parser.feed(parser.unescape(d))
            stripped_d = parser.get_data()
            if stripped_d:
                text.append(stripped_d)
        text = ' '.join(text)
        text = text.replace(']][[', ']] [[')
        words = (re.split(': |\. |, |\? |! |\n | |\(|\)', text))
        words = [wo for wo in words if wo]

        idx = words.index('[[[ENDLEAD]]]')
        lead_length[a] = idx
        del words[idx]

        # idx = words.index('[[[ENDIB]]]')
        # ib_length[a] = idx
        # del words[idx]

        # for wi, word in enumerate(reversed(words)):
        #     if word.startswith('[['):
        #         try:
        #             aid = title2id[word[2:-2].replace('%25', '%')]
        #             lpos_first[aid] = len(words) - wi - 1
        #         except KeyError:
        #           pass

        for wi, word in enumerate(words):
            if word.startswith('[['):
                try:
                    aid = title2id[word[2:-2].replace('%25', '%')]
                    posl[wi] = aid
                except KeyError:
                    try:
                        aid = title2id[title2redirect[word[2:-2].replace('%25', '%')]]
                        posl[wi] = aid
                    except KeyError:
                        pass
        pos2link[a] = posl
        length[a] = len(words)
        # for k in sorted(pos2link[a]):
        #     print(k, id2title[pos2link[a][k]])
        # pdb.set_trace()
    path = os.path.join('link_positions.obj')
    with open(path, 'wb') as outfile:
        pickle.dump([pos2link, lead_length], outfile, -1)


def analyze_clicks(titles, split_type='equal'):
    clickstream = pd.read_pickle('clickstream.obj')
    results = {}
    for i, title in enumerate(titles):
        print(i+1, '/', len(titles), end='\r')
        clicks_lead, clicks_rest = 0, 0
        link2pos = collections.defaultdict(list)
        for pos, link in pos2link[title].items():
            link2pos[link].append(pos)
        ll = lead_length[title]
        clicks = clickstream[clickstream['prev'] == title2id[title]]
        for row in clicks.iterrows():
            target = int(row[1]['curr'])
            count = int(row[1]['n'])
            positions = link2pos[target]
            if split_type == 'first' and len(positions):
                positions = [sorted(positions)[0]]
            for p in positions:
                if p < ll:
                    clicks_lead += count / len(positions)
                else:
                    clicks_rest += count / len(positions)
        results[title] = (clicks_lead, clicks_rest)
    with open('results_' + split_type + '.obj', 'wb') as outfile:
        pickle.dump(results, outfile, -1)


def analyze_click_positions(titles):
    results = {}
    for i, title in enumerate(titles):
        print(i+1, '/', len(titles), end='\r')
        ll = lead_length[title]
        links_lead = len([p for p, l in pos2link[title].items() if p <= ll])
        links_rest = len(pos2link[title]) - links_lead
        results[title] = (links_lead, links_rest)
    with open('results_number.obj', 'wb') as outfile:
        pickle.dump(results, outfile, -1)


def get_titles():
    folders = os.listdir('wp')
    titles = []
    for folder in folders:
        path = os.path.join('wp', folder)
        titles += [f.split('.htm')[0] for f in os.listdir(path)]
    titles = [t for t in titles if '%' not in t]
    titles = [t for t in titles if '__' not in t]
    return titles


def print_results():
    for label, fname in [
        # ('first', 'results_first.obj'),
        # ('split', 'results_split.obj'),
        ('number', 'results_number.obj'),
    ]:
        print(label)
        results = pd.read_pickle(fname)
        data = []
        for lead, rest in results.values():
            total = lead + rest
            if total:
                data.append(lead / total)

        print(np.mean(data), np.median(data))


if __name__ == '__main__':

    # pos2link, lead_length = pd.read_pickle('link_positions.obj')
    # id2title = pd.read_pickle('id2title.obj')
    # title2id = {v: k for k, v in id2title.iteritems()}
    # title2redirect = pd.read_pickle('title2redirect.obj')

    # titles = get_titles()
    # get_wiki_pages(titles)
    # compute_link_positions(titles)

    # analyze_clicks(titles, split_type='first')
    # analyze_click_positions(titles)

    print_results()



