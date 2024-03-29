# -*- coding: utf-8 -*-

from __future__ import division, print_function
import atexit
import bisect
import collections
import cPickle as pickle
import datetime
import HTMLParser
import io
import operator
import os
import pdb
import re

import numpy as np
import pandas as pd
import pymysql
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy

import decorators
import ngram
import viewcounts
import model

# set a few options
pd.options.mode.chained_assignment = None
pd.set_option('display.width', 1000)


class DbConnector(object):
    def __init__(self, db):
        self.db_host = '127.0.0.1'
        self.db_connection = pymysql.connect(host=self.db_host,
                                             port=3306,
                                             user='root',
                                             passwd='master',
                                             db=db,
                                             charset='utf8')
        self.db_cursor = self.db_connection.cursor(pymysql.cursors.DictCursor)
        self.db_cursor_nobuff = self.db_connection.cursor(
            pymysql.cursors.SSCursor)
        self.db = db
        atexit.register(self.close)

    def close(self):
        self.db_cursor.close()
        self.db_connection.close()

    def execute(self, _statement, _args=None):
        self.db_cursor.execute(_statement, _args)

        if _statement.lower().startswith("select"):
            return self.db_cursor.fetchall()

    def commit(self):
        self.db_cursor.connection.commit()

    def fetch_cursor(self, _statement, _args):
        self.db_cursor.execute(_statement, _args)
        return self.db_cursor

    def last_id(self):
        return self.db_connection.insert_id()

    def fetch_cursor_nobuff(self, _statement, _args):
        self.db_cursor_nobuff.execute(_statement, _args)
        return self.db_cursor_nobuff


class Wikigame(object):
    def __init__(self, label):
        # print(label)
        self.label = label
        self.data = None
        self.graph = None
        self.html_base_folder = os.path.join('data', label, 'wpcd', 'wp')
        self.plaintext_folder = os.path.join('data', label, 'wpcd', 'plaintext')
        self.cache_folder = os.path.join('data', label, 'cache')

        self.link2pos_first, self.link2pos_last = {}, {}
        self.length, self.pos2link, self.pos2linklength = {}, {}, {}
        self.ib_length, self.lead_length = {}, {}
        self.link_context, self.link_sets = None, None

        # build some mappings from the database
        self.db_connector = DbConnector(self.label)
        pages = self.db_connector.execute('SELECT * FROM pages')
        self.id2title = {p['id']: p['name'] for p in pages}
        self.id2name = {p['id']: re.findall(r'\\([^\\]*?)\.htm', p['link'])[0]
                        for p in pages}
        self.name2id = {v: k for k, v in self.id2name.items()}

        games = self.db_connector.execute("""SELECT * FROM games
                                     WHERE `game_name` LIKE 'PLAIN%'""")
        self.game2start_target = {v['game_name']:
                                  (self.id2name[v['start_page_id']],
                                   self.id2name[v['goal_page_id']])
                                  for v in games}

        nodes = self.db_connector.execute('SELECT * FROM node_data')
        self.id2deg_out = {p['node_id']: p['out_degree'] for p in nodes}
        self.id2deg_in = {p['node_id']: p['in_degree'] for p in nodes}

        links = self.db_connector.execute('''SELECT page_id,
                                                    SUM(amount) as links
                                             FROM links GROUP BY page_id;''')
        self.id2links = {p['page_id']: int(p['links']) for p in links}

    def load_graph(self, graph_tool=False):
        # read the graph
        path = os.path.join('data', self.label, 'links.txt')
        if graph_tool:
            self.graph = self.read_edge_list_gt(path)
        else:
            self.graph = self.read_edge_list_nx(path)

    def read_edge_list_gt(self, filename, directed=True, parallel_edges=False):
        if gt is None:
            import graph_tool.all as gt
        graph = gt.Graph(directed=directed)
        id_mapping = collections.defaultdict(lambda: graph.add_vertex())
        graph.vertex_properties['NodeId'] = graph.new_vertex_property('string')
        with io.open(filename, encoding='utf-8') as infile:
            for line in infile:
                line = line.strip().split()
                if len(line) == 2:
                    src, dest = line
                    src_v, dest_v = id_mapping[src], id_mapping[dest]
                    graph.add_edge(src_v, dest_v)
                elif len(line) == 1:
                    node = line[0]
                    _ = id_mapping[node]
        for orig_id, v in id_mapping.iteritems():
            graph.vertex_properties['NodeId'][v] = orig_id
        if not parallel_edges:
            gt.remove_parallel_edges(graph)
        return graph

    def read_edge_list_nx(self, filename, directed=True, parallel_edges=False):
        if nx is None:
            import networkx as nx
        if directed:
            if parallel_edges:
                graph = nx.MultiDiGraph()
            else:
                graph = nx.DiGraph()
        else:
            graph = nx.Graph()
        with io.open(filename, encoding='utf-8') as infile:
            for line in infile:
                line = line.strip().split()
                if len(line) == 2:
                    src, dest = line
                    graph.add_edge(src, dest)
                elif len(line) == 1:
                    graph.add_node(line)
        return graph

    def extract_plaintext(self):
        class MLStripper(HTMLParser.HTMLParser):
            def __init__(self):
                HTMLParser.HTMLParser.__init__(self)
                self.reset()
                self.fed = []

            def handle_data(self, dat):
                self.fed.append(dat)

            def get_data(self):
                return ''.join(self.fed)

            def reset(self):
                self.fed = []
                HTMLParser.HTMLParser.reset(self)

        parser = MLStripper()
        if not os.path.exists(self.plaintext_folder):
            os.makedirs(self.plaintext_folder)
        files = set(os.listdir(self.plaintext_folder))
        file_last = sorted(files)[-1] if files else None
        for i, a in enumerate(sorted(self.name2id.keys())):
            ofname = self.plaintext_folder + a + '.txt'
            if a + '.txt' in files:
                if a + '.txt' != file_last:
                    continue
                else:
                    print(a + '.txt', 'overwrite')
            print(unicode(i+1) + '/' + unicode(len(self.name2id)) +
                  ' ' + a + '.txt')
            fname = os.path.join(self.html_base_folder + a[0].lower(),
                                 a + '.htm')
            with io.open(fname, encoding='utf-8') as infile:
                data = infile.read()
            data = data.split('<!-- start content -->')[1]
            data = data.split('<div class="printfooter">')[0]
            data = [d.strip() for d in data.splitlines()]
            data = [d for d in data if d]
            data[0] = data[0].split('</p></div></div>')[1]
            text = []
            for d in data:
                parser.reset()
                parser.feed(parser.unescape(d))
                stripped_d = parser.get_data()
                if stripped_d:
                    text.append(stripped_d)
            text = '\n'.join(text)
            with io.open(ofname, 'w', encoding='utf-8') as outfile:
                outfile.write(text)

    @decorators.Cached
    def get_tfidf_similarity(self, start, target):
        if start > target:
            start, target = target, start

        query = '''SELECT similarity FROM tfidf_similarities
                   WHERE page_id=%d AND target_page_id=%d''' % (start, target)
        similarity = self.db_connector.execute(query)
        return similarity[0]['similarity']

    @decorators.Cached
    def get_category_depth(self, node):
        query = '''SELECT category_depth FROM node_data
                   WHERE node_id=%d''' % node
        depth = self.db_connector.execute(query)
        return depth[0]['category_depth']

    @decorators.Cached
    def get_category_distance(self, start, target):
        if start > target:
            start, target = target, start

        query = '''SELECT distance FROM category_distances
                   WHERE page_id=%d AND target_page_id=%d''' % (start, target)
        distance = self.db_connector.execute(query)
        return distance[0]['distance']

    @decorators.Cached
    def get_spl(self, start, target):
        """ get the shortest path length for two nodes from the database
        if this is too slow, add an index to the table as follows:
        ALTER TABLE path_lengths ADD INDEX page_id (page_id);
        """
        query = '''SELECT path_length FROM path_lengths
                   WHERE page_id=%d AND target_page_id=%d''' % (start, target)
        length = self.db_connector.execute(query)
        if not length:
            return np.NaN
        else:
            return length[0]['path_length']

    def compute_link_positions(self):
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
        link_regex = re.compile(('(<a (class="mw-redirect" )?'
                                 'href="../../wp/[^/]+/(.+?)\.htm" '
                                 'title="[^"]+">.+?</a>)'))
        folder = os.path.join('data', self.label, 'wpcd', 'wp')
        link2pos_first, link2pos_last, pos2link, pos2linklength = {}, {}, {}, {}
        length, ib_length, lead_length = {}, {}, {}
        for i, a in enumerate(self.name2id.keys()):
            print(unicode(i+1), '/', unicode(len(self.name2id)), end='\r')
            lpos_first, lpos_last, posl, posll = {}, {}, {}, {}
            fname = os.path.join(folder, a[0].lower(), a + '.htm')
            try:
                with io.open(fname, encoding='utf-8') as infile:
                    data = infile.read()
            except UnicodeDecodeError:
                # there exist decoding errors for a few irrelevant pages
                print(fname)
                continue
            data = data.split('<!-- start content -->')[1]
            data = data.split('<div class="printfooter">')[0]
            if self.label == WIKTI.label:
                # skip a boilerplate line
                data = data.split('\n', 2)[2].strip()

            regex_results = link_regex.findall(data)
            regex_results = [(r[0], r[2]) for r in regex_results]
            for link in regex_results:
                link = [l for l in link if l]
                data = data.replace(link[0], ' [['+link[1]+']] ')

            # find infobox
            if '<table' in data[:100]:
                idx = data.find('</table>')
                data = data[:idx] + ' [[[ENDIB]]] ' + data[idx:]
            else:
                data = ' [[[ENDIB]]] ' + data

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

            idx = words.index('[[[ENDIB]]]')
            ib_length[a] = idx
            del words[idx]

            for wi, word in enumerate(reversed(words)):
                if word.startswith('[['):
                    try:
                        aid = self.name2id[word[2:-2].replace('%25', '%')]
                        lpos_first[aid] = len(words) - wi - 1
                    except KeyError:
                        pass
            for wi, word in enumerate(words):
                if word.startswith('[['):
                    try:
                        aid = self.name2id[word[2:-2].replace('%25', '%')]
                        lpos_last[aid] = wi
                        posl[wi] = aid
                        posll[wi] = word.count('_') + 1
                    except KeyError:
                        pass
            link2pos_first[a] = lpos_first
            link2pos_last[a] = lpos_last
            pos2link[a] = posl
            pos2linklength[a] = posll
            length[a] = len(words)
        path = os.path.join('data', self.label, 'link_positions.obj')
        with open(path, 'wb') as outfile:
            pickle.dump([link2pos_first, link2pos_last, length, pos2link,
                         pos2linklength, ib_length, lead_length], outfile, -1)

    def load_link_positions(self):
        if not self.link2pos_first:
            path = os.path.join('data', self.label, 'link_positions.obj')
            with open(path, 'rb') as infile:
                self.link2pos_first, self.link2pos_last, self.length,\
                    self.pos2link, self.pos2linklength, self.ib_length,\
                    self.lead_length = pickle.load(infile)
            self.link_sets = {k: sorted(v.keys())
                              for k, v in self.pos2link.iteritems()}

    @decorators.Cached
    def get_link_possibilities(self, start, target):
        return sorted([k for k, v in self.pos2link[start].items()
                       if v == self.name2id[target]])

    @decorators.Cached
    def get_link_context_count(self, start, pos):
        if np.isnan(pos):
                return np.NaN
        else:
            ctxt = sum(pos - 10 <= l <= pos + 10
                       for l in self.link_sets[start])
            return ctxt

    @decorators.Cached
    def get_link_length(self, start, pos):
        return self.pos2linklength[start][pos]

    def load_data(self, force=False):
        if force or self.data is None:
            path = os.path.join('data', self.label, 'data.obj')
            self.data = pd.read_pickle(path)

    def save_data(self, data=None):
        if data is None:
            data = self.data
        data.to_pickle(os.path.join('data', self.label, 'data.obj'))

    def print_error(self, message):
        print('        Error:', message)

    @staticmethod
    def apply_debug(df, func, label):
        result = []
        for i, d in enumerate(df.iterrows()):
            if (i % 100) == 0:
                print('       ', i, '/', df.shape[0], end='\r')
            result.append(func(d[1]))
        df[label] = result

    def complete_dataframe(self):
        print('complete_dataframe()')
        self.load_data()
        self.load_link_positions()
        df = self.data

        df['node_id'] = df['node'].apply(lambda n: self.name2id[n])
        df['node_next_id'] = df['node_next'].apply(lambda n: self.name2id[n] if isinstance(n, unicode) else np.NaN)
        df['degree_out'] = df['node_id'].apply(lambda n: self.id2deg_out[n])
        df['degree_in'] = df['node_id'].apply(lambda n: self.id2deg_in[n])

        print('     getting ngram frequencies...')
        get_ngrams = lambda n: ngram.ngram_frequency.get_frequency(n)
        df['ngram'] = df['node'].apply(get_ngrams)
        df['ngram'] = df['ngram'].apply(np.exp)

        print('     getting Wikipedia view counts...')
        get_view_count = lambda n: viewcounts.viewcount.get_frequency(n['node'])
        Wikigame.apply_debug(df, get_view_count, 'view_count')

        print('     getting word counts and shortest paths...')
        df['word_count'] = df['node'].apply(lambda n: self.length[n])
        # spl_target = lambda d: self.get_spl(d['node_id'], d['target_id'])
        # Wikigame.apply_debug(df, spl_target, 'spl_target')

        print('     getting TF-IDF similarities...')
        tidf_target = lambda d: 1 - self.get_tfidf_similarity(d['node_id'],
                                                              d['target_id'])
        Wikigame.apply_debug(df, tidf_target, 'tfidf_target')

        # print('     getting category statistics...')
        # category_depth = lambda n: self.get_category_depth(n)
        # # df['category_depth'] = df['node_id'].apply(category_depth)
        # Wikigame.apply_debug(df, category_depth, 'category_depth')
        #
        # category_target = lambda d: self.get_category_distance(d['node_id'],
        #                                                        d['target_id'])
        # # df['category_target'] = df.apply(category_target, axis=1)
        # Wikigame.apply_debug(df, category_target, 'category_target')

        print('     getting link positions...')
        link2pos_all = {}
        for article, article_dict in self.pos2link.iteritems():
            link2pos = collections.defaultdict(list)
            for k, v in article_dict.iteritems():
                link2pos[v].append(k)
            link2pos_all[article] = link2pos
        first, last, links_all = [], [], []
        for i in range(df.shape[0] - 1):
            print('         ', i+1, '/', df.shape[0], end='\r')
            if df.iloc[i]['subject'] != df.iloc[i+1]['subject'] or\
                    df.iloc[i]['backtrack']:
                    # if data belongs to different missions or is a backtrack
                first.append(np.NaN)
                last.append(np.NaN)
                links_all.append(np.NaN)
            else:
                a = df.iloc[i]['node']
                b = df.iloc[i+1]['node_id']
                first.append(self.link2pos_first[a][b])
                last.append(self.link2pos_last[a][b])
                links_all.append(link2pos_all[a][b])
        first.append(np.NaN)
        last.append(np.NaN)
        links_all.append(np.NaN)
        df['linkpos_first'] = first
        df['linkpos_last'] = last
        df['linkpos_all'] = links_all

        # find out whether the linkposition was in the infobx or the lead
        print('     getting infobox and lead positions...')
        lp = 'linkpos_actual' if 'linkpos_actual' in df else 'linkpos_first'
        ibs = [self.ib_length[d] for d in df['node']]
        leads = [self.lead_length[d] for d in df['node']]
        linkpos_ib, linkpos_lead = [], []
        for p, i, l in zip(df[lp], ibs, leads):
            print('         ', i+1, '/', df.shape[0], end='\r')
            if np.isnan(p):
                linkpos_ib.append(np.NaN)
                linkpos_lead.append(np.NaN)
            elif np.isnan(i):
                linkpos_ib.append(np.NaN)
                linkpos_lead.append(int(p < l))
            else:
                linkpos_ib.append(int(p < i))
                linkpos_lead.append(int(i < p < l))
        df['linkpos_ib'] = linkpos_ib
        df['linkpos_lead'] = linkpos_lead

        self.data = df
        self.save_data()

    def add_link_context(self):
        print('add_link_context()')
        self.load_data()
        self.load_link_positions()

        df = self.data
        lp = 'linkpos_actual' if 'linkpos_actual' in df else 'linkpos_first'

        # get link context
        context = []
        anchor_length = []
        for i in range(df.shape[0] - 1):
            print('         ', i+1, '/', df.shape[0], end='\r')
            if (df.iloc[i]['subject'] != df.iloc[i+1]['subject']) or\
                    df.iloc[i]['backtrack']:
                    # if data belongs to different missions or is a backtrack
                context.append(np.NaN)
                anchor_length.append(np.NaN)
            else:
                a = df.iloc[i]['node']
                b = df.iloc[i][lp]
                context.append(self.get_link_context_count(a, b))
                anchor_length.append(self.get_link_length(a, b))
                # print(a, df.iloc[i+1]['node'], b, self.get_link_length(a, b))
                # pdb.set_trace()

        df['link_context'] = context + [np.NaN]
        df['link_anchor_length'] = anchor_length + [np.NaN]
        self.save_data()

    def add_means(self):
        print('add_means()')
        self.load_data()
        self.data['mission'] = self.data['start'] + '-' + self.data['target']
        df = self.data

        df = df.join(df.groupby('mission')['pl'].mean(), on='mission',
                     rsuffix='_mission_mean')
        mission_mean_mean = df['pl_mission_mean'].mean()
        df['above_pl_mission_mean'] = df['pl_mission_mean'] > mission_mean_mean

        df = df.join(df.groupby('user')['pl'].mean(), on='user',
                     rsuffix='_user_mean')
        user_mean_mean = df['pl_user_mean'].mean()
        df['above_pl_user_mean'] = df['pl_user_mean'] > user_mean_mean
        self.data = df
        self.save_data()

    def add_all_linkpos(self):
        self.load_link_positions()
        link2pos_all = {}
        for article, article_dict in self.pos2link.iteritems():
            link2pos = collections.defaultdict(list)
            for k, v in article_dict.iteritems():
                link2pos[v].append(k)
            link2pos_all[article] = link2pos
        self.load_data()
        df = self.data
        links_all = []
        for i in range(df.shape[0] - 1):
            print('   ', i+1, '/', df.shape[0], end='\r')
            if df.iloc[i]['subject'] != df.iloc[i+1]['subject'] or\
                    df.iloc[i]['backtrack']:
                    # if data belongs to different missions or is a backtrack
                links_all.append(np.NaN)
            else:
                a = df.iloc[i]['node']
                b = df.iloc[i+1]['node_id']
                links_all.append(link2pos_all[a][b])
        links_all.append(np.NaN)
        df['linkpos_all'] = links_all
        self.data = df
        self.save_data()

    def create_correlation_data(self):
        node_id = sorted(self.id2name.keys())
        df = pd.DataFrame(data=node_id, columns=['node_id'])
        df['node'] = [self.id2name[n] for n in node_id]
        df['degree_in'] = [self.id2deg_in[n] for n in node_id]
        df['degree_out'] = [self.id2deg_out[n] for n in node_id]

        print('    getting category statistics...')
        category_depth = lambda n: self.get_category_depth(n)
        df['category_depth'] = df['node_id'].apply(category_depth)

        # print('    getting ngram frequencies...')
        # ngrams = []
        # for i, n in enumerate(df['node']):
        #     print('   ', i+1, '/', len(df['node']), end='\r')
        #     ngrams.append(ngram.ngram_frequency.get_frequency(n))
        # df['ngram'] = ngrams

        print('    getting view counts...')
        view_counts = []
        for i, n in enumerate(df['node']):
            print('   ', i+1, '/', len(df['node']), end='\r')
            view_counts.append(viewcounts.viewcount.get_frequency(n))
        df['view_count'] = view_counts

        # df.to_pickle(os.path.join('data', self.label, 'data_correlation.obj'))

    def lead_links(self):
        self.load_link_positions()
        ib_links, lead_links, ib_lead_links, rest_links = [], [], [], []
        for i, node in enumerate(self.name2id.keys()):
            print(i+1, '/', len(self.name2id), end='\r')
            try:
                ib_limit = self.ib_length[node]
                lead_limit = self.lead_length[node]
            except KeyError:
                continue
            for pos, link in self.pos2link[node].items():
                if pos < ib_limit:
                    ib_links.append(link)
                    ib_lead_links.append(link)
                elif ib_limit < pos < lead_limit:
                    lead_links.append(link)
                    ib_lead_links.append(link)
                else:
                    rest_links.append(link)

        for stat_label, stat_func in [
            ('mean', np.mean),
            ('median', np.median)
        ]:
            print(stat_label)
            for label, func in [
                ('degree_in', lambda x: self.id2deg_in[x]),
                ('ngram', lambda x: np.exp(ngram.ngram_frequency.get_frequency(self.id2name[x]))),
                ('view_count', lambda x: viewcounts.viewcount.get_frequency(self.id2name[x])),
            ]:
                ib = stat_func([func(l) for l in ib_links])
                lead = stat_func([func(l) for l in lead_links])
                ib_lead = stat_func([func(l) for l in ib_lead_links])
                rest = stat_func([func(l) for l in rest_links])
                if label == 'ngram':
                    ib, lead = np.log(ib), np.log(lead)
                    ib_lead, rest = np.log(ib_lead), np.log(rest)

                print('    %s:' % label)
                print('        %.4f IB' % ib)
                print('        %.4f Lead' % lead)
                print('        %.4f IB & Lead' % ib_lead)
                print('        %.4f Rest' % rest)

    def lead_links_all(self):
        self.load_link_positions()
        links, ibs, leads = [], [], []
        for i, node in enumerate(self.name2id.keys()):
            print(i+1, '/', len(self.name2id), end='\r')
            try:
                ib_limit = self.ib_length[node]
                lead_limit = self.lead_length[node]
            except KeyError:
                continue
            for pos, link in self.pos2link[node].items():
                links.append(link)
                ibs.append(0)
                leads.append(0)
                if pos < ib_limit:
                    ibs[-1] = 1
                elif ib_limit < pos < lead_limit:
                    leads[-1] = 1

        df = pd.DataFrame(data=zip(links, ibs, leads), columns=['target', 'ib', 'lead'])
        for label, func in [
            ('degree_in', lambda x: self.id2deg_in[x]),
            ('ngram', lambda x: np.exp(ngram.ngram_frequency.get_frequency(self.id2name[x]))),
            ('view_count', lambda x: viewcounts.viewcount.get_frequency(self.id2name[x])),
        ]:
            print(label)
            df[label] = df['target'].apply(func)
        df.to_pickle('data/data_ib_lead_rest_wikispeedia.obj')

    def debug(self):
        self.load_link_positions()
        for i, node in enumerate(self.name2id.keys()):
            ib_links, lead_links, ib_lead_links, rest_links = [], [], [], []
            print(node)
            try:
                ib_limit = self.ib_length[node]
                lead_limit = self.lead_length[node]
            except KeyError:
                print('KeyError')
                continue
            for pos, link in self.pos2link[node].items():
                if pos < ib_limit:
                    ib_links.append(link)
                    ib_lead_links.append(link)
                elif ib_limit < pos < lead_limit:
                    lead_links.append(link)
                    ib_lead_links.append(link)
                else:
                    rest_links.append(link)

            for stat_label, stat_func in [
                ('mean', np.mean),
                ('median', np.median)
            ]:
                print(stat_label)
                for label, func in [
                    ('degree_in', lambda x: self.id2deg_in[x]),
                    ('ngram', lambda x: np.exp(ngram.ngram_frequency.get_frequency(self.id2name[x]))),
                    ('view_count', lambda x: viewcounts.viewcount.get_frequency(self.id2name[x])),
                ]:
                    ib = stat_func([func(l) for l in ib_links])
                    lead = stat_func([func(l) for l in lead_links])
                    ib_lead = stat_func([func(l) for l in ib_lead_links])
                    rest = stat_func([func(l) for l in rest_links])
                    if label == 'ngram':
                        ib, lead = np.log(ib), np.log(lead)
                        ib_lead, rest =  np.log(ib_lead), np.log(rest)

                    print('    %s:' % label)
                    print('        %.4f IB' % ib)
                    print('        %.4f Lead' % lead)
                    # print('        %.4f IB & Lead' % ib_lead)
                    print('        %.4f Rest' % rest)
            pdb.set_trace()

    def compare_models_stepwise(self):
        self.load_data()
        self.data = self.data[~self.data['backtrack']]
        self.data = self.data[self.data['spl'] == 3]  # TODO
        self.data = self.data[self.data['successful']]
        self.load_link_positions()
        df_result = pd.DataFrame(columns=['df', 'pl', 'step', 'model', 'kld'])
        for label, df_full in [
            ('all', self.data),
            # ('usa', self.data[self.data['usa']]),
            ('no usa', self.data[~self.data['usa']]),
        ]:
            print('+++++++++++++++++', label, '+++++++++++++++++')
            for pl in [
                4,
                5,
                6,
                7
            ]:
                df = df_full[df_full['pl'] == pl]
                print('----------------PATH LENGTH', pl, '----------------')
                for step in range(pl-1):
                    print('\n--------', step, '--------')
                    # df = df[df['node'] == 'Africa']
                    first = df[df['step'] == step]['node_id']
                    pos = df[df['step'] == step]['linkpos_first']
                    second = df[df['step'] == step]['node_next_id']
                    gm = model.GroundTruthModel(first, pos, second, self)
                    gm.compute()
                    mdls = []
                    for mdl in [
                        model.UniformModel,
                        model.DegreeModel,
                        model.ViewCountModel,
                        model.NgramModel,
                        model.CategoryModel,
                        model.TfidfModel,
                        model.LinkPosModel,
                        ## model.LinkPosDegreeModel,
                        ## model.LinkPosNgramModel,
                        ## model.LinkPosViewCountModel,
                    ]:
                        mdls.append(mdl(first, pos, self))

                    # compare models
                    results = {}
                    for n in mdls:
                        n.compute()
                        kld = gm.get_kld(n)
                        results[n.label] = kld
                        idx = df_result.index.shape[0]
                        df_result.loc[idx] = [label, pl, step, n.label, kld]
                    for r in sorted(results.items(), key=operator.itemgetter(1)):
                        print('%.2f\t%s' % (r[1], r[0]))
        df_result.to_pickle(os.path.join('data', self.label, 'models.obj'))

    def compare_mi(self):
        self.load_data()
        self.load_link_positions()
        step = 0
        df = self.data
        degs = df[df['step'] == step+1]['degree_in']
        ngrams = df[df['step'] == step+1]['ngram']
        nodes = df[df['step'] == step+1]['node_id']

        ct = np.histogram2d(degs, ngrams)[0]
        mi = mutual_info_score(None, None, ct) / max(entropy(degs), entropy(ngrams))
        print('degs ngrams %.4f' % mi)

        ct = np.histogram2d(degs, nodes)[0]
        mi = mutual_info_score(None, None, ct) / max(entropy(degs), entropy(nodes))
        print('degs nodes %.4f' % mi)

        ct = np.histogram2d(ngrams, nodes)[0]
        mi = mutual_info_score(None, None, ct) / max(entropy(ngrams), entropy(nodes))
        print('ngrams nodes %.4f' % mi)

        # degs = np.histogram(degs, bins=100, density=True)[0]
        # ngrams = np.histogram(ngrams, bins=100, density=True)[0]
        # nodes = np.histogram(nodes, bins=100, density=True)[0]

        def je(a, b):
            ct = np.histogram2d(a, b)[0]
            return entropy(ct.flatten())

        def je3(a, b, c):
            ct = np.histogramdd([a, b, c])[0]
            return entropy(ct.flatten())

        x = degs
        y = ngrams
        z = nodes
        cmi = je(x, z) + je(y, z) - je3(x, y, z) - entropy(np.histogram(z)[0])
        print('degs ngrams nodes %.4f' % cmi)
        pdb.set_trace()

    # @decorators.Cached
    def get_source2target(self, kind, step=None, spl=None, pl=None,
                          no_usa=False):
        self.load_data(force=True)
        df = self.data
        if no_usa:
            self.data = self.data[~self.data['usa']]
        df['target'] = df['node'].shift(-1)
        df['target_id'] = df['node_id'].shift(-1)
        df = df[~df['backtrack']]
        df = df.dropna()
        if kind == 'successful':
            df = df[df['successful']]
        elif kind == 'unsuccessful':
            df = df[~df['successful']]
        elif kind == 'successful_first':
            df = df[(df['successful']) & (df['step'] == 0)]
        elif kind == 'successful_middle':
            df = df[(df['successful']) & (df['step'] != 0) &
                    (df['distance-to-go'] != 1)]
        elif kind == 'successful_last':
            df = df[(df['successful']) & (df['distance-to-go'] == 1)]
        elif kind == 'successful_first_limited':
            df = df[(df['successful']) & (df['step'] == 0) & (df['spl'] <= 5)]
        elif kind == 'successful_middle_limited':
            df = df[(df['successful']) & (df['step'] != 0) & (df['distance-to-go'] != 1) & (df['spl'] <= 5)]
        elif kind == 'successful_last_limited':
            df = df[(df['successful']) & (df['distance-to-go'] == 1) & (df['spl'] <= 5)]
        elif kind == 'successful_first_limited_pl':
            df = df[(df['successful']) & (df['step'] == 0) & (df['spl'] <= 5) & (df['pl'] <= 10)]
        elif kind == 'successful_middle_limited_pl':
            df = df[(df['successful']) & (df['step'] != 0) & (df['distance-to-go'] != 1) & (df['spl'] <= 5) & (df['pl'] <= 10)]
        elif kind == 'successful_last_limited_pl':
            df = df[(df['successful']) & (df['distance-to-go'] == 1) & (df['spl'] <= 5) & (df['pl'] <= 10)]
        elif kind == 'successful_high_deg_targets':
            df2 = df[(df['successful']) & (df['step'] == 0)]
            df2['target_deg_in'] = df2['target_id'].apply(lambda x: self.id2deg_in[x])
            targets = set(df2[df2['target_deg_in'] >= 100]['target'])
            df = df[(df['successful']) & (df['target'].isin(targets))]
        elif kind == 'successful_low_deg_targets':
            df2 = df[(df['successful']) & (df['step'] == 0)]
            df2['target_deg_in'] = df2['target_id'].apply(lambda x: self.id2deg_in[x])
            pdb.set_trace()
            targets = set(df2[df2['target_deg_in'] < 100]['target'])
            df = df[(df['successful']) & (df['target'].isin(targets))]
        elif kind == 'successful_deg_above_median':
            df2 = df[(df['successful']) & (df['step'] == 0)]
            df2['target_deg_in'] = df2['target_id'].apply(lambda x: self.id2deg_in[x])
            targets = set(df2[df2['target_deg_in'] > 28]['target'])
            df = df[(df['successful']) & (df['target'].isin(targets))]
        elif kind == 'successful_deg_below_median':
            df2 = df[(df['successful']) & (df['step'] == 0)]
            df2['target_deg_in'] = df2['target_id'].apply(lambda x: self.id2deg_in[x])
            targets = set(df2[df2['target_deg_in'] <= 28]['target'])
            df = df[(df['successful']) & (df['target'].isin(targets))]

        if step is not None:
            df = df[df['step'] == step]
        if spl is not None:
            df = df[df['spl'] == spl]
        if pl is not None:
            df = df[df['pl'] == pl]
        df = df[['node', 'node_id', 'node_next', 'node_next_id']]
        df.columns = ['source', 'source_id', 'target', 'target_id']
        df['amount'] = df.groupby(['source', 'target']) \
            .transform('count')['source_id']
        df = df.drop_duplicates(subset=['source', 'target'])
        source2target = {}
        for source in set(df['source']):
            df_sub = df[df['source'] == source]
            zipped = zip(df_sub['target'], df_sub['amount'])
            source2target[source] = {self.name2id[k]: v for k, v in zipped}
        return source2target

    # @decorators.Cached
    def get_model_df(self, kind, step=None, spl=None, pl=None, no_usa=False):
        source2target = self.get_source2target(kind, step, spl, pl, no_usa)
        self.load_link_positions()
        results = []
        for idx, key in enumerate(sorted(self.length.keys())):
            print(idx+1, '/', len(self.length), key, end='\r')
            ib_length = self.ib_length[key]
            lead_length = self.lead_length[key]
            targets = sorted(set(self.link2pos_first[key].keys()))
            target2amount = {k: 0 for k in targets}
            try:
                target2amount.update(source2target[key])
            except KeyError:
                pass
            link2pos = collections.defaultdict(list)
            for pos, link in self.pos2link[key].iteritems():
                if link in targets:
                    link2pos[link].append(pos)
            link2pos = {k: sorted(v) for k, v in link2pos.iteritems()}
            if not link2pos:
                # no outlinks on page
                continue
            source = [key] * len(link2pos)
            source_id = [self.name2id[key]] * len(link2pos)
            linkpos_first = [[link2pos[t][0]] for t in targets]
            linkpos_last = [[link2pos[t][-1]] for t in targets]
            linkpos_ib = [
                [l for l in link2pos[t] if l < ib_length] for t in targets
            ]
            linkpos_lead = [
                [l for l in link2pos[t] if ib_length < l < lead_length]
                for t in targets
            ]
            linkpos_ib_lead = [
                [l for l in link2pos[t] if l < lead_length]
                for t in targets
            ]
            linkpos_not_ib = [
                [l for l in link2pos[t] if l > ib_length] for t in targets
            ]
            linkpos_not_lead = [
                [l for l in link2pos[t] if l > lead_length or l < ib_length]
                for t in targets
            ]
            linkpos_not_ib_lead = [
                [l for l in link2pos[t] if l > lead_length] for t in targets
            ]
            linkpos_all = [link2pos[t] for t in targets]
            word_count = [self.length[key]] * len(link2pos)
            target = [self.id2name[t] for t in targets]
            target_id = targets
            amount = [target2amount[t] for t in targets]
            df = pd.DataFrame(
                data=zip(source, source_id, linkpos_first, linkpos_last,
                         linkpos_ib, linkpos_lead, linkpos_ib_lead,
                         linkpos_not_ib, linkpos_not_lead, linkpos_not_ib_lead,
                         linkpos_all,
                         word_count, target, target_id, amount),
                columns=['source', 'source_id', 'linkpos_first', 'linkpos_last',
                         'linkpos_ib', 'linkpos_lead', 'linkpos_ib_lead',
                         'linkpos_not_ib',
                         'linkpos_not_lead', 'linkpos_not_ib_lead',
                         'linkpos_all',
                         'word_count', 'target', 'target_id', 'amount']
            )
            results.append(df)
        df = pd.concat(results)
        suffix = ''
        if step is not None:
            suffix += '_step_' + unicode(step)
        if spl is not None:
            suffix += '_spl_' + unicode(spl)
        if pl is not None:
            suffix += '_pl_' + unicode(pl)
        usa_suffix = '_no_usa' if no_usa else ''
        stepwise = 'stepwise/' if step is not None else ''
        df.to_pickle('data/clickmodels/' + stepwise + 'wikispeedia_' + kind +
                      usa_suffix + suffix + '.obj')

    def get_model_df_stats(self):
        def print_stats(data, label):
            print('max: %.2f, mean: %.2f, median: %.2f, (%s)' % (max(data), np.mean(data), np.median(data), label))

        # get statistics for all nodes
        print_stats(self.id2deg_in.values(), 'all nodes')

        self.load_data(force=True)
        df_full = self.data
        df_successful = df_full[df_full['successful']]
        df_unsuccessful = df_full[~df_full['successful']]
        for df, label in [
            (df_successful, 'successful games'),
            (df_unsuccessful, 'unsuccessful games'),
        ]:
            targets = df[df['step'] == 0]['target_id']
            target_indegs = [self.id2deg_in[t] for t in targets]
            print_stats(target_indegs, label)

        df2 = df_successful
        df2['target_deg_in'] = df2['target_id'].apply(lambda x: self.id2deg_in[x])
        pdb.set_trace()

    def get_stats(self):
        stats = {
            'deg_in':
                {n: self.id2deg_in[i] for i, n in self.id2name.iteritems()},
            'view_count':
                {n: viewcounts.viewcount.get_frequency(n)
                 for n in self.name2id},
            'ngram':
                {n: np.exp(ngram.ngram_frequency.get_frequency(n))
                 for n in self.name2id},
        }
        with open('data/clickmodels/wikispeedia_stats.obj', 'wb') as outfile:
            pickle.dump(stats, outfile, -1)


class WIKTI(Wikigame):
    label = 'wikti'

    def __init__(self, successful=True):
        super(WIKTI, self).__init__(WIKTI.label, successful)

    def fill_database(self):
        from modules.TfidfCalculator import TfidfCalculator
        from modules.CategoryCalculator import CategoryCalculator

        db_connector = DbConnector('wikti')

        tfidf_calculator = TfidfCalculator(db_connector, self.plaintext_folder)
        tfidf_calculator.run()

        cat_calculator = CategoryCalculator(db_connector, self.html_base_folder,
                                            self.label)
        cat_calculator.run()

    def create_dataframe(self):
        """compute the click data as a pandas frame"""
        print('creating dataframe...')
        if not self.successful:
            raise NotImplementedError

        def parse_node(node_string):
            m = re.findall(r'/([^/]*?)\.htm', node_string)
            return m[0].replace('%25', '%') if m else ''

        def parse_node_link(node_string):
            m = re.findall(r"offset': (\d+)", node_string)
            return int(m[0]) if m else np.NaN

        self.load_link_positions()
        prefix = "u'current_page': u'http://0.0.0.0/wikigame/wiki-schools/wp/"
        results = []
        folder_logs = os.path.join('data', self.label, 'logfiles')
        folders = ['U' + '%02d' % i for i in range(1, 11)]
        for folder in folders:
            print('\n', folder)

            files = sorted(os.listdir(os.path.join(folder_logs, folder)))
            files = [f for f in files if f.startswith('PLAIN')]
            mission2fname = {int(re.findall(r'PLAIN_\d+_(\d+)', m)[0]): m
                             for m in files}
            for mission in sorted(mission2fname.keys()):
                filename = mission2fname[mission]
                print('   ', filename)
                fname = os.path.join(folder_logs, folder, filename)
                df_full = pd.read_csv(fname, sep='\t', usecols=[1, 2, 3],
                                      encoding='utf-8',
                                      names=['time', 'action', 'node'])

                # perform sanity checks
                action_counts = df_full['action'].value_counts()
                if action_counts['GAME_STARTED'] > 1:
                    self.print_error('duplicated_game_start, dropping')
                    continue
                elif action_counts['load'] < 2:
                    self.print_error('game too short, dropping')
                    continue

                # ensure correct start and target nodes
                miss = re.findall(r'(PLAIN_[\d]+_[a-z0-9_\-]+)\.', filename)[0]
                start, target = self.game2start_target[miss]
                successful = df_full.iloc[-1]['action'] == 'GAME_COMPLETED'
                df = df_full[df_full['action'] == 'load'][['node']]
                df['node'] = df['node'].apply(parse_node)
                if not all(n in self.name2id for n in df['node']):
                    self.print_error('node outside article node set, dropping')
                    continue
                if not df.iloc[0]['node'] == start:
                    self.print_error('start node does not match mission start')
                    pdb.set_trace()
                if successful and not target == df.iloc[-1]['node']:
                    # insert the target if the load event is not present
                    # this is the case for some of the earlier log files
                    last = df_full[df_full['action'] == 'link_data'].iloc[-1]
                    last = parse_node(last['node'])
                    last_time = df_full.iloc[-2]['time']
                    last_node = prefix + last[0].lower() + '/' + last + ".htm'"
                    df_full.loc[df_full.index[-1] + 1] = [last_time, 'load',
                                                          last_node]
                    if last != target:
                        # in some cases, the target is entirely missing
                        target_node = prefix + target[0].lower() + '/' +\
                            target + ".htm'"
                        df_full.loc[df_full.index[-1] + 1] = [last_time, 'load',
                                                              target_node]

                # make sure that all load events are present
                # some are skipped in case of a fast click on the loading page
                df = df_full[(df_full['action'] == 'link_data') |
                             (df_full['action'] == 'load')]
                df['node_parsed'] = df['node'].apply(parse_node)
                df_new = []
                for i in range(df.shape[0] - 1):
                    df_new.append(df.iloc[i])
                    actions = df.iloc[i: i+2]['action'].tolist()
                    if actions == ['link_data', 'load']:
                        nodes = df.iloc[i: i+2]['node_parsed'].tolist()
                        if nodes[0] != nodes[1]:
                            node_load = prefix + nodes[0][0].lower() + '/' +\
                                nodes[0] + ".htm'",
                            load = pd.Series(data={
                                'action': 'load',
                                'time': df.iloc[i]['time'],
                                'node': node_load,
                                'node_parsed': nodes[0]
                            })
                            df_new.append(load)

                            nodes_1 = self.name2id[nodes[1]]
                            try:
                                # use the first link position in this case
                                # fast click --> first position likely
                                first = self.link2pos_first[nodes[0]][nodes_1]
                                link = pd.Series(data={
                                    'action': 'link_data',
                                    'time': df.iloc[i+1]['time'],
                                    'node': prefix + nodes[1][0].lower() + '/' +
                                            nodes[1] + ".htm', u'offset': " +
                                            str(first + 30),
                                    'node_parsed': nodes[1]
                                })
                                df_new.append(link)
                            except KeyError:
                                # no quick click but a quick backtrack
                                backtrack = pd.Series(data={
                                    'action': 'load',
                                    'time': df.iloc[i+1]['time'],
                                    'node': prefix + nodes[1][0].lower() + '/' +
                                            nodes[1] + ".htm'",
                                    'node_parsed': nodes[1]
                                })
                                df_new.append(backtrack)
                df_new.append(df.iloc[-1])
                df = pd.DataFrame(data=df_new)

                # get actual link position from logs
                link_data = []
                for i in range(df.shape[0]):
                    actions = df.iloc[i: i+2]['action'].tolist()
                    if actions == ['link_data', 'load']:
                        link_data.append(df.iloc[i]['node'])
                    elif actions == ['load', 'load']:
                        link_data.append('')
                    elif actions == ['link_data', 'link_data']:
                        if '#' not in df.iloc[i]['node']:
                            n = parse_node(df.iloc[i]['node'])
                            m = parse_node(df.iloc[i+1]['node'])
                            print('        Problem with link_data:', n, m)
                    elif actions == ['load', 'link_data']:
                        pass
                link_data = map(parse_node_link, link_data)
                df['node'] = df['node_parsed']
                df = df[df['action'] == 'load']
                df = df[['time', 'action', 'node']]
                df['backtrack'] = [True if np.isnan(l) else False
                                   for l in link_data] + [False]

                # correct the link position from the logs to our metric
                link_data_correct = []
                zipped = zip(df['node'], df['node'].iloc[1:], link_data)
                for s, t, pos in zipped:
                    if np.isnan(pos):
                        link_data_correct.append(np.NaN)
                    else:
                        links = self.get_link_possibilities(s, t)
                        if len(links) == 0:
                            link_data_correct.append(np.NaN)
                        elif len(links) == 1:
                            link_data_correct.append(links[0])
                        else:
                            pos = bisect.bisect(links, pos)
                            link_data_correct.append(links[pos - 1])
                link_data = link_data_correct
                df['linkpos_actual'] = link_data + [np.NaN]

                # make sure a link from the log actually exists in the articles
                for i in range(df.shape[0] - 1):
                    if df.iloc[i]['backtrack']:
                        continue
                    a = df.iloc[i]['node']
                    b = df.iloc[i+1]['node']
                    links = self.get_link_possibilities(a, b)
                    if len(links) == 0:
                        print('        link does not exist:', a, b)

                # get time information
                df['duration'] = df['time'].iloc[-1] / 1000
                time = df['time'].diff().shift(-1)
                time_normalized = time / sum(time.iloc[:-1])
                word_count = [self.length[a] if a in self.length else np.NaN
                              for a in df['node']]
                link_count = [len(self.pos2link[a])
                              if a in self.length else np.NaN
                              for a in df['node']]
                time_word = time / word_count
                time_link = time / link_count
                # divide by 1000 to convert to seconds
                df['time'] = time / 1000
                df['time_normalized'] = time_normalized
                df['time_word'] = time_word / 1000
                df['time_link'] = time_link / 1000

                # set overall dataframe attributes
                spl = self.get_spl(self.name2id[start], self.name2id[target])
                df['successful'] = successful
                df['spl'] = spl
                df['pl'] = df.shape[0]
                df['step'] = range(df.shape[0])
                df['distance-to-go'] = list(reversed(range(df.shape[0])))
                df['user'] = folder
                df['mission'] = mission
                df['subject'] = df['user'] + '_' + df['mission'].astype('str')
                df['start'] = start
                df['start_id'] = df['start'].apply(lambda n: self.name2id[n])
                df['target'] = target
                df['target_id'] = df['target'].apply(lambda n: self.name2id[n])
                results.append(df)
        data = pd.concat(results)
        self.save_data(data)


class Wikispeedia(Wikigame):
    label = 'wikispeedia'

    def __init__(self):
        super(Wikispeedia, self).__init__(Wikispeedia.label)

    @staticmethod
    def build_database(label):
        db_connector = DbConnector('')
        path = os.path.join('data', label, 'SQL', 'structure.sql')
        with io.open(path, encoding='utf-8') as infile:
            stmt = infile.read()
        db_connector.execute(stmt)
        db_connector.execute('USE ' + label + ';')
        db_connector.commit()

    def fill_database(self):
        from modules.PageExtractor import PageExtractor
        from modules.LinkExtractor import LinkExtractor
        from modules.LinkCleaner import LinkCleaner
        from modules.PathCalculator import PathCalculator
        from modules.NodeValues import NodeValues
        from modules.TfidfCalculator import TfidfCalculator
        from modules.CategoryCalculator import CategoryCalculator

        db_connector = DbConnector('wikispeedia')

        page_extractor = PageExtractor(db_connector)
        page_extractor.run()

        link_extractor = LinkExtractor(db_connector)
        link_extractor.run()

        link_cleaner = LinkCleaner(db_connector)
        link_cleaner.run()

        path_calculator = PathCalculator(db_connector)
        path_calculator.run()

        node_values = NodeValues(db_connector)
        node_values.run()

        tfidf_calculator = TfidfCalculator(db_connector, self.plaintext_folder)
        tfidf_calculator.run()

        cat_calculator = CategoryCalculator(db_connector, self.html_base_folder,
                                            self.label)
        cat_calculator.run()

    def create_dataframe(self, limit=None):
        results = []
        folder_logs = os.path.join('data', self.label, 'logfiles')
        self.load_link_positions()
        filenames = ['paths_finished.tsv', 'paths_unfinished.tsv']
        for filename in filenames:
            print('    ', filename)
            fname = os.path.join(folder_logs, filename)
            successful = False if 'unfinished' in filename else True
            df_full = pd.read_csv(fname, sep='\t', comment='#', index_col=False,
                                  encoding='utf-8', usecols=[0, 1, 2, 3],
                                  names=['user', 'timestamp', 'duration',
                                         'path'])
            df_full = df_full.iloc[:limit]  # for debug trials

            def convert_time(t):
                tm = datetime.datetime.fromtimestamp(t)
                return tm.strftime('%Y-%m-%d %H:%M:%S')

            df_full['timestamp'] = df_full['timestamp'].apply(convert_time)
            paths = df_full['path'].str.split(';').tolist()
            start = pd.DataFrame([t[0] for t in paths])
            df_full['start'] = start
            df_full['start_id'] = df_full['start'].apply(lambda n:
                                                         self.name2id[n])
            if successful:
                target = pd.DataFrame([t[-1] for t in paths])
            else:
                target = pd.read_csv(fname, sep='\t', comment='#',
                                     encoding='utf-8',
                                     usecols=[4], names=['target'])
            df_full['target'] = target
            df_full = df_full[df_full['start'].isin(self.name2id)]
            df_full = df_full[df_full['target'].isin(self.name2id)]
            df_full['target_id'] = df_full['target'].apply(lambda n:
                                                           self.name2id[n])
            spl = lambda d: self.get_spl(self.name2id[d['start']],
                                         d['target_id'])
            df_full['spl'] = df_full.apply(spl, axis=1)

            def resolve_backtracks(path):
                path = path.split(';')
                backtrack = [False] * len(path)
                if '<' not in path:
                    return path, backtrack

                path.reverse()
                path_resolved = [path[-1]]
                stack = [path.pop()]
                i = 0
                while path:
                    i += 1
                    p = path.pop()
                    if p == '<':
                        stack.pop()
                        backtrack[i-1] = True
                    else:
                        stack.append(p)
                    path_resolved.append(stack[-1])
                return path_resolved, backtrack

            result = map(resolve_backtracks, df_full['path'])
            path = [r[0] for r in result]
            backtrack = [r[1] for r in result]
            df_full['path'], df_full['backtrack'] = path, backtrack

            df_full['pl'] = df_full['path'].apply(lambda p: len(p))

            def nonexisting_links_present(dtfr):
                # make sure a link from the log actually exists in the articles
                for i in range(dtfr.shape[0] - 1):
                    if dtfr.iloc[i]['backtrack']:
                        continue
                    a = dtfr.iloc[i]['node']
                    b = dtfr.iloc[i+1]['node']
                    links = self.get_link_possibilities(a, b)
                    if len(links) == 0:
                        print('        link does not exist:', a, b)
                        return True
                return False

            for eid, entry in df_full.iterrows():
                print('    ', eid + 1, '/', df_full.shape[0], end='\r')
                data = zip(entry['path'], entry['path'][1:] + [np.NaN],
                           entry['backtrack'])
                columns = ['node', 'node_next', 'backtrack']
                df = pd.DataFrame(data=data, columns=columns)
                if nonexisting_links_present(df):
                    continue
                df['successful'] = successful
                df['spl'] = entry['spl']
                df['pl'] = entry['pl']
                df['step'] = range(df.shape[0])
                df['distance-to-go'] = list(reversed(range(df.shape[0])))
                df['subject'] = eid
                df['user'] = entry['user']
                df['usa'] = 'United_States' in entry['path']
                df['timestamp'] = entry['timestamp']
                df['duration'] = entry['duration']
                df['start'] = entry['start']
                df['start_id'] = entry['start_id']
                df['target'] = entry['target']
                df['target_id'] = entry['target_id']
                results.append(df)
        data = pd.concat(results)
        self.save_data(data)


def get_stats_wikipedia():
    # df = pd.read_pickle('data/clickmodels/wikipedia_all.obj')
    # targets = set(df['target'])
    # with open('data/clickmodels/wikipedia_targets.obj', 'wb') as outfile:
    #     pickle.dump(targets, outfile, -1)
    # sys.exit()
    with open('data/clickmodels/wikipedia_targets.obj', 'rb') as infile:
        targets = pickle.load(infile)
    with open('data/clickmodels/wikipedia_indegrees.obj', 'rb') as infile:
        name2deg_in = pickle.load(infile)
    with open('data/clickmodels/wikipedia_view_counts.obj', 'rb') as infile:
        name2vc = pickle.load(infile)
    with open('data/clickmodels/wikipedia_ngrams.obj', 'rb') as infile:
        name2ng = pickle.load(infile)
    stats = {
        'deg_in':
            {t: (name2deg_in[t] if t in name2deg_in else 0) for t in targets},
        'view_count':
            {t: (name2vc[t] if t in name2vc else 0) for t in targets},
        'ngram':
            {t: (name2ng[t] if t in name2ng else 0) for t in targets},
    }
    with open('data/clickmodels/wikipedia_stats.obj', 'wb') as outfile:
        pickle.dump(stats, outfile, -1)


def get_wikipedia_ngrams(start=None, end=None):
    print(start, end)
    with open('data/clickmodels/wikipedia_targets.obj', 'rb') as infile:
        targets = pickle.load(infile)
    targets = targets - {None}
    ngrams = {}
    for idx, t in enumerate(sorted(targets)[start:end]):
        print(idx+1, '/', len(targets), end='\r')
        ngrams[t] = ngram.ngram_frequency.get_frequency(t.encode('utf-8'))

    fname = 'data/clickmodels/wikipedia_ngrams_' + unicode(start) + '_' +\
            unicode(end) + '.obj'
    with open(fname, 'wb') as outfile:
        pickle.dump(ngrams, outfile, -1)


def combine_stats():
    # view_counts = {}
    # for suffix in [
    #     '0_100000',
    #     '100000_200000',
    #     '200000_300000',
    #     '300000_400000',
    #     '400000_500000',
    #     '500000_None',
    # ]:
    #     fname = 'data/clickmodels/wikipedia_view_counts_' + suffix + '.obj'
    #     with open(fname, 'rb') as infile:
    #         d = pickle.load(infile)
    #     view_counts.update(d)
    #
    # with open('data/clickmodels/wikipedia_view_counts.obj', 'wb') as outfile:
    #     pickle.dump(view_counts, outfile, -1)
    ngrams = {}
    for suffix in [
        '0_50000',
        '50000_100000',
        '100000_150000',
        '150000_200000',
        '200000_300000',
        '300000_350000',
        '350000_400000',
        '400000_500000',
        '500000_None',
    ]:
        fname = 'data/clickmodels/wikipedia_ngrams_' + suffix + '.obj'
        with open(fname, 'rb') as infile:
            d = pickle.load(infile)
        ngrams.update(d)

    with open('data/clickmodels/wikipedia_ngrams.obj', 'wb') as outfile:
        pickle.dump(ngrams, outfile, -1)


def get_wikipedia_view_counts(start=None, end=None):
    print(start, end)
    with open('data/clickmodels/wikipedia_targets.obj', 'rb') as infile:
        targets = pickle.load(infile)
    targets = targets - {None}
    view_counts = {}
    for idx, t in enumerate(sorted(targets)[start:end]):
        print(idx+1, '/', len(targets), end='\r')
        view_counts[t] = viewcounts.viewcount.get_frequency(t.encode('utf-8'))

    fname = 'data/clickmodels/wikipedia_view_counts_' + unicode(start) + '_' +\
            unicode(end) + '.obj'
    with open(fname, 'wb') as outfile:
        pickle.dump(view_counts, outfile, -1)


if __name__ == '__main__':

    # get_wikipedia_ngrams(350000, 400000)
    # get_wikipedia_view_counts(500000, None)
    # combine_stats()
    # get_stats_wikipedia()

    for wg in [
        # WIKTI(successful=True),
        Wikispeedia()
    ]:
        # wg.compute_link_positions()
        # wg.create_correlation_data()
        # wg.create_dataframe(limit=None)
        # wg.complete_dataframe()
        # wg.add_means()
        # pdb.set_trace()

        # wg.compare_models_lead()
        # wg.compare_models_stepwise()
        # wg.compare_models_first()
        # wg.compare_mi()
        # wg.get_stats()
        # wg.lead_links()
        # wg.lead_links_all()
        # wg.debug()

        # wg.get_model_df_stats()
        # wg.get_model_df('all')
        # wg.get_model_df('successful')
        # wg.get_model_df('successful_first')
        # wg.get_model_df('successful_middle')
        # wg.get_model_df('successful_last')
        # wg.get_model_df('successful_first_limited')
        # wg.get_model_df('successful_middle_limited')
        # wg.get_model_df('successful_last_limited')
        # wg.get_model_df('successful_first_limited_pl')
        # wg.get_model_df('successful_middle_limited_pl')
        # wg.get_model_df('successful_last_limited_pl')
        # wg.get_model_df('unsuccessful')

        for spl in [
            # 3,
            # 4,
            5,
        ]:
            print('spl=%d' % spl)
            for pl in range(spl+1, 11):
                print('    pl=%d' % pl)
                for step in range(pl):
                    print('        step=%d' % step)
                    # wg.get_model_df('successful', step=step, spl=spl, pl=pl)
                    # wg.get_model_df('successful_deg_above_median', step=step, spl=spl, pl=pl)
                    wg.get_model_df('successful_deg_below_median', step=step, spl=spl, pl=pl)
