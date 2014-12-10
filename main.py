# -*- coding: utf-8 -*-

from __future__ import division, print_function
from collections import defaultdict
import cPickle as pickle
import HTMLParser
import io
import os
import pdb
import re
import urllib2

# import graph_tool.all as gt
import networkx as nx
import numpy as np
import pandas as pd
import pymysql
import statsmodels.api as sm


class DbConnector(object):
    def __init__(self):
        self.db_host = '127.0.0.1'
        self.db_connection = pymysql.connect(host=self.db_host,
                                             port=3306,
                                             user='root',
                                             passwd='',
                                             db='wikigame',
                                             charset='utf8')
        self.db_cursor = self.db_connection.cursor(pymysql.cursors.DictCursor)
        self.db_cursor_nobuff = self.db_connection.cursor(
            pymysql.cursors.SSCursor)

    def __exit__(self):
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

    # ------------------------ NON BUFFERED STUFF

    def fetch_cursor_nobuff(self, _statement, _args):
        self.db_cursor_nobuff.execute(_statement, _args)
        return self.db_cursor_nobuff


class NgramConnector(object):
    def __init__(self):
        token = 'bec1748d-5665-4266-83fb-e657ef4070ea'
        corpus = 'bing-body/2013-12/5/'
        base_url = 'http://weblm.research.microsoft.com/weblm/rest.svc/'
        self.base_url = base_url + corpus + 'jp?u=' + token + '&p='

    def get(self, word):
        word = word.replace(' ', '+').replace('_', '+')
        return float(urllib2.urlopen(self.base_url + word).read())

    def get_ngrams(self):
        db_connector = DbConnector()
        query = '''CREATE TABLE IF NOT EXISTS `ngrams` (
                       `node_id` int(11) NOT NULL,
                       `probability` float NOT NULL,
                        PRIMARY KEY (`node_id`)
                        ) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_bin;
                       '''
        db_connector.execute(query)
        db_connector.commit()
        pages = db_connector.execute('SELECT * FROM pages')
        page_ids = set(p['id'] for p in pages)
        id2name = {p['id']:
                   re.findall(r'\\([^\\]*?)\.htm', p['link'])[0] for p in pages}
        ids = db_connector.execute('SELECT id FROM ngrams')
        ngram_ids = set(p['id'] for p in ids)
        ids = sorted(page_ids - ngram_ids)
        ngram_connector = NgramConnector()
        for i in ids:
            print(i, '/', len(ids))
            probability = ngram_connector.get(id2name[i])
            stmt = '''INSERT INTO `ngrams` (node_id, probability)
                      values (%s, %s)''' % (i, probability)
            db_connector.execute(stmt)
            db_connector.commit()


class LogisticRegressor(object):
    def __init__(self):
        pass

    def regress(self, dependent, independent):
        logit = sm.Logit(dependent, independent)
        result = logit.fit()
        print(result.summary())

        # import matplotlib.pyplot as plt
        # plt.scatter(clicks['time'], clicks['success'])
        # plt.xlabel('duration')
        # plt.ylabel('success')
        # x = np.arange(0, 1000000, 100)
        # b1, b0 = result.params.values
        # p = [(np.exp(b0 + i*b1))/(1+np.exp(b0 + i*b1)) for i in x]
        # plt.plot(x, p)
        # plt.show()
        # pdb.set_trace()

        pred = result.predict(independent)
        correct = 0
        for t, p in zip(dependent, pred):
            if t == 0 and p < 0.5:
                correct += 1
            elif t == 1 and p > 0.5:
                correct += 1
        print(correct, 'correct out of', len(pred), '(', correct/len(pred), ')')
        print('Baseline:', sum(dependent) / dependent.shape[0])

    def test_regression(self):
        fname = 'data/SAheart.txt'
        df = pd.read_csv(fname, index_col=0,
                         usecols=[0, 1, 2, 3, 5, 7, 8, 9, 10])
        dummies = pd.get_dummies(df['famhist'], prefix='famhist')
        cols_to_keep = ['sbp', 'tobacco', 'ldl', 'obesity', 'alcohol', 'age',
                        'chd']
        df = df[cols_to_keep].join(dummies.ix[:, 1:])
        df['intercept'] = 1.0
        regressor = LogisticRegressor()
        regressor.regress(df['chd'],
                          df[['intercept', 'sbp', 'tobacco', 'ldl', 'obesity',
                              'famhist_Present', 'alcohol', 'age']])
        regressor.regress(df['chd'],
                          df[['intercept', 'tobacco', 'ldl', 'famhist_Present',
                              'age']])


class Network(object):
    def __init__(self, filename, graph_tool=False):
        # read the graph
        if graph_tool:
            self.graph = self.read_edge_list_gt(filename)
        else:
            self.graph = self.read_edge_list_nx(filename)

        # build some mappings from the database
        db_connector = DbConnector()
        pages = db_connector.execute('SELECT * FROM pages')
        self.id2title = {p['id']: p['name'] for p in pages}
        self.id2name = {p['id']:
                   re.findall(r'\\([^\\]*?)\.htm', p['link'])[0] for p in pages}
        self.name2id = {v: k for k, v in self.id2name.items()}

        games = db_connector.execute("""SELECT * FROM games
                                     WHERE `game_name` LIKE 'PLAIN%'""")
        self.game2start_target = {v['game_name']:
                                  (self.id2name[v['start_page_id']],
                                   self.id2name[v['goal_page_id']])
                                  for v in games}

        nodes = pages = db_connector.execute('SELECT * FROM node_data')
        self.id2deg = {p['id']: p['degree'] for p in nodes}
        self.id2pr = {p['id']: p['pagerank'] for p in nodes}

        ngrams = db_connector.execute('SELECT * FROM ngrams')
        self.id2ngram = {p['id']: p['probability'] for p in ngrams}

        links = db_connector.execute('''SELECT page_id, SUM(amount) as links
                                     FROM links GROUP BY page_id;''')
        self.id2links = {p['page_id']: int(p['links']) for p in links}

    def read_edge_list_gt(self, filename, directed=True, parallel_edges=False):
        graph = gt.Graph(directed=directed)
        id_mapping = defaultdict(lambda: graph.add_vertex())
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


def main():
    # build the network
    nw = Network('data/links.txt')

    # load or compute the click data as a pandas frame
    try:  # load the precomputed click data
        clicks = pd.read_pickle('clicks.pd')
    except IOError:  # compute click data
        def parse_node(node_string):
            match = re.findall(r'/([^/]*?)\.htm', node_string)
            return match[0].replace('%25', '%') if match else ''

        results = []
        folder_logs = 'data/logfiles/'
        for folder in sorted(os.listdir(folder_logs)):
            print('\n', folder)
            files = sorted([f for f in os.listdir(folder_logs + folder)
                            if f.startswith('PLAIN')])
            for filename in files:
                print('   ', filename)
                fname = folder_logs + folder + '/' + filename
                df = pd.read_csv(fname, header=None, usecols=[1, 2, 3],
                                 names=['time', 'action', 'node'],
                                 infer_datetime_format=True, sep='\t')
                if df['action'].value_counts()['GAME_STARTED'] > 1:
                    print('        Error: duplicated game start, dropping',
                          folder, fname)
                    continue

                # ld = df[df['action'] == 'link_data']
                # ld['node'] = ld['node'].apply(lambda k: int(re.findall(r"link_nr': (\d+)", k)[0]))

                if df.iloc[-1]['action'] == 'GAME_COMPLETED':
                    idx = -1
                    while df.iloc[idx]['action'] != 'link_data':
                        idx -= 1
                        if idx < -df.shape[0]:
                            print('        Error: no link_data entry found')
                            pdb.set_trace()
                    df.loc[df.index[-1] + 1] = [df['time'].max(), 'load',
                                                df.iloc[idx]['node']]
                    df['success'] = pd.Series(np.ones(df.shape[0]),
                                              index=df.index)
                else:
                    df['success'] = pd.Series(np.zeros(df.shape[0]),
                                              index=df.index)
                df = df[df['action'] == 'load']
                if df.shape[0] < 2:
                    print('        Error: game too short, dropping',
                          folder, fname)
                    continue
                df['node'] = df['node'].apply(parse_node)
                df.index = np.arange(len(df))
                try:
                    df['node_id'] = [nw.name2id[n] for n in df['node']]
                    df['degree'] = [nw.id2deg[i] for i in df['node_id']]
                    df['pagerank'] = [nw.id2pr[i] for i in df['node_id']]
                    df['ngram'] = [nw.id2ngram[i] for i in df['node_id']]
                except KeyError, e:
                    print('        Error: key not found', folder, fname, e)
                    continue

                df.iloc[:, 0] = df.iloc[:, 0] - df.iloc[:, 0].min()
                if df['time'].min() < 0:
                    print(folder, fname)
                    pdb.set_trace()
                time_diff = df.iloc[:, 0].diff()[1:]
                time_diff.index = np.arange(time_diff.shape[0])
                df = df.iloc[:-1]
                df.iloc[:, 0] = time_diff
                # if df.shape[0] != ld.shape[0]:
                #     print('Error: ld.shape != df.shape', folder, fname)
                #     continue
                # linkpos = ld['node'] / [id2links[i] for i in df['node_id']]
                # df['linkpos'] = [v for v in linkpos]
                results.append(df)
                match = re.findall(r'(PLAIN_[\d]+_[a-z0-9_\-]+)\.', filename)
                if not match:
                    print('        Error: game mission not matched')
                    pdb.set_trace()
                else:
                    match = match[0]
                start, target = nw.game2start_target[match]

                if not start == df['node'].iloc[0] and\
                        target == df['node'].iloc[-1]:
                    print('        Error: start or target not present')
                    print(df)
                # pdb.set_trace()

        clicks = pd.concat(results, ignore_index=True)
        clicks['intercept'] = 1.0
        clicks.to_pickle('clicks.pd')

    pdb.set_trace()
    # run logistic regression
    train_cols = ['time', 'degree', 'pagerank', 'ngram', 'intercept']  #, 'linkpos']
    regressor = LogisticRegressor()
    regressor.regress(clicks['success'], clicks[train_cols])
    pdb.set_trace()


def compute_link_positions():
    nw = Network('data/links.txt')

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
    link_regex = re.compile('(<a href="../../wp/[^/]+/(.+?)\.htm" ' +
                             'title="[^"]+">.+?</a>)' +
                            '|' +
                            '(<area.+?href=' +
                            '"../../wp/[^/]+/(.+?)\.htm".+?/>)')
    folder = 'data/wiki-schools/wp/'
    plaintext_dir = 'data/wiki-schools/plaintext/'
    if not os.path.exists(plaintext_dir):
        os.makedirs(plaintext_dir)
    for i, a in enumerate(nw.name2id.keys()):
        print(unicode(i+1) + '/' + unicode(len(nw.name2id)))
        fname = folder + a[0].lower() + '/' + a + '.htm'
        with io.open(fname, encoding='utf-8') as infile:
            data = infile.read()
        data = data.split('<!-- start content -->')[1]
        data = data.split('<div class="printfooter">')[0]
        data = [d.strip() for d in data.splitlines()]
        data = [d for d in data if d]
        data = data[1:]  # skip background information about W4S
        text = []
        for d in data:
            parser.reset()
            parser.feed(parser.unescape(d))
            stripped_d = parser.get_data()
            if stripped_d:
                text.append(stripped_d)
        text = '\n'.join(text)
        with io.open(plaintext_dir + a + '.txt', 'w', encoding='utf-8') as outfile:
            outfile.write(text)


if __name__ == '__main__':
    # main()
    compute_link_positions()
    pdb.set_trace()
