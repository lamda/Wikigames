# -*- coding: utf-8 -*-

from __future__ import division
from collections import defaultdict
import io
import os
import pdb
import re
import urllib
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


def read_edge_list_gt(filename, directed=False, parallel_edges=False):
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


def read_edge_list_nx(filename, directed=False, parallel_edges=False):
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
    # build the network from Patrick's links.txt file (graph_tool variant)
    # graph = read_edge_list_gt('data/links.txt', directed=True,
    #                           parallel_edges=True)
    # graph.save('data/links.gt')
    # graph = gt.load_graph('data/links.gt')
    # build the network from Patrick's links.txt file (networkx variant)
    graph = read_edge_list_nx('data/links.txt', directed=True,
                              parallel_edges=True)
    db_connector = DbConnector()
    # games = db_connector.execute('SELECT * FROM games')
    # game2st = {g['game_name']: (g['start_page_id'], g['goal_page_id'])
    #            for g in games}
    pages = db_connector.execute('SELECT * FROM pages')
    id2title = {p['id']: p['name'] for p in pages}
    id2name = {p['id']: re.findall(r'\\([^\\]*?)\.htm', p['link'])[0] for p in pages}
    name2id = {v: k for k, v in id2name.items()}
    nodes = pages = db_connector.execute('SELECT * FROM node_data')
    id2deg = {p['id']: p['degree'] for p in nodes}
    id2pr = {p['id']: p['pagerank'] for p in nodes}

    def parse_node(node_string):
        match = re.findall(r'/([^/]*?)\.htm', node_string)
        return match[0].replace('%25', '%') if match else ''

    results = []
    for folder in sorted(os.listdir('data/logfiles')):
        for filename in sorted(os.listdir('data/logfiles/' + folder)):
            fname = 'data/logfiles/' + folder + '/' + filename
            df = pd.read_csv(fname, header=None, usecols=[1, 2, 3],
                             names=['time', 'action', 'node'],
                             infer_datetime_format=True, sep='\t',
                             converters={'node': parse_node})
            if df['action'].value_counts()['GAME_STARTED'] > 1:
                print 'Error: duplicated game start, dropping', folder, fname
                continue
            success = df.iloc[-1]['action'] == 'GAME_COMPLETED'
            if success:
                idx = -1
                while df.iloc[idx]['action'] != 'link_data':
                    idx -= 1
                    if idx < -df.shape[0]:
                        print 'Error: no link_data entry found'
                        pdb.set_trace()
                df.loc[df.index[-1] + 1] = [df['time'].max(), 'load',
                                            df.iloc[idx]['node']]
                df['success'] = pd.Series(np.ones(df.shape[0]), index=df.index)
            else:
                df['success'] = pd.Series(np.zeros(df.shape[0]), index=df.index)
            df = df[df['action'] == 'load']
            df.index = np.arange(len(df))
            try:
                df['node_id'] = [name2id[n] for n in df['node']]
                df['degree'] = [id2deg[i] for i in df['node_id']]
                df['pagerank'] = [id2pr[i] for i in df['node_id']]
            except KeyError, e:
                print 'Error: key not found', folder, fname, e
                continue

            df.iloc[:, 0] = df.iloc[:, 0] - df.iloc[:, 0].min()
            if df['time'].min() < 0:
                print folder, fname
                pdb.set_trace()
            time_diff = df.iloc[:, 0].diff()[1:]
            time_diff.index = np.arange(time_diff.shape[0])
            df = df.iloc[:-1]
            df.iloc[:, 0] = time_diff
            results.append(df)

    clicks = pd.concat(results, ignore_index=True)
    clicks['intercept'] = 1.0
    train_cols = ['time', 'degree', 'pagerank', 'intercept']
    logit = sm.Logit(clicks['success'], clicks[train_cols])
    result = logit.fit()
    print result.summary()
    print 'Baseline:', clicks[clicks['success'] == 1].shape[0] / clicks.shape[0]

    import matplotlib.pyplot as plt
    plt.scatter(clicks['time'], clicks['success'])
    plt.xlabel('duration')
    plt.ylabel('success')
    x = np.arange(0, 1000000, 100)
    b1, b0 = result.params.values
    p = [(np.exp(b0 + i*b1))/(1+np.exp(b0 + i*b1)) for i in x]
    plt.plot(x, p)
    plt.show()
    pdb.set_trace()


def get_ngrams():
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
        print i, '/', len(ids)
        probability = ngram_connector.get(id2name[i])
        stmt = 'INSERT INTO `ngrams` (node_id, probability) values (%s, %s)'\
               % (i, probability)
        db_connector.execute(stmt)
        db_connector.commit()

if __name__ == '__main__':
    # main()
    get_ngrams()