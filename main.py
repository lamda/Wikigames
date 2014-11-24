# -*- coding: utf-8 -*-

from __future__ import division
import io
import os
import pdb
import re

# import graph_tool.all as gt
import numpy as np
import pandas as pd
import pymysql
import statsmodels.api as sm


class DbConnector:
    def __init__(self):
        self.db_host = '127.0.0.1'
        self.db_connection = pymysql.connect(host=self.db_host,
                                             port=3306,
                                             user='root',
                                             passwd='master',
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

        if _statement.startswith("SELECT"):
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


# def read_edge_list(filename, directed=False, parallel_edges=False):
#     graph = gt.Graph(directed=directed)
#     id_mapping = defaultdict(lambda: graph.add_vertex())
#     graph.vertex_properties['NodeId'] = graph.new_vertex_property('string')
#     with io.open(filename, encoding='utf-8') as infile:
#         for line in infile:
#             line = line.strip().split()
#             if len(line) == 2:
#                 src, dest = line
#                 src_v, dest_v = id_mapping[src], id_mapping[dest]
#                 graph.add_edge(src_v, dest_v)
#             elif len(line) == 1:
#                 node = line[0]
#                 _ = id_mapping[node]
#     for orig_id, v in id_mapping.iteritems():
#         graph.vertex_properties['NodeId'][v] = orig_id
#     if not parallel_edges:
#         gt.remove_parallel_edges(graph)
#     return graph


def main():
    # build the network from Patrick's links.txt file
    # graph = read_edge_list('data/links.txt', directed=True)
    # graph.save('data/links.gt')
    # graph = gt.load_graph('data/links.gt')
    # db_connector = DbConnector()
    # games = db_connector.execute('SELECT * FROM games')
    # game2st = {g['game_name']: (g['start_page_id'], g['goal_page_id'])
    #            for g in games}
    # pages = db_connector.execute('SELECT * FROM pages')
    # id2title = {p['id']: p['name'] for p in pages}
    # id2file = {p['id']: p['link'] for p in pages}

    def parse_node(node_string):
        match = re.findall(r'/([^/]*?)\.htm', node_string)
        return match[0] if match else ''

    results = []
    for folder in sorted(os.listdir('data/logfiles')):
        for filename in sorted(os.listdir('data/logfiles/' + folder)):
            fname = 'data/logfiles/' + folder + '/' + filename
            df = pd.read_csv(fname, header=None, usecols=[1, 2, 3],
                             names=['time', 'action', 'node'],
                             infer_datetime_format=True, sep='\t',
                             converters={'node': parse_node})
            if df['action'].value_counts()['GAME_STARTED'] > 1:
                print 'Error: duplicated game start'
                print 'dropping game', folder, fname
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
    train_cols = ['time', 'intercept']
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


if __name__ == '__main__':
    main()