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
pd.options.mode.chained_assignment = None
import pymysql
from sklearn.feature_extraction.text import TfidfVectorizer
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
    def __init__(self, filename='data/links.txt', graph_tool=False):
        self.html_base_folder = 'data/wiki-schools/wp/'
        self.plaintext_folder = 'data/wiki-schools/plaintext/'

        # read the graph
        if graph_tool:
            self.graph = self.read_edge_list_gt(filename)
        else:
            self.graph = self.read_edge_list_nx(filename)

        # build some mappings from the database
        self.db_connector = DbConnector()
        pages = self.db_connector.execute('SELECT * FROM pages')
        self.id2title = {p['id']: p['name'] for p in pages}
        self.id2name = {p['id']:
                        re.findall(r'\\([^\\]*?)\.htm', p['link'])[0]
                        for p in pages}
        self.name2id = {v: k for k, v in self.id2name.items()}

        games = self.db_connector.execute("""SELECT * FROM games
                                     WHERE `game_name` LIKE 'PLAIN%'""")
        self.game2start_target = {v['game_name']:
                                  (self.id2name[v['start_page_id']],
                                   self.id2name[v['goal_page_id']])
                                  for v in games}

        nodes = pages = self.db_connector.execute('SELECT * FROM node_data')
        self.id2deg = {p['id']: p['degree'] for p in nodes}
        self.id2pr = {p['id']: p['pagerank'] for p in nodes}

        ngrams = self.db_connector.execute('SELECT * FROM ngrams')
        self.id2ngram = {p['id']: p['probability'] for p in ngrams}

        links = self.db_connector.execute('''SELECT page_id,
                                                    SUM(amount) as links
                                             FROM links GROUP BY page_id;''')
        self.id2links = {p['page_id']: int(p['links']) for p in links}

    def get_spl(self, start, target):
        """ get the shortest path length for two nodes from the database
        if this is too slow, add an index to the table as follows:
        ALTER TABLE path_lengths ADD INDEX page_id (page_id);
        """
        query = '''SELECT path_length FROM path_lengths
                   WHERE page_id=%d AND target_page_id=%d''' % (start, target)
        length = self.db_connector.execute(query)
        return length[0]['path_length']

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

    def extract_plaintext(self):
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
            fname = self.html_base_folder + a[0].lower() + '/' + a + '.htm'
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

    def compute_tfidf_similarity(self):
        """compute the TF-IDF cosine similarity between articles

        articles are sorted by their ID as in the MySQL database
        """

        # read plaintext files
        content = []
        for i, title in self.id2name.items():
            with io.open(self.plaintext_folder + title + '.txt',
                         encoding='utf-8') as infile:
                data = infile.read()
            content.append(data)

        # compute cosine TF-IDF similarity
        with io.open('data/stopwords.txt', encoding='utf-8') as infile:
            stopwords = infile.read().splitlines()
        tvec = TfidfVectorizer(stop_words=stopwords)
        tfidf = tvec.fit_transform(content)
        tfidf_similarity = tfidf * tfidf.T
        with open('data/tfidf_similarity_sparse.obj', 'wb') as outfile:
            pickle.dump(tfidf_similarity, outfile, -1)
        tfidf_similarity = tfidf_similarity.todense()
        with open('data/tfidf_similarity_dense.obj', 'wb') as outfile:
            pickle.dump(tfidf_similarity, outfile, -1)

    def get_tfidf_similarity(self):
        with open('data/tfidf_similarity_dense.obj', 'rb') as infile:
            tfidf_similarity = pickle.load(infile)
        return tfidf_similarity


def main():
    # build the network
    nw = Network()

    # load or compute the click data as a pandas frame
    try:  # load the precomputed click data
        data = pd.read_pickle('data/data.pd')
    except IOError:  # compute click data
        # helper functions
        def parse_node(node_string):
            m = re.findall(r'/([^/]*?)\.htm', node_string)
            return m[0].replace('%25', '%') if m else ''

        def print_error(message):
            print('        Error:', message, folder, filename)

        results = []
        folder_logs = 'data/logfiles/'
        for folder in sorted(os.listdir(folder_logs)):
            print('\n', folder)
            files = sorted([f for f in os.listdir(folder_logs + folder)
                            if f.startswith('PLAIN')])
            for filename in files:
                print('   ', filename)
                fname = folder_logs + folder + '/' + filename
                df_full = pd.read_csv(fname, sep='\t',
                                      usecols=[2, 3],
                                      names=['action', 'node'])

                # perform sanity checks
                action_counts = df_full['action'].value_counts()
                if action_counts['GAME_STARTED'] > 1:
                    print_error('duplicated_game_start, dropping')
                    continue
                elif action_counts['load'] < 2:
                    print_error('game too short, dropping')
                    continue

                # get additional mission attributes
                successful = df_full.iloc[-1]['action'] == 'GAME_COMPLETED'
                match = re.findall(r'(PLAIN_[\d]+_[a-z0-9_\-]+)\.', filename)[0]
                start, target = nw.game2start_target[match]
                df = df_full[df_full['action'] == 'load']
                df.drop('action', inplace=True, axis=1)
                df['node'] = df['node'].apply(parse_node)
                if not df.iloc[0]['node'] == start:
                    print_error('start node not present')
                    pdb.set_trace()
                if successful and not target == df.iloc[-1]['node']:
                    last = df_full[df_full['action'] == 'link_data']
                    last = parse_node(last.iloc[-1]['node'])
                    df.loc[df.index[-1] + 1] = [last]
                spl = nw.get_spl(nw.name2id[start], nw.name2id[target])

                df.index = np.arange(len(df))
                try:
                    df['node_id'] = [nw.name2id[n] for n in df['node']]
                    df['degree'] = [nw.id2deg[i] for i in df['node_id']]
                    df['pagerank'] = [nw.id2pr[i] for i in df['node_id']]
                    df['ngram'] = [nw.id2ngram[i] for i in df['node_id']]
                except KeyError, e:
                    print_error('key not found, dropping' + repr(e))
                    continue

                results.append({
                    'data': df,
                    'successful': successful,
                    'spl': spl,
                    'pl': len(df)
                })

        data = pd.DataFrame(results)
        data.to_pickle('data/data.pd')

    pdb.set_trace()


if __name__ == '__main__':
    main()
