# -*- coding: utf-8 -*-

from __future__ import division, print_function
from collections import defaultdict
import cPickle as pickle
import HTMLParser
import io
import os
import pdb
import re
import sys
import urllib2

import numpy as np
import pandas as pd
import pymysql
import PySide.QtCore
import PySide.QtGui
import PySide.QtWebKit
import statsmodels.api as sm

import credentials


# set a few options
pd.options.mode.chained_assignment = None


class DbConnector(object):
    def __init__(self, db):
        self.db_host = '127.0.0.1'
        self.db_connection = pymysql.connect(host=self.db_host,
                                             port=3306,
                                             user='root',
                                             passwd='',
                                             db=db,
                                             charset='utf8')
        self.db_cursor = self.db_connection.cursor(pymysql.cursors.DictCursor)
        self.db_cursor_nobuff = self.db_connection.cursor(
            pymysql.cursors.SSCursor)
        self.db = db

    def __exit__(self):
        self.db_cursor.close()
        self.db_connection.close()

    def execute(self, _statement, _args=None, _type=None):
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


class NgramFrequency(object):
    def __init__(self):
        token = credentials.microsoft_token
        corpus = 'bing-body/2013-12/5/'
        base_url = 'http://weblm.research.microsoft.com/weblm/rest.svc/'
        self.base_url = base_url + corpus + 'jp?u=' + token + '&p='
        try:
            with open('data/ngrams.obj', 'rb') as infile:
                    self.ngrams = pickle.load(infile)
        except (IOError, EOFError):
            self.ngrams = {}

    def get_frequency(self, title):
        try:
            return self.ngrams[title]
        except KeyError:
            self.retrieve_frequency(title)
            return self.ngrams[title]

    def retrieve_frequency(self, title):
        title_url = title.replace(' ', '+').replace('_', '+')
        url = self.base_url + title_url
        self.ngrams[title] = float(urllib2.urlopen(url).read())

    def __del__(self):
        with open('data/ngrams.obj', 'wb') as outfile:
            pickle.dump(self.ngrams, outfile, -1)


class LogisticRegressor(object):
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


class WebPageSize(PySide.QtGui.QMainWindow):
    def __init__(self, qt_application, label):
        self.qt_application = qt_application
        PySide.QtGui.QMainWindow.__init__(self)
        self.web_view = PySide.QtWebKit.QWebView()
        self.setCentralWidget(self.web_view)
        self.web_view.loadFinished.connect(self._load_finished)
        self.pickle_path = os.path.join('data', label, 'webpagesizes.obj')
        try:
            with open(self.pickle_path, 'rb') as infile:
                    self.size = pickle.load(infile)
        except (IOError, EOFError):
            self.size = {}
        self.curr_page = ''
        self.curr_width = 0
        self.server = True
        if self.server:
            self.base_url = 'http://localhost:8000/wp/'
        else:
            self.base_url = 'file:///C:/PhD/Code/Wikigames/data/' + label +\
                            '/wpcd/wp/'

    def get_size(self, page, width):
        try:
            return self.size[(page, width)]
        except KeyError:
            self.compute_size(page, width)
            return self.size[(page, width)]

    def compute_size(self, page, width):
        self.curr_page = page
        self.curr_width = width
        if self.server:
            path = self.base_url + '/' + page[0].lower() + '/' + page + '.htm'
        else:
            path = PySide.QtCore.QUrl(page[0].lower() + '/' + page + '.htm')
        self.web_view.load(path)
        self.web_view.page().setViewportSize(PySide.QtCore.QSize(width, 1))
        self.qt_application.exec_()

    def _load_finished(self):
        frame = self.web_view.page().mainFrame()
        html_data = frame.toHtml()
        result = (frame.contentsSize().width(), frame.contentsSize().height())
        self.size[(self.curr_page, self.curr_width)] = result
        self.close()

    def __del__(self):
        with open(self.pickle_path, 'wb') as outfile:
            pickle.dump(self.size, outfile, -1)


class Wikigame(object):
    def __init__(self, label):
        self.label = label
        self.data = None
        self.graph = None
        self.html_base_folder = 'data/' + label + '/wpcd/wp/'
        self.plaintext_folder = 'data/' + label + '/wpcd/plaintext/'
        self.tfidf_similarity = None
        self.category_depth = None
        self.category_distance = None
        self.link2pos_first, self.link2pos_last = None, None
        self.lengths, self.pos2link = None, None
        self.spl = {}

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

    def read_edge_list_gt(self, filename, directed=True, parallel_edges=False):
        if gt is None:
            import graph_tool.all as gt
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

        from sklearn.feature_extraction.text import TfidfVectorizer

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
        tfidf_similarity = tfidf_similarity.todense()
        tfidf_path = 'data/' + self.label + '/tfidf_similarity.obj'
        with open(tfidf_path, 'wb') as outfile:
            pickle.dump(tfidf_similarity, outfile, -1)

    def get_tfidf_similarity(self, start, target):
        if self.tfidf_similarity is None:
            tfidf_path = 'data/' + self.label + '/tfidf_similarity.obj'
            with open(tfidf_path, 'rb') as infile:
                self.tfidf_similarity = pickle.load(infile)
        # subtract one because Wikipedia ids start with 1 and not 0
        return self.tfidf_similarity[start-1, target-1]

    def compute_category_stats(self):
        category = defaultdict(lambda: np.NaN)
        category_depth = defaultdict(lambda: np.NaN)
        for i in sorted(self.id2name.keys()):
            a = self.id2name[i]
            print(i, '/', len(self.name2id), end='\r')
            ofname = self.html_base_folder + a[0].lower() + '/' + a + '.htm'
            try:
                with io.open(ofname, encoding='utf-8') as infile:
                    data = infile.readlines()
            except UnicodeDecodeError:
                with io.open(ofname) as infile:
                    data = infile.readlines()

            for line in data:
                m = re.findall(r'subject\.(.+?)\.ht', line)
                if m:
                    category_depth[i] = np.mean([(p.count('.') + 1) for p in m])
                    category[i] = [p.split('.') for p in m]
                    break
        category_distance = np.zeros((len(self.name2id), len(self.name2id))) - 1
        for i, ai in enumerate(sorted(self.name2id.keys())):
            print(i, '/', len(self.name2id), end='\r')
            i += 1
            for j, ja in enumerate(sorted(self.name2id.keys())):
                if i == j:
                    category_distance[i, j] = 0
                elif i > j:
                    category_distance[i, j] = category_distance[j, i]
                else:
                    min_dists = []
                    for p in category[i]:
                        min_dist = 1000
                        for q in category[j]:
                            shared = 2 * sum([a == b for a, b in zip(p, q)])
                            d = len(p) + len(q) - shared
                            if d < min_dist:
                                min_dist = d
                        min_dists.append(min_dist)

                    for q in category[j]:
                        min_dist = 1000
                        for p in category[i]:
                            shared = 2 * sum([a == b for a, b in zip(p, q)])
                            d = len(p) + len(q) - shared
                            if d < min_dist:
                                min_dist = d
                        min_dists.append(min_dist)

                    num_cats = len(category[i]) + len(category[j])
                    if num_cats > 0:
                        category_distance[i, j] = sum(min_dists) / num_cats
                    else:
                        # pages do not have categories
                        category_distance[i, j] = 100

        category_path = 'data/' + self.label + '/category_distance.obj'
        with open(category_path, 'wb') as outfile:
            pickle.dump(category_distance, outfile, -1)

        category_path = 'data/' + self.label + '/category_depth.obj'
        with open(category_path, 'wb') as outfile:
            pickle.dump(category_depth, outfile, -1)

    def get_category_depth(self, node):
        if self.category_depth is None:
            category_path = 'data/' + self.label + '/category_depth.obj'
            with open(category_path, 'rb') as infile:
                self.category_depth = pickle.load(infile)
        return self.category_depth[node]

    def get_category_distance(self, start, target):
        if self.category_distance is None:
            path = os.path.join('data', self.label, 'category_distance.obj')
            with open(path, 'rb') as infile:
                self.category_distance = pickle.load(infile)
        # subtract one because Wikipedia ids start with 1 and not 0
        return self.category_distance[start-1, target-1]

    def get_spl(self, start, target):
        """ get the shortest path length for two nodes from the database
        if this is too slow, add an index to the table as follows:
        ALTER TABLE path_lengths ADD INDEX page_id (page_id);
        """
        try:
            return self.spl[(start, target)]
        except KeyError:
            query = '''SELECT path_length FROM path_lengths
                       WHERE page_id=%d AND target_page_id=%d'''\
                    % (start, target)
            length = self.db_connector.execute(query)
        if not length:
            self.spl[(start, target)] = np.NaN
        else:
            self.spl[(start, target)] = length[0]['path_length']
        return self.spl[(start, target)]

    def compute_link_positions(self):
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
        link2pos_first = {}
        link2pos_last = {}
        pos2link = {}
        lengths = {}
        for i, a in enumerate(self.name2id.keys()):
            print(unicode(i+1) + '/' + unicode(len(self.name2id)), end='\r')
            lpos_first = defaultdict(lambda: np.NaN)
            lpos_last = defaultdict(lambda: np.NaN)
            posl = defaultdict(int)
            fname = os.path.join(folder, a[0].lower(), a + '.htm')
            try:
                with io.open(fname, encoding='utf-8') as infile:
                    data = infile.read()
            except UnicodeDecodeError, e:
                # there exist decoding errors for a few irrelevant pages
                print(fname)
                continue
            data = data.split('<!-- start content -->')[1]
            data = data.split('<div class="printfooter">')[0]

            regex_results = link_regex.findall(data)
            regex_results = [(r[0], r[2]) for r in regex_results]
            for link in regex_results:
                link = [l for l in link if l]
                data = data.replace(link[0], ' [['+link[1]+']] ')
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
            words = (re.split(': |\. |, |\? |! |\n | ', text))
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
                    except KeyError:
                        pass
            link2pos_first[a] = lpos_first
            link2pos_last[a] = lpos_last
            pos2link[a] = posl
            lengths[a] = len(words)
        path = os.path.join('data', self.label, 'link_positions.obj')
        with open(path, 'wb') as outfile:
            pickle.dump([link2pos_first, link2pos_last, lengths, pos2link],
                        outfile, -1)

    def load_link_positions(self):
        if self.link2pos_first is None:
            path = os.path.join('data', self.label, 'link_positions.obj')
            with open(path, 'rb') as infile:
                self.link2pos_first, self.link2pos_last, self.lengths,\
                    self.pos2link = pickle.load(infile)

    def load_data(self):
        self.data = pd.read_pickle(os.path.join('data', self.label, 'data.pd'))

    def load_graph(self, graph_tool=False):
        # read the graph
        path = os.path.join('data', self.label, 'links.txt')
        if graph_tool:
            self.graph = self.read_edge_list_gt(path)
        else:
            self.graph = self.read_edge_list_nx(path)

    def plot_link_amount_distribution(self):
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set_palette(sns.color_palette(["#9b59b6", "#3498db", "#95a5a6",
                                   "#e74c3c", "#34495e", "#2ecc71"]))
        query = 'SELECT amount, COUNT(*) FROM links GROUP BY amount;'
        df = pd.io.sql.read_sql(query, self.db_connector.db_connection)
        df.index = df.amount
        df.plot(kind='bar')
        plt.show()
        frac = df.iloc[1:]['COUNT(*)'].sum() / df['COUNT(*)'].sum()
        print('links with multiple occurrences:', frac)

    # dataframe preparation helpers ----

    def check_spl(self, data, successful):
        # assert that the spl data is correct
        assert all(y >= x-1 for x, y in zip(data, data[1:])),\
            'error in SPL: erroneous difference in SPL'
        if successful:
            assert data[-1] == 0,\
                'error in SPL: last node is not target in successful mission'

    def print_error(self, message):
        print('        Error:', message)


class WIKTI(Wikigame):
    def __init__(self, graph_tool=False):
        super(WIKTI, self).__init__('wikti')

    def create_dataframe(self):
        # load or compute the click data as a pandas frame
        # helper functions

        regex_parse_node = re.compile(r'/([^/]*?)\.htm')
        regex_parse_node_link = re.compile(r"offset': (\d+)")

        def parse_node(node_string):
            m = regex_parse_node.findall(node_string)
            return m[0].replace('%25', '%') if m else ''

        def parse_node_link(node_string):
            m = regex_parse_node_link.findall(node_string)
            return int(m[0]) if m else ''

        # web page size calculation disabled for now - needs a workover
        # regex_scroll = r"u'scroll': {u'y': (\d+), u'x': \d+}," \
        #                r" u'size': {u'y': (\d+), u'x': (\d+)"
        qt_application = PySide.QtGui.QApplication(sys.argv)
        # page_size = WebPageSize(qt_application, self.label)
        results = []
        folder_logs = 'data/' + self.label + '/logfiles/'
        ngrams = NgramFrequency()
        self.load_link_positions()
        folders = ['U' + '%02d' % i for i in range(1, 10)]
        for folder in folders:
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
                    self.print_error('duplicated_game_start, dropping')
                    continue
                elif action_counts['load'] < 2:
                    self.print_error('game too short, dropping')
                    continue

                # get additional mission attributes
                successful = df_full.iloc[-1]['action'] == 'GAME_COMPLETED'
                match = re.findall(r'(PLAIN_[\d]+_[a-z0-9_\-]+)\.',
                                   filename)[0]
                start, target = self.game2start_target[match]
                df = df_full[df_full['action'] == 'load']
                df.drop('action', inplace=True, axis=1)
                df['node'] = df['node'].apply(parse_node)
                if not df.iloc[0]['node'] == start:
                    self.print_error('start node not present')
                    pdb.set_trace()
                if successful and not target == df.iloc[-1]['node']:
                    # insert the target if the load event is not present
                    # this is the case for some of the earlier log files
                    last = df_full[df_full['action'] == 'link_data']
                    last = parse_node(last.iloc[-1]['node'])
                    df.loc[df.index[-1] + 1] = [last]
                    df_full.loc[df_full.index[-1] + 1] = ['load', last]
                    if last != target:
                        # in some cases, the target is entirely missing
                        df.loc[df.index[-1] + 1] = target
                        df_full.loc[df_full.index[-1] + 1] = ['load', target]

                link_data = df_full[df_full['action'] == 'link_data']
                link_data.drop('action', inplace=True, axis=1)
                link_data = link_data['node'].apply(parse_node_link)

                spl = self.get_spl(self.name2id[start],
                                   self.name2id[target])

                # get scrolling range
                # idx = list(df_full[df_full['action'] == 'load'].index)
                # df_groups = [df_full.loc[a:b, :]
                #              for a, b in zip(idx, idx[1:])]
                # exploration = [np.nan]
                # for i, g in enumerate(df_groups):
                #     print('            ', df.iloc[i]['node'])
                #     slct = (g['action'] == 'scroll') | (g['action'] == 'resize')
                #     if len(g[slct]) == 0:
                #         from_index = None
                #         print('            ', 'from_index is None')
                #     else:
                #         from_index = g[slct].index[0]
                #     df_scroll = g.loc[from_index:]
                #     df_scroll = df_scroll.node.str.extract(regex_scroll)
                #     df_scroll = df_scroll.dropna()
                #     df_scroll.columns = ['scrolled', 'height', 'width']
                #     df_scroll['scrolled'] = df_scroll['scrolled'].apply(int)
                #     df_scroll['height'] = df_scroll['height'].apply(int)
                #     df_scroll['width'] = df_scroll['width'].apply(int)
                #     seen_log = df_scroll.loc[df_scroll['scrolled'].idxmax()]
                #     seen_max = page_size.get_size(df.iloc[i].node, seen_log[2])[1]
                #     seen = seen_log['scrolled'] + seen_log['height']
                #     if from_index is None:
                #         seen_max = seen
                #     exploration.append(seen / seen_max)
                #     print(df.iloc[0].node, seen, seen_max)
                #     pdb.set_trace()
                #     TODO: This currently doesn't work

                try:
                    df['node_id'] = [self.name2id[n]
                                     for n in df['node']]
                    df['degree_out'] = [self.id2deg_out[i]
                                        for i in df['node_id']]
                    df['degree_in'] = [self.id2deg_in[i]
                                       for i in df['node_id']]
                    df['ngram'] = [ngrams.get_frequency(n)
                                   for n in df['node']]

                    tid = self.name2id[target]
                    df['spl_target'] = [self.get_spl(i, tid)
                                        for i in df['node_id']]
                    try:
                        self.check_spl(df['spl_target'].tolist(), successful)
                    except AssertionError, a:
                        pdb.set_trace()
                    df['tfidf_target'] = [1 - self.get_tfidf_similarity(i,
                                                                        tid)
                                          for i in df['node_id']]
                    df['category_depth'] = [self.get_category_depth(i)
                                            for i in df['node_id']]
                    df['category_target'] = [self.get_category_distance(i,
                                                                        tid)
                                             for i in df['node_id']]
                    # df['exploration'] = exploration
                    zipped = zip(df['node'].iloc[0:], df['node_id'].iloc[1:])
                    df['linkpos_first'] = [self.link2pos_first[a][b]
                                           for a, b in zipped] + [np.NaN]
                    df['linkpos_last'] = [self.link2pos_last[a][b]
                                          for a, b in zipped] + [np.NaN]
                    # try:
                    #     df['linkpos_actual'] = link_data.tolist() + [np.NaN]
                    # except ValueError:
                    #     pdb.set_trace()
                    if 0 in df['linkpos_first'].tolist() or 0 in df['linkpos_last'].tolist():
                        print(df)
                        pdb.set_trace()
                except KeyError, e:
                    self.print_error('key not found, dropping' + repr(e))
                    continue

                results.append({
                    'data': df,
                    'successful': successful,
                    'spl': spl,
                    'pl': len(df)
                })

        data = pd.DataFrame(results)
        data.to_pickle('data/' + self.label + '/data.pd')


class Wikispeedia(Wikigame):
    def __init__(self):
        super(Wikispeedia, self).__init__('wikispeedia')

    @staticmethod
    def build_database(label):
        db_connector = DbConnector('')
        with io.open('data/' + label + '/SQL/structure.sql', encoding='utf-8') \
                as infile:
            stmt = infile.read()
        db_connector.execute(stmt)
        db_connector.execute('USE ' + label + ';')
        db_connector.commit()

    @staticmethod
    def fill_database():
        from modules import PageExtractor
        from modules import LinkExtractor
        from modules import LinkCleaner
        from modules import PathCalculator
        from modules import NodeValues

        db_connector = DbConnector('wikispeedia')

        page_extractor = PageExtractor.PageExtractor(db_connector)
        page_extractor.run()

        link_extractor = LinkExtractor.LinkExtractor(db_connector)
        link_extractor.run()

        link_cleaner = LinkCleaner.LinkCleaner(db_connector)
        link_cleaner.run()

        path_calculator = PathCalculator.PathCalculator(db_connector)
        path_calculator.run()

        node_values = NodeValues.NodeValues(db_connector)
        node_values.run()

    def create_dataframe(self):
        from datetime import datetime
        # load or compute the click data as a pandas frame
        results = []
        folder_logs = os.path.join('data', self.label, 'logfiles')
        ngrams = NgramFrequency()
        self.load_link_positions()
        for filename in sorted(os.listdir(folder_logs))[:1]:
            print(filename)
            fname = os.path.join(folder_logs, filename)
            successful = False if 'unfinished' in filename else True
            df_full = pd.read_csv(fname, sep='\t', comment='#',
                                  usecols=[3], names=['path'])
            paths = df_full['path'].str.split(';').tolist()
            start = pd.DataFrame([t[0] for t in paths])
            df_full['start'] = start
            if successful:
                target = pd.DataFrame([t[-1] for t in paths])
            else:
                target = pd.read_csv(fname, sep='\t', comment='#',
                                     usecols=[4], names=['target'])
            df_full['target'] = target
            for eid, entry in enumerate(df_full.iterrows()):
                print(eid, end='\r')
                node = entry[1]['path'].split(';')
                if '<' in node:
                    # resolve backtracks
                    node.reverse()
                    game = [node[-1]]
                    stack = [node.pop()]
                    while node:
                        p = node.pop()
                        if p == '<':
                            stack.pop()
                        else:
                            stack.append(p)
                        game.append(stack[-1])
                    node = game
                try:
                    node_id = [self.name2id[n] for n in node]
                    degree_out = [self.id2deg_out[i] for i in node_id]
                    degree_in = [self.id2deg_in[i] for i in node_id]
                    ngram = [ngrams.get_frequency(n) for n in node]
                    tid = self.name2id[entry[1]['target']]
                    spl_target = [self.get_spl(i, tid) for i in node_id]
                    tfidf_target = [1 - self.get_tfidf_similarity(i, tid)
                                    for i in node_id]
                    category_depth = [self.get_category_depth(i)
                                      for i in node_id]
                    category_target = [self.get_category_distance(i, tid)
                                       for i in node_id]
                    zipped = zip(node, node_id[1:])
                    linkpos_first = [self.link2pos_first[a][b]
                                     for a, b in zipped] + [np.NaN]
                    linkpos_last =[self.link2pos_last[a][b]
                                   for a, b in zipped] + [np.NaN]
                except KeyError:
                    continue
                data = zip(node, node_id, degree_out, degree_in,
                           ngram, spl_target, tfidf_target,
                           category_depth, category_target, linkpos_first,
                           linkpos_last)
                columns = ['node', 'node_id', 'degree_out', 'degree_in',
                           'ngram', 'spl_target', 'tfidf_target',
                           'category_depth', 'category_target',
                           'linkpos_first', 'linkpos_last']
                df = pd.DataFrame(data=data, columns=columns)
                spl = self.get_spl(self.name2id[entry[1]['start']],
                                   self.name2id[entry[1]['target']])

                results.append({
                    'data': df,
                    'successful': successful,
                    'spl': spl,
                    'pl': len(entry[1]['path'].split(';'))
                })

        data = pd.DataFrame(results)
        data.to_pickle('data/' + self.label + '/data.pd')


if __name__ == '__main__':
    # Wikispeedia.fill_database()

    w = WIKTI()
    # w = Wikispeedia()
    w.compute_tfidf_similarity()
    w.compute_category_stats()
    w.compute_link_positions()
    w.create_dataframe()
