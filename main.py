# -*- coding: utf-8 -*-

from __future__ import division, print_function
import bisect
from collections import defaultdict
import cPickle as pickle
import HTMLParser
import io
import os
import pdb
import re
import urllib2

import numpy as np
import pandas as pd
import pymysql
import PySide.QtCore
import PySide.QtGui
import PySide.QtWebKit

from decorators import Cached
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


class NgramFrequency(object):
    def __init__(self):
        self.token = credentials.microsoft_token
        self.types = (
            # 'anchor',
            # 'body',
            'query',
            # 'title',
        )
        try:
            with open(os.path.join('data', 'ngram.obj'), 'rb') as infile:
                    self.ngram = pickle.load(infile)
        except (IOError, EOFError):
            self.ngram = {tp: {} for tp in self.types}
        url_base = 'http://weblm.research.microsoft.com/rest.svc/bing-'
        self.url = {tp: url_base + tp + '/2013-12/5/jp?u=' + self.token + '&p='
                    for tp in self.types}

    def get_frequency(self, ngram_type, title):
        try:
            return self.ngram[ngram_type][title]
        except KeyError:
            self.retrieve_frequency(ngram_type, title)
            return self.ngram[ngram_type][title]

    def retrieve_frequency(self, ngram_type, title):
        title_url = title.replace(' ', '+').replace('_', '+')
        url = self.url[ngram_type] + title_url
        self.ngram[ngram_type][title] = float(urllib2.urlopen(url).read())

    def save(self):
        with open(os.path.join('data', 'ngram.obj'), 'wb') as outfile:
            pickle.dump(self.ngram, outfile, -1)


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
            path = os.path.join(self.base_url, page[0].lower(), page + '.htm')
        else:
            path = PySide.QtCore.QUrl(os.path.join(page[0].lower(),
                                                   page + '.htm'))
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
        print(label)
        self.label = label
        self.data = None
        self.graph = None
        self.html_base_folder = os.path.join('data', label, 'wpcd', 'wp')
        self.plaintext_folder = os.path.join('data', label, 'wpcd', 'plaintext')
        self.cache_folder = os.path.join('data', label, 'cache')

        self.tfidf_similarity = None

        self.category_depth = None
        self.category_distance = None
        self.link2pos_first, self.link2pos_last = None, None
        self.length, self.pos2link = None, None
        self.ib_length, self.lead_length = None, None
        self.spl = None
        self.link_context, self.link_sets = None, None
        self.ngram = NgramFrequency()

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

    def __enter__(self):
        return self

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

    @Cached
    def get_tfidf_similarity(self, start, target):
        if start < target:
            start, target = target, start

        query = '''SELECT similarity FROM tfidf_similarities
                   WHERE page_id=%d AND target_page_id=%d''' % (start, target)
        similarity = self.db_connector.execute(query)
        return similarity[0]['similarity']

    @Cached
    def get_category_depth(self, node):
        query = '''SELECT category_depth FROM node_data
                   WHERE id=%d''' % node
        depth = self.db_connector.execute(query)
        return depth[0]['depth']

    @Cached
    def get_category_distance(self, start, target):
        if start < target:
            start, target = target, start

        query = '''SELECT distance FROM category_distance
                   WHERE page_id=%d AND target_page_id=%d''' % (start, target)
        distance = self.db_connector.execute(query)
        return distance[0]['distance']

    @Cached
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
        link2pos_first, link2pos_last, pos2link = {}, {}, {}
        length, ib_length, lead_length = {}, {}, {}
        for i, a in enumerate(self.name2id.keys()):
            print(unicode(i+1), '/', unicode(len(self.name2id)), end='\r')
            lpos_first, lpos_last, posl = {}, {}, {}
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
                    except KeyError:
                        pass
            link2pos_first[a] = lpos_first
            link2pos_last[a] = lpos_last
            pos2link[a] = posl
            length[a] = len(words)
        path = os.path.join('data', self.label, 'link_positions.obj')
        with open(path, 'wb') as outfile:
            pickle.dump([link2pos_first, link2pos_last,
                         length, pos2link, ib_length, lead_length], outfile, -1)

    def load_link_positions(self):
        if self.link2pos_first is None:
            path = os.path.join('data', self.label, 'link_positions.obj')
            with open(path, 'rb') as infile:
                self.link2pos_first, self.link2pos_last, self.length,\
                    self.pos2link, self.ib_length,\
                    self.lead_length = pickle.load(infile)

    def get_link_context(self, start, pos):
        if self.link_context is None:
            try:
                path = os.path.join('data', self.label, 'link_context.obj')
                with open(path, 'rb') as infile:
                    self.link_context = pickle.load(infile)
            except (IOError, EOFError):
                self.link_context = {}
        if self.link_sets is None:
            self.link_sets = {k: sorted(self.pos2link[k].keys())
                              for k in self.pos2link}
        try:
            return self.link_context[(start, pos)]
        except KeyError:
            if np.isnan(pos):
                self.link_context[(start, pos)] = np.NaN
            else:
                ctxt = sum(pos - 10 <= l <= pos + 10
                           for l in self.link_sets[start])
                self.link_context[(start, pos)] = ctxt

        return self.link_context[(start, pos)]

    def load_data(self):
        if self.data is None:
            path = os.path.join('data', self.label, 'data.obj')
            self.data = pd.read_pickle(path)

    def save_data(self, data=None):
        if data is None:
            data = self.data
        data.to_pickle(os.path.join('data', self.label, 'data.obj'))

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

    def __exit__(self, type, value, traceback):
        self.db_connector.close()
        self.ngram.save()
        if self.spl and len(self.spl) > 10:
            path = os.path.join('data', self.label, 'spl.obj')
            with open(path, 'wb') as outfile:
                pickle.dump(self.spl, outfile, -1)
        if self.link_context and len(self.link_context) > 10:
            path = os.path.join('data', self.label, 'link_context.obj')
            with open(path, 'wb') as outfile:
                pickle.dump(self.link_context, outfile, -1)


class WIKTI(Wikigame):
    label = 'wikti'

    def __init__(self):
        super(WIKTI, self).__init__(WIKTI.label)

    def fill_database(self):
        from modules.TfidfCalculator import TfidfCalculator
        from modules.CategoryCalculator import CategoryCalculator

        db_connector = DbConnector('wikti')

        # path_calculator = TfidfCalculator(db_connector, self.plaintext_folder)
        # path_calculator.run()

        cat_calculator = CategoryCalculator(db_connector, self.html_base_folder,
                                            self.label)
        cat_calculator.run()

        db_connector.close()

    def create_dataframe(self):
        """compute the click data as a pandas frame"""
        print('creating dataframe...')

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
        # qt_application = PySide.QtGui.QApplication(sys.argv)
        # page_size = WebPageSize(qt_application, self.label)
        results = []
        folder_logs = os.path.join('data', self.label, 'logfiles')
        self.load_link_positions()
        folders = ['U' + '%02d' % i for i in range(1, 10)]
        for folder in folders:
            print('\n', folder)
            # get missions and sort them numerically
            files = sorted(os.listdir(os.path.join(folder_logs, folder)))
            files = [f for f in files if f.startswith('PLAIN')]
            mission2fname = {int(re.findall(r'PLAIN\_\d+\_(\d+)', m)[0]): m
                             for m in files}
            for mission in sorted(mission2fname.keys()):
                filename = mission2fname[mission]
                print('   ', filename)
                fname = os.path.join(folder_logs, folder, filename)
                df_full = pd.read_csv(fname, sep='\t', usecols=[1, 2, 3],
                                      names=['time', 'action', 'node'])

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
                df.drop(['time', 'action'], inplace=True, axis=1)
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
                    last_time = df_full.iloc[-2]['time']
                    df_full.loc[df_full.index[-1]+1] = [last_time, 'load', last]
                    if last != target:
                        # in some cases, the target is entirely missing
                        df.loc[df.index[-1] + 1] = target
                        df_full.loc[df_full.index[-1] + 1] = [last_time, 'load',
                                                              target]

                # get time information
                time_data = df_full[df_full['action'] == 'load']['time']
                time = time_data.diff().shift(-1)
                time_normalized = time / sum(time.iloc[:-1])
                word_count = [self.length[a] if a in self.length else np.NaN
                              for a in df['node']]
                link_count = [len(self.pos2link[a])
                              if a in self.length else np.NaN
                              for a in df['node']]
                time_word = time / word_count
                time_link = time / link_count

                # get raw link position information
                link_data = df_full[(df_full['action'] == 'link_data') |
                                    (df_full['action'] == 'load')]
                link_data = link_data.copy(deep=True)
                actions = link_data['action'].tolist()[1:]
                zipped = zip(actions, actions[1:])

                action_indices = []
                for idx, act in enumerate(zipped):
                    if act == ('link_data', 'link_data'):
                        action_indices.append(idx + 1)
                for a in action_indices:
                    link_data.drop(link_data.index[a], inplace=True)

                action_indices = []
                for idx, act in enumerate(zipped):
                    if act == ('load', 'load'):
                        action_indices.append(idx + 2)
                for a in action_indices:
                    link_data.loc[link_data.index[a]-1] = ['', 'link_data', '']

                link_data = link_data[link_data['action'] == 'link_data']
                link_data.sort_index(inplace=True)
                link_data.drop(['time', 'action'], inplace=True, axis=1)
                link_data = link_data['node'].apply(parse_node_link)

                # correct to actual link position
                link_data_correct = []

                for name_start, name_target, pos in zip(df['node'].tolist(),
                                                        df['node'].tolist()[1:],
                                                        link_data.tolist()):
                    try:
                        links = [k for k, v in self.pos2link[name_start].items()
                                 if v == self.name2id[name_target]]
                    except KeyError:
                        continue
                    if len(links) == 0:
                        link_data_correct.append(np.NaN)
                    elif len(links) == 1:
                        link_data_correct.append(links[0])
                    else:
                        links = sorted(links)
                        pos = bisect.bisect(links, pos)
                        link_data_correct.append(links[pos - 1])
                    link_data = link_data_correct

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
                #     TODO: This currently doesn't work
                #     TODO: add this to Wikigame.close()

                try:
                    df['node_id'] = [self.name2id[n] for n in df['node']]
                    df['degree_out'] = [self.id2deg_out[i]
                                        for i in df['node_id']]
                    df['degree_in'] = [self.id2deg_in[i] for i in df['node_id']]

                    for tp in self.ngram.types:
                        df['ngram_' + tp] = [self.ngram.get_frequency(tp, n)
                                             for n in df['node']]
                    tid = self.name2id[target]
                    df['spl_target'] = [self.get_spl(i, tid)
                                        for i in df['node_id']]
                    try:
                        self.check_spl(df['spl_target'].tolist(), successful)
                    except AssertionError, a:
                        print(a)
                        pdb.set_trace()
                    df['tfidf_target'] = [1 - self.get_tfidf_similarity(i, tid)
                                          for i in df['node_id']]
                    df['category_depth'] = [self.get_category_depth(i)
                                            for i in df['node_id']]
                    df['category_target'] = [self.get_category_distance(i, tid)
                                             for i in df['node_id']]
                    # df['exploration'] = exploration
                    zipped = zip(df['node'].iloc[0:], df['node_id'].iloc[1:])
                    df['linkpos_first'] =\
                        [self.link2pos_first[a][b]
                         if b in self.link2pos_first[a] else np.NaN
                         for a, b in zipped] + [np.NaN]
                    df['linkpos_last'] =\
                        [self.link2pos_last[a][b]
                         if b in self.link2pos_last[a] else np.NaN
                         for a, b in zipped] + [np.NaN]
                    try:
                        df['linkpos_actual'] = link_data + [np.NaN]
                        zipped2 = zip(df['node'], df['linkpos_actual'])[:-1]
                        df['link_context'] = [self.get_link_context(a, b)
                                              for a, b in zipped2] + [np.NaN]

                        # to substitute first possible link use the following
                        # click_data = df['linkpos_first'].tolist()[:-1]

                        ibs = [self.ib_length[d] for d in df['node']][:-1]
                        leads = [self.lead_length[d] for d in df['node']][:-1]
                        linkpos_ib, linkpos_lead = [], []
                        for idx in range(len(link_data)):
                            c = link_data[idx]
                            i = ibs[idx]
                            l = leads[idx]
                            if np.isnan(i):
                                linkpos_ib.append(np.NaN)
                            else:
                                linkpos_ib.append(c < i)
                            if np.isnan(l):
                                linkpos_lead.append(np.NaN)
                            else:
                                linkpos_lead.append(i < c < l)
                        df['linkpos_ib'] = linkpos_ib + [np.NaN]
                        df['linkpos_lead'] = linkpos_lead + [np.NaN]
                    except ValueError, e:
                        print(e)
                        self.print_error('???')
                        pdb.set_trace()
                    df['time'] = time
                    df['time_normalized'] = time_normalized
                    df['time_word'] = time_word
                    df['time_link'] = time_link
                    time_cols = [k for k in df if 'time' in k]
                    for t in time_cols:
                        df[t] /= 1000  # convert to seconds

                    df['word_count'] = word_count[:-1] + [np.NaN]
                except KeyError, e:
                    self.print_error('key not found, dropping' + repr(e))
                    continue
                spl = self.get_spl(self.name2id[start], self.name2id[target])

                # set overall dataframe attributes
                df['successful'] = successful
                df['spl'] = spl
                df['pl'] = df.shape[0]
                df['pos'] = range(df.shape[0])
                df['distance-to-go'] = list(reversed(range(df.shape[0])))
                df['user'] = folder
                df['mission'] = mission
                df['subject'] = df['user'] + '_' + df['mission'].astype('str')
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
        db_connector.close()

    def fill_database(self):
        from modules.PageExtractor import PageExtractor
        from modules.LinkExtractor import LinkExtractor
        from modules.LinkCleaner import LinkCleaner
        from modules.PathCalculator import PathCalculator
        from modules.NodeValues import NodeValues
        from modules.TfidfCalculator import TfidfCalculator
        from modules.CategoryCalculator import CategoryCalculator

        db_connector = DbConnector('wikispeedia')

        # page_extractor = PageExtractor(db_connector)
        # page_extractor.run()
        #
        # link_extractor = LinkExtractor(db_connector)
        # link_extractor.run()
        #
        # link_cleaner = LinkCleaner(db_connector)
        # link_cleaner.run()
        #
        # path_calculator = PathCalculator(db_connector)
        # path_calculator.run()
        #
        # node_values = NodeValues(db_connector)
        # node_values.run()

        # path_calculator = TfidfCalculator(db_connector, self.plaintext_folder)
        # path_calculator.run()

        cat_calculator = CategoryCalculator(db_connector, self.html_base_folder,
                                            self.label)
        cat_calculator.run()


        db_connector.close()

    def create_dataframe(self):
        # load or compute the click data as a pandas frame
        results = []
        folder_logs = os.path.join('data', self.label, 'logfiles')
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
                if eid > 2500:
                    break
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
                spl = self.get_spl(self.name2id[entry[1]['start']],
                                   self.name2id[entry[1]['target']])
                if len(node) > 8 or spl != 3:
                    # for now, we only consider games < 9 hops and spl of 3
                    continue
                try:
                    node_id = [self.name2id[n] for n in node]
                    degree_out = [self.id2deg_out[i] for i in node_id]
                    degree_in = [self.id2deg_in[i] for i in node_id]
                    ngram_query = [self.ngram.get_frequency('query', n)
                                    for n in node]
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
                    linkpos_last = [self.link2pos_last[a][b]
                                    for a, b in zipped] + [np.NaN]
                    zipped2 = zip(node, linkpos_first)[:-1]
                    link_context = [self.get_link_context(a, b)
                                    for a, b in zipped2] + [np.NaN]
                    click_data = linkpos_first[:-1]

                    ibs = [self.ib_length[d] for d in node][:-1]
                    leads = [self.lead_length[d] for d in node][:-1]
                    linkpos_ib, linkpos_lead = [], []
                    for idx in range(len(click_data)):
                        c = click_data[idx]
                        i = ibs[idx]
                        l = leads[idx]
                        if np.isnan(i):
                            linkpos_ib.append(np.NaN)
                        else:
                            linkpos_ib.append(c < i)
                        if np.isnan(l):
                            linkpos_lead.append(np.NaN)
                        else:
                            linkpos_lead.append(i < c < l)
                    linkpos_ib = linkpos_ib + [np.NaN]
                    linkpos_lead = linkpos_lead + [np.NaN]
                    word_count = [self.length[a] if a in self.length else np.NaN
                                  for a in node][:-1] + [np.NaN]
                except KeyError:
                    continue
                data = zip(node, node_id, degree_out, degree_in,
                           ngram_query, spl_target, tfidf_target,
                           category_depth, category_target,
                           linkpos_first, linkpos_last, link_context,
                           linkpos_ib, linkpos_lead, word_count
                )
                columns = ['node', 'node_id', 'degree_out', 'degree_in',
                           'ngram_query', 'spl_target', 'tfidf_target',
                           'category_depth', 'category_target',
                           'linkpos_first', 'linkpos_last', 'link_context',
                           'linkpos_ib', 'linkpos_lead', 'word_count'
                ]
                df = pd.DataFrame(data=data, columns=columns)

                # set overall dataframe attributes
                df['successful'] = successful
                df['spl'] = spl
                df['pl'] = df.shape[0]
                df['pos'] = range(df.shape[0])
                df['distance-to-go'] = list(reversed(range(df.shape[0])))
                df['subject'] = eid
                results.append(df)

        data = pd.concat(results)
        self.save_data(data)


if __name__ == '__main__':

    # Cached.clear_cache()

    for wg in [
        # WIKTI(),
        Wikispeedia(),
    ]:
        with wg:
            # wg.create_dataframe()
            # wg.compute_tfidf_similarity()
            wg.fill_database()
            # wg.get_tfidf_similarity(1, 10)

