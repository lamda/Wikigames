# -*- coding: utf-8 -*-

from __future__ import division, print_function

import cPickle as pickle
import os

import PySide.QtCore
import PySide.QtGui
import PySide.QtWebKit

from decorators import Cached


class WebPageSize(PySide.QtGui.QMainWindow):
    def __init__(self, qt_application, label):
        self.qt_application = qt_application
        PySide.QtGui.QMainWindow.__init__(self)
        self.web_view = PySide.QtWebKit.QWebView()
        self.setCentralWidget(self.web_view)
        self.web_view.loadFinished.connect(self._load_finished)
        self.curr_page = ''
        self.curr_width = 0
        self.server = True
        if self.server:
            self.base_url = 'http://localhost:8000/wp/'
        else:
            self.base_url = 'file:///C:/PhD/Code/Wikigames/data/' + label +\
                            '/wpcd/wp/'

    @Cached
    def get_size(self, page, width):
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

"""
use as follows (needs testing):

    regex_scroll = r"u'scroll': {u'y': (\d+), u'x': \d+}," \
                   r" u'size': {u'y': (\d+), u'x': (\d+)"
    qt_application = PySide.QtGui.QApplication(sys.argv)
    page_size = WebPageSize(qt_application, self.label)

    # get scrolling range
    idx = list(df_full[df_full['action'] == 'load'].index)
    df_groups = [df_full.loc[a:b, :]
                 for a, b in zip(idx, idx[1:])]
    exploration = [np.nan]
    for i, g in enumerate(df_groups):
        print('            ', df.iloc[i]['node'])
        slct = (g['action'] == 'scroll') | (g['action'] == 'resize')
        if len(g[slct]) == 0:
            from_index = None
            print('            ', 'from_index is None')
        else:
            from_index = g[slct].index[0]
        df_scroll = g.loc[from_index:]
        df_scroll = df_scroll.node.str.extract(regex_scroll)
        df_scroll = df_scroll.dropna()
        df_scroll.columns = ['scrolled', 'height', 'width']
        df_scroll['scrolled'] = df_scroll['scrolled'].apply(int)
        df_scroll['height'] = df_scroll['height'].apply(int)
        df_scroll['width'] = df_scroll['width'].apply(int)
        seen_log = df_scroll.loc[df_scroll['scrolled'].idxmax()]
        seen_max = page_size.get_size(df.iloc[i].node, seen_log[2])[1]
        seen = seen_log['scrolled'] + seen_log['height']
        if from_index is None:
            seen_max = seen
        exploration.append(seen / seen_max)
        print(df.iloc[0].node, seen, seen_max)

    df['exploration'] = exploration
"""
