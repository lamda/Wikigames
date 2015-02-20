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
