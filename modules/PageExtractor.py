# coding=UTF-8

from HTMLParser import HTMLParser
import os
import pdb

from shared import root_path
from shared import link_prefix


class PageParser(HTMLParser):
    db_connection = None
    current_file_name = None
    copyright = [
        'Copyright © SOS Children',
        "Copyright (c) SOS Children's Villages UK",
        'Copyright SOS Children.',
        "SOS Children: the world's largest charity for orphanned ",
    ]
    data_to_store = [True, None, None]

    def store_data(self):
        if self.data_to_store[0]:
            # if 'kashmir' in self.data_to_store[1].lower():
            #     print self.data_to_store
            #     pdb.set_trace()
            self.db_connection.execute("INSERT INTO pages (name, link) VALUES (%s,%s)",
                                       (self.data_to_store[1], self.data_to_store[2]),
                                       "INSERT")
        self.data_to_store[0] = True

    def handle_data(self, data):
        parent_tag = HTMLParser.get_starttag_text(self)
        if (parent_tag == "<title>") and not (data.strip() == ""):
            self.data_to_store[1] = data
            self.data_to_store[2] = os.path.join(link_prefix,
                                                 self.current_file_name)

    def handle_starttag(self, tag, attrs):
        if tag == 'meta':
            for attr in attrs:
                if (attr[0] == 'content') and (attr[1] in self.copyright):
                    print('Skipping: ' + self.current_file_name)
                    self.data_to_store[0] = False

    def handle_endtag(self, tag):
        if tag == "html":
            self.store_data()

    def set_db_connection(self, _db_connection):
        self.db_connection = _db_connection

    def set_file_name(self, _file_name):
        self.current_file_name = _file_name


class PageExtractor:
    db_connection = None

    def __init__(self, _db_connection):
        self.db_connection = _db_connection

    def run(self):
        print("RUNNING PAGE-EXTRACTOR")
        folders = os.listdir(root_path)

        for folder in folders:
            if not folder == "index":
                files = os.listdir(os.path.join(root_path + folder))

                for current_file in files:
                    if self.db_connection.db == 'wikispeedia' and\
                            current_file.startswith('Demographics'):
                        # skip the Demographics articles for Wikispeedia
                        # as they are never linked to and
                        # were not included in West's articles
                        print('Skipping Wikispeedia Demographics file')
                        continue
                    file_name = os.path.join(folder, current_file)
                    extension = os.path.splitext(file_name)[1]
                    if not extension == ".htm":
                        continue
                    if current_file[:6] == "Portal":
                        print("Skipping portal " + current_file)
                        continue
                    print current_file
                    wikifile = open(root_path + file_name, "r")
                    pageparser = PageParser()
                    pageparser.set_db_connection(self.db_connection)
                    pageparser.set_file_name(file_name)
                    data = wikifile.read().decode("UTF-8", 'ignore')
                    pageparser.feed(pageparser.unescape(data).encode('utf-8'))
                    del pageparser
                self.db_connection.commit()
