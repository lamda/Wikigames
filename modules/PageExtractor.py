# coding=UTF-8

from HTMLParser import HTMLParser
import os

from shared import root_path


class PageParser(HTMLParser):
    db_connection = None
    current_file_name = None

    data_to_store = [True, None, None]

    def store_data(self):
        if self.data_to_store[0]:

            self.db_connection.execute("INSERT INTO pages (name, link) VALUES (%s,%s)",
                                       (self.data_to_store[1], self.data_to_store[2]),
                                       "INSERT")
        self.data_to_store[0] = True

    def handle_data(self, data):
        parent_tag = HTMLParser.get_starttag_text(self)
        if (parent_tag == "<title>") and not (data.strip() == ""):
            self.data_to_store[1] = data
            self.data_to_store[2] = link_prefix + self.current_file_name

    def handle_starttag(self, tag, attrs):
        if tag == "meta":
            for attr in attrs:
                if (attr[0] == "content") and ((attr[1] == "Copyright © SOS Children") or (attr[1] == "Copyright (c) SOS Children's Villages UK")):
                    print("Skipping: " + self.current_file_name)
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
                    file_name = os.path.join(folder, current_file)
                    extension = os.path.splitext(file_name)[1]
                    if not extension == ".htm":
                       # print("Skipping non HTM: " + current_file)
                       continue
                    if current_file[:6] == "Portal":
                        print("Skipping portal " + current_file)
                        continue

                    wikifile = open(root_path + file_name, "r")
                    pageparser = PageParser()
                    pageparser.set_db_connection(self.db_connection)
                    pageparser.set_file_name(file_name)
                    pageparser.feed(pageparser.unescape(wikifile.read().decode("UTF-8")).encode("UTF-8"))
                    del pageparser
                self.db_connection.commit()
