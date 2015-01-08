from HTMLParser import HTMLParser
import os
import pdb

from shared import root_path
from shared import link_prefix


class LinkParser(HTMLParser):
    current_id = None
    db_connection = None

    def find_page_id(self, _link):
        link_pre = _link[:6]
        if link_pre != "../../":
            return
        link_short = link_prefix + _link[8:]
        link_short = link_short.replace("/", os.path.sep)

        link_short = link_short.replace("%25", "%")
        pages_found = self.db_connection.execute("SELECT * FROM pages WHERE (link=%s)",
                                                 link_short,
                                                 "SELECT")

        for page_found in pages_found:
            link_amount = int(self.db_connection.execute("SELECT COUNT(id) AS amount FROM links WHERE (page_id=%s) AND (linked_page_id = %s)",
                                                         (int(self.current_id), int(page_found['id'])),
                                                         "SELECT")[0]['amount'])
            if link_amount > 0:
                self.db_connection.execute("UPDATE links SET amount=(amount+1) WHERE (page_id=%s) AND (linked_page_id = %s)",
                                           (self.current_id, page_found['id']),
                                           "INSERT")
            else:
                self.db_connection.execute("INSERT INTO links (page_id, linked_page_id) VALUES (%s, %s)",
                                           (int(self.current_id), int(page_found['id'])),
                                           "INSERT")

    def set_db_connection(self, _db_connection):
        self.db_connection = _db_connection

    def set_id(self, _id):
        self.current_id = _id

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for attr in attrs:
                if attr[0] == "href":
                    self.find_page_id(attr[1])


class LinkExtractor:
    db_connection = None

    def __init__(self, _db_connection):
        self.db_connection = _db_connection

    def run(self):
        print("RUNNING LINK-EXTRACTOR")

        # pages = self.db_connection.execute("SELECT * FROM pages WHERE 1",
        query = '''SELECT * FROM pages WHERE id not in
                   (SELECT DISTINCT page_id from links);'''
        pages = self.db_connection.execute(query, (), "SELECT")

        for page in pages:
            current_id = page['id']
            print("Currently processing #" + str(current_id) + "(" + page['name'] + ")")
            wikifile = open(root_path + page['link'][16:], "r")
            linkparser = LinkParser()
            linkparser.set_db_connection(self.db_connection)
            linkparser.set_id(current_id)
            linkparser.feed(wikifile.read())
            del linkparser
            self.db_connection.commit()
