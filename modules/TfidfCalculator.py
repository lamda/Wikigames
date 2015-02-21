# -*- coding: utf-8 -*-

from __future__ import division, print_function

import io
import os
import re

from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfCalculator:
    db_connection = None

    def __init__(self, _db_connection, _plaintext_folder):
        self.db_connection = _db_connection
        self.plaintext_folder = _plaintext_folder

    def run(self):
        print("RUNNING TFIDF-CALCULATOR")
        # read plaintext files
        query = """SELECT * FROM pages ORDER BY id ASC"""
        pages = self.db_connection.execute(query)
        pids = [p['id'] for p in pages]
        pages = [re.findall(r'\\([^\\]*?)\.htm', p['link'])[0] for p in pages]

        content = []
        for title in pages:
            with io.open(os.path.join(self.plaintext_folder, title + '.txt',),
                         encoding='utf-8') as infile:
                data = infile.read()
            content.append(data)

        # compute cosine TF-IDF similarity
        with io.open(os.path.join('data', 'stopwords.txt'), encoding='utf-8')\
                as infile:
            stopwords = infile.read().splitlines()
        tvec = TfidfVectorizer(stop_words=stopwords)
        tfidf = tvec.fit_transform(content)
        tfidf_similarity = tfidf * tfidf.T
        tfidf_similarity = tfidf_similarity.todense()

        query = """CREATE TABLE IF NOT EXISTS `tfidf_similarities` (
                   `id` int(11) NOT NULL AUTO_INCREMENT,
                   `page_id` int(11) NOT NULL,
                   `target_page_id` int(11) NOT NULL,
                   `similarity` float NOT NULL,
                   PRIMARY KEY (`id`),
                   KEY `page_id` (`page_id`)
                   ) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_bin;"""
        self.db_connection.execute(query)
        self.db_connection.commit()

        for i in pids:
            print(i, '/', len(pids), end='\r')
            for j in pids:
                if i > j:
                    continue
                query = """INSERT INTO tfidf_similarities
                                       (page_id, target_page_id, similarity)
                           VALUES (%s, %s, %s)"""\
                           % (i, j, tfidf_similarity[i-1, j-1])
                self.db_connection.execute(query)
        self.db_connection.commit()
