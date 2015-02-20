# -*- coding: utf-8 -*-

from __future__ import division, print_function

from collections import defaultdict
import io
import numpy as np
import os
import re


class CategoryCalculator:
    db_connection = None

    def __init__(self, _db_connection, _html_base_folder, label):
        self.db_connection = _db_connection
        self.html_base_folder = _html_base_folder
        self.label = label

    def run(self):
        print("RUNNING CATEGORY-CALCULATOR")

        # read plaintext files
        query = """SELECT * FROM pages ORDER BY id ASC"""
        pages = self.db_connection.execute(query)
        pids = [p['id'] for p in pages]
        pages = [re.findall(r'\\([^\\]*?)\.htm', p['link'])[0] for p in pages]

        query = """SELECT DISTINCT COLUMN_NAME
                   FROM information_schema.COLUMNS
                   WHERE TABLE_SCHEMA = '%s'
                   AND TABLE_NAME = 'node_data'""" % self.label
        columns = self.db_connection.execute(query)
        columns = [c['COLUMN_NAME'] for c in columns]
        if 'category_depth' not in columns:
            queries = [
                "CREATE TABLE node_data_new LIKE node_data",
                "ALTER TABLE node_data_new ADD COLUMN category_depth FLOAT",
                "INSERT INTO node_data_new"
                "     (id, node_id, degree, in_degree, out_degree, pagerank)"
                "    SELECT * FROM node_data",
                "RENAME TABLE node_data TO node_data_old",  # takes forever
                "RENAME TABLE node_data_new TO node_data",  # not sure why
                "DROP TABLE node_data_old"
            ]
            for query in queries:
                print(query)
                self.db_connection.execute(query)
            self.db_connection.commit()

        category = defaultdict(list)
        category_depth = defaultdict(float)

        for i, a in zip(pids, pages):
            print(i, '/', len(pids) - 1, end='\r')
            fname = os.path.join(self.html_base_folder, a[0].lower(),
                                  a + '.htm')
            try:
                with io.open(fname, encoding='utf-8') as infile:
                    data = infile.readlines()
            except UnicodeDecodeError:
                with io.open(fname) as infile:
                    data = infile.readlines()

            for line in data:
                m = re.findall(r'subject\.(.+?)\.ht', line)
                if m:
                    category_depth[i] = np.mean([(p.count('.') + 1) for p in m])
                    category[i] = [p.split('.') for p in m]
                    break
            query = """UPDATE node_data SET category_depth = %s
                       WHERE node_id = %s""" % (category_depth[i], i)
            self.db_connection.execute(query)
        self.db_connection.commit()

        category_distance = {}
        for i in pids:
            print(i, '/', len(pids) - 1, end='\r')
            category_distance[i] = {}
            for j in pids:
                if i == j:
                    category_distance[i][j] = 0
                elif i < j:
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
                        category_distance[i][j] = sum(min_dists) / num_cats
                    else:
                        # pages do not have categories
                        category_distance[i][j] = 'NULL'

        query = """CREATE TABLE IF NOT EXISTS `category_distances` (
                   `id` int(11) NOT NULL AUTO_INCREMENT,
                   `page_id` int(11) NOT NULL,
                   `target_page_id` int(11) NOT NULL,
                   `distance` float,
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
                query = """INSERT INTO category_distances
                                       (page_id, target_page_id, distance)
                           VALUES (%s, %s, %s)"""\
                           % (i, j, category_distance[i][j])
                self.db_connection.execute(query)
        self.db_connection.commit()


