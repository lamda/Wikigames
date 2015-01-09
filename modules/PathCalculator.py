from pygraph.classes.digraph import digraph
from pygraph.algorithms.searching import breadth_first_search
from pygraph.readwrite.dot import write

from pygraph.algorithms.minmax import shortest_path
import graphviz
import sys


class PathCalculator:
    db_connection = None

    def __init__(self, _db_connection):
        self.db_connection = _db_connection

    def run(self):
        print("RUNNING PATH-CALCULATOR")
        existing_pages = list()

        wiki_graph = digraph()

        max_id = self.db_connection.execute("SELECT MAX(id) AS max_id FROM pages WHERE 1 ",
                                            (),
                                            "SELECT"
                                            )[0]['max_id']
        print("Max ID: " + str(max_id))

        for i in range(1, max_id+1):
            page_found = bool(self.db_connection.execute("SELECT COUNT(id) AS page_found FROM pages WHERE id = %s LIMIT 1",
                                                         i,
                                                         "SELECT"
                                                         )[0]["page_found"])

            if page_found:
                wiki_graph.add_node(str(i))
                existing_pages.append(str(i))

        edges = self.db_connection.execute("SELECT DISTINCT page_id,linked_page_id FROM links", (), "SELECT")

        for edge in edges:
            wiki_graph.add_edge((str(edge['page_id']), str(edge['linked_page_id'])))

        for i in existing_pages:
            self.db_connection.commit()
            print("Processing node " + i)
            shortest_paths = shortest_path(wiki_graph, i)

            for path in shortest_paths[1]:
                self.db_connection.execute("INSERT INTO path_lengths (page_id, target_page_id, path_length) VALUES (%s, %s, %s)",
                                           (i, path, shortest_paths[1][path]),
                                           "INSERT")
