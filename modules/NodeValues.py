import snap


class NodeValues:
    db_connection = None

    def __init__(self, _db_connection):
        self.db_connection = _db_connection

    def run(self):

        print("RUNNING NODE VALUES")
        existing_pages = list()

        wiki_graph = snap.TNGraph.New()

        max_id = self.db_connection.execute('SELECT MAX(id) '
                                            'AS max_id '
                                            'FROM pages '
                                            'WHERE 1 ',
                                            (),
                                            "SELECT"
                                            )[0]['max_id']
        print("Max ID: " + str(max_id))

        for i in range(1, max_id+1):
            page_found = bool(self.db_connection.execute('SELECT COUNT(id) AS page_found '
                                                         'FROM pages '
                                                         'WHERE id = %s '
                                                         'LIMIT 1',
                                                         i,
                                                         "SELECT"
                                                         )[0]["page_found"])

            if page_found:
                existing_pages.append(str(i))
                wiki_graph.AddNode(i)

        edges = self.db_connection.execute("SELECT DISTINCT id,page_id,linked_page_id "
                                           "FROM links "
                                           "ORDER BY id",
                                           (),
                                           "SELECT")

        for edge in edges:
            wiki_graph.AddEdge(edge['page_id'], edge['linked_page_id'])

        for node in wiki_graph.Nodes():
            self.db_connection.execute("INSERT INTO "
                                       "node_data (node_id,degree,in_degree,out_degree) "
                                       "VALUES (%s,%s,%s,%s)",
                                       (node.GetId(),
                                        node.GetInDeg() + node.GetOutDeg(),
                                        node.GetInDeg(),
                                        node.GetOutDeg()),
                                       "INSERT")

        pageranks = snap.TIntFltH()
        snap.GetPageRank(wiki_graph, pageranks)
        for node in pageranks:
            print "[Pagerank] " + str(node) + " => " + str(pageranks[node])
            self.db_connection.execute("UPDATE node_data "
                                       "SET pagerank = %s "
                                       "WHERE node_id = %s",
                                       (pageranks[node], node),
                                       "INSERT")

        self.db_connection.commit()