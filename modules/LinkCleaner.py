class LinkCleaner:
    db_connection = None

    def __init__(self, _db_connection):
        self.db_connection = _db_connection

    def run(self):
        print("RUNNING LINK-CLEANER")

        max_id = self.db_connection.execute("SELECT MAX(id) AS max_id FROM pages WHERE 1 ",
                                            (),
                                            "SELECT"
                                            )[0]['max_id']
        print("Max ID: " + str(max_id))

        for i in range(1, max_id + 1):
            page_found = bool(self.db_connection.execute("SELECT COUNT(id) AS page_found FROM pages WHERE id = %s",
                                                         i,
                                                         "SELECT"
                                                        )[0]["page_found"])
            if not page_found:
                print("Removing indices: " + str(i))
                self.db_connection.execute("DELETE FROM links WHERE (page_id = %s)",
                                           i,
                                           "DELETE"
                                           )

                self.db_connection.execute("DELETE FROM links WHERE (linked_page_id = %s)",
                                           i,
                                           "DELETE"
                                           )

                self.db_connection.execute("DELETE FROM category_pages WHERE (page_id = %s)",
                                           i,
                                           "DELETE"
                                           )

        if self.db_connection.db == 'wikispeedia':
            # remove links to the following article from the database:
            # Wikipedia:Text of the GNU Free Documentation License
            name = "Wikipedia:Text of the GNU Free Documentation License"
            wid = self.db_connection.execute('''SELECT id FROM pages
                                             WHERE name = %s''',
                                             name)
            # import pdb; pdb.set_trace()
            self.db_connection.execute('''DELETE FROM links
                                       WHERE linked_page_id = %s''',
                                       wid[0]['id'])

        self.db_connection.commit()
