import logging

import networkx as nx
from typing import List

from lifelike_gds.arango_network.config_utils import read_config
import pandas as pd
from arango import ArangoClient

class Database:
    def __init__(self, collection, dbname=None, uri=None, username=None, password=None):
        self.collection = collection
        if (not uri) or (not username) or (not password):
            # if not specified, use settings from lifelike_gds/arango_network/config.yml file
            config = read_config()
            if not uri:
                uri = config['arango']['uri']
            if not username:
                username = config['arango']['user']
            if not password:
                password = config['arango']['password']
            if not dbname:
                dbname = config['arango']['dbname']
        self.driver = ArangoClient(hosts=uri, verify_override=False)
        self.db = self.driver.db(dbname, username=username, password=password)
        # Check if connection works
        self.db.collections()

    def close(self):
        self.driver.close()

    def run_query(self, query, **kwparameters):
        cursor = self.db.aql.execute(query, bind_vars=kwparameters)
        result = [r for r in cursor]
        return result

    def get_dict(self, query, **kwparameters):
        results = self.run_query(query, **kwparameters)
        records = [dict(record) for record in results]
        return records

    def get_dataframe(self, query: str, **kwparameters) -> pd.DataFrame:
        results = self.run_query(query, **kwparameters)
        df = pd.DataFrame(results)
        return df

    def get_raw_value(self, query: str, **kwparameters):
        result = self.run_query(query, **kwparameters)
        return result

    def get_single_value(self, query: str, **kwparameters):
        cursor = self.db.aql.execute(query, bind_vars=kwparameters)
        result = cursor.next()
        del cursor
        return result

    def get_query_values(self, query: str, **kwparameters):
        """
        returns: list of values lists
        """
        return self.run_query(query, **kwparameters)

    def get_nodes_by_node_ids(self, id_list: List[int]):
        """
        Get list of nodes by arango id
        Args:
            id_list: list of node ids (arango id)
        Returns:
        """
        query = f"""
        FOR n IN {self.collection}
            FILTER TO_NUMBER(n._key) in @ids
        """ + """
            RETURN {n: n}
        """
        return self.get_raw_value(query, ids=id_list)

    def get_nodes_by_attr(self, attr_values: [], attr_name, node_label):
        """
        Given list of values, return list of Node objects
        """
        query = f"""
        FOR n in {self.collection}
            FILTER n.{attr_name} in @values
        """ + """
            RETURN n
        """
        return self.get_raw_value(query, values=attr_values)

    def get_currency_nodes(self):
        query = f"""
        FOR n IN {self.collection} 
            FILTER "CurrencyMetabolite" in n.labels OR "SecondaryMetabolite" in n.labels
        """ + """
            RETURN DISTINCT {n: n}
        """
        return self.get_raw_value(query)

    def _get_where_stmt(self, rels=True, exclude_nodes=True, include_nodes=False):
        wheres = list()
        if rels:
            wheres.append("all(r in relationships(p) where type(r) in $rels)")
        if exclude_nodes:
            wheres.append("none(n in nodes(p) where id(n) in $exclude_ids)")
        if include_nodes:
            wheres.append("any(n in nodes(p) where id(p) in $include_ids)")
        where_stmt = ""
        if wheres:
            where_stmt = f"where {' AND '.join(wheres)}"
        return where_stmt

    def _get_shortest_paths_match_stmt(self):
        query_stmt = """
                MATCH(n1), (n2) where id(n1) in $source_ids and id(n2) in $target_ids
                with n1, n2 match p=allShortestPaths((n1)-[*]->(n2))
                """
        return query_stmt

    def get_shortest_path_len(self, sources: [], targets: [], rels=[], exclude_nodes=[], include_nodes=[]):
        source_ids = [n.id for n in sources]
        target_ids = [n.id for n in targets]
        exclude_ids = [n.id for n in exclude_nodes]
        include_ids = [n.id for n in include_nodes]
        return_stmt = """
        return n1.displayName as sourceName, n2.displayName as targetName, length(p) as shortestPathLen;  
        """
        query = self._get_shortest_paths_match_stmt() + self._get_where_stmt(len(rels)>0, len(exclude_nodes)>0, len(include_nodes)>0) + return_stmt
        # print(query)
        return self.get_dataframe(query, source_ids=source_ids, target_ids=target_ids, rels=rels,
                                  exclude_ids=exclude_ids, include_ids=include_ids)

    def get_shortest_paths(self,  sources: [], targets: [], rels=[], exclude_nodes=[], include_nodes=[]):
        """
        Query all shortest paths using source and target ids. Make sure that the input Nodes were from the same arango database,
        otherwise, the id's will not match properly
        """
        source_ids = [n.id for n in sources]
        target_ids = [n.id for n in targets]
        exclude_ids = [n.id for n in exclude_nodes]
        include_ids = [n.id for n in include_nodes]
        return_stmt = """
                return p;
                """
        query = self._get_shortest_paths_match_stmt() + self._get_where_stmt(len(rels)>0, len(exclude_nodes)>0, len(include_nodes)>0) + return_stmt
        # print(query)
        return self.get_raw_value(query, source_ids=source_ids, target_ids=target_ids, rels=rels,
                                  exclude_ids=exclude_ids, include_ids=include_ids)

    def add_shortest_paths_nodes_rels_to_nx(self, D: nx.DiGraph, sources: [], targets: [], rels=[],
                                            exclude_nodes: [] = [], include_nodes: [] = []):
        """
        Query all shortest paths using source and target ids.
        Make sure that the input Nodes were from the same arango database, otherwise, the id's will not match properly.
        Get unique node ids and relationships from the paths.
        Those can be used for build a light-weight networkx graph
        """
        source_ids = [n.id for n in sources]
        target_ids = [n.id for n in targets]
        exclude_ids = [n.id for n in exclude_nodes]
        include_ids = [n.id for n in include_nodes]
        node_return = """
        with nodes(p) as nodes unwind nodes as n return distinct id(n) as node_id;
        """
        rel_query = "match (n)-[r]->(m) where id(n) in $node_ids and id(m) in $node_ids "
        if rels:
            rel_query += "and type(r) in $rels "
        rel_query += "return id(n) as source, id(m) as target, type(r) as type"

        node_query = self._get_shortest_paths_match_stmt() + self._get_where_stmt(len(rels) > 0, len(exclude_nodes) > 0,
                                                                             len(include_nodes) > 0) + node_return
        logging.info("node query:" + node_query)
        node_data = self.get_dataframe(node_query, source_ids=source_ids, target_ids=target_ids, rels=rels,
                                  exclude_ids=exclude_ids, include_ids=include_ids)
        nodes = [n for n in node_data['node_id']]
        D.add_nodes_from(nodes)

        rel_data = self.get_dataframe(rel_query, node_ids=nodes, rels=rels)
        for index, row in rel_data.iterrows():
            D.add_edge(row['source'], row['target'], **{"label": row['type']})
        logging.info(nx.info(D))

    def export_json(self, filename, query):
        """
        set config:
        apoc.export.file.enabled=true
        apoc.import.file.use_arango_config=false

        query example:
        CALL apoc.export.json.all("all.json",{useTypes:true});

        MATCH (nod:db_RegulonDB)
        MATCH (:db_RegulonDB)-[rels]->(:db_RegulonDB)
        WITH collect(nod) as a, collect(rels) as b
        CALL apoc.export.json.data(a, b, "/yourfolder/regulondb.json", null)
        YIELD file, source, format, nodes, relationships, properties, time, rows, batchSize, batches, done, data
        RETURN file, source, format, nodes, relationships, properties, time, rows, batchSize, batches, done, data
        @param query
        """


class GraphSource:
    def __init__(self, database: Database, node_label_prop='displayName'):
        self.database = database
        self.node_label_prop = node_label_prop

    @classmethod
    def get_node_name(cls, node):
        pass

    @classmethod
    def get_node_desc(cls, node):
        pass

    def set_nodes_description(self, arango_nodes, D):
        pass

    def set_edges_description(self, arango_edges, D):
        pass

    @classmethod
    def set_edge_description(cls, D, startNode, endNode, edgeType, key=None):
        pass

    def retrieve_node_properties(self, graph):
        pass

    def initiate_trace_graph(self, tracegraph, exclude_currency=True):
        pass

    def load_graph_to_tracegraph(self, tracegraph, exclude_ndoes = None):
        pass

    def get_node_data_for_excel(self, node_ids:[]):
        pass

