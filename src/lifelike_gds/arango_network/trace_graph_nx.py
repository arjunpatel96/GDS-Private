import os.path
from typing import List

from lifelike_gds.arango_network.database import GraphSource
from lifelike_gds.network.groups import set_default_groups
from lifelike_gds.network.graph_utils import MultiDirectedGraph, DirectedGraph
from lifelike_gds.network.trace_utils import add_trace_network, get_traced_nodes
from lifelike_gds.arango_network.trace_graph_utils import write_sankey_file,\
    write_cytoscape_file, k_shortest_paths, single_shortest_paths, get_node_set_nodes
from lifelike_gds.network.collection_utils import dict_max_ties, dict_min_ties
import networkx as nx
import pandas as pd
import logging

from lifelike_gds.utils import get_id


class TraceGraphNx:
    def __init__(self, graphsource: GraphSource, directed=True, multigraph=True):
        """
        Create a light-weight TraceGraph using a projected graph from arango using cypher queries for nodes and relationship.
        The graph has only node ids and relationships (with type). All details need to be loaded from arango before exporting data
        to files.
        @param graphsource the graph database source e.g. BioCyc, Reactom
        @param directed the graph is directed or non-directed graph.  If non-directed, each edge has both directions
        @param multigraph if the original database has multiedges between two nodes, use multigraph = true. Otherwise create digraph
        """
        self.graphsource = graphsource
        self.directed = directed
        self.paths = []
        self.datadir = "."
        if multigraph:
            self.graph = MultiDirectedGraph(nx.MultiDiGraph())
        else:
            self.graph = DirectedGraph(nx.DiGraph())
        # save a copy of the original graph in case there are multiple analysis to perform
        self.orig_graph = self.graph

    def init_default_graph(self, exclude_currency=True, exclude_secondary=True):
        self.graphsource.initiate_trace_graph(self, exclude_currency, exclude_secondary)

    def set_datadir(self, datadir):
        self.datadir = datadir

    def add_nodes(self, node_query, **parameters):
        """
        @param node_query a query that return node id (arango-id), e.g match(n) return id(n) as node_id;
        """
        node_data = self.graphsource.database.get_dataframe(node_query, **parameters)
        nodes = [n for n in node_data['node_id']]
        self.graph.add_nodes_from(nodes)

    def add_rels(self, rel_query, **parameters):
        """
        rel_query example: match (n)-[r]->(m) return id(n) as source, id(m) as target, type(r) as type, r.weight as weight;
        @param rel_query a query that return source, target, type and optional weight
        """
        rel_data = self.graphsource.database.get_dataframe(rel_query, **parameters)
        for index, row in rel_data.iterrows():
            self.graph.add_edge(row['source'], row['target'], **{"label": row['type']})

    def add_nodes_rels_from_paths(self, paths):
        """
        add nodes and relationships to graph.
        @param paths list of arango Paths from cypher query. e.g. match p=(n)-[]->(m) return p
        """
        for p in paths:
            self.graph.add_nodes_from([n.id for n in p.nodes])
            for r in p.relationships:
                self.graph.add_edge(get_id(r.start_node), get_id(r.end_node), **{"label": r.type})

    def set_node_set(self, key: str, nodes, **meta):
        self.graph.set_node_set(key, nodes, **meta)

    def set_node_set_from_arango_nodes(self, nodes, name, desc):
        node_ids = [get_id(n) for n in nodes]
        node_set = set(node_ids).intersection(self.graph.nodes)
        # print(node_set)
        self.graph.set_node_set(name, node_set, name=name, description=desc)

    def set_node_set_for_node(self, node):
        """
        Set node set for given node, and return node set key
        Args:
            node: arango Node object
        Returns: node set key
        """
        key = f'node_{get_id(node)}'
        name = node.get('displayName')
        self.set_node_set(key, [get_id(node)], name=name, description=name)
        return key

    def add_graph_description(self, desc):
        self.graph.describe(desc)

    def set_nodes_flag(self, node_set_name:str, flag_val):
        """
        set flag value for given node set nodes. e.g. set flag='start' for starting nodes
        Args:
            node_set_name:  node set name
            flag_val: the value for property 'flag'
        Returns:
        """
        nodes = self.graph.node_set(node_set_name)
        node_vals = {n:flag_val for n in nodes}
        nx.set_node_attributes(self.graph, node_vals, 'flag')

    def get_node_label(self, node_id):
        return self.graph.nodes[node_id][self.graphsource.node_label_prop]

    def load_node_detail_from_arango(self, nodes: List[int]):
        node_ids = [n for n in nodes]
        query = """
        FOR n IN reactome
            FILTER TO_NUMBER(n._key) IN @nodes
            RETURN { n: n }
        """
        arango_nodes = self.graphsource.database.get_raw_value(query, nodes=node_ids)
        node_dict = dict()
        for node in arango_nodes:
            node_dict[get_id(node)] = node
            self.graph.add_node(get_id(node), **{**{**node},
                                            **{"labels": list(node['labels']), "label": node.get("displayName")}})


    def clean_graph(self):
        """
        Remove nodes and edges not used in the traces, that will make the graph light-weighted
        """
        node_size1 = len(self.graph)
        nodes = get_traced_nodes(self.graph)
        nodeset_nodes = get_node_set_nodes(self.graph)
        nodes.update(nodeset_nodes)
        self.graph = self.graph.keep(nodes)
        node_size2 = len(self.graph)
        logging.info(f"clean graph: number of graph nodes decreased from {node_size1} to {node_size2}")
        set_default_groups(self.graph)

    def load_graph_detail(self):
        query = f"""
        FOR n IN {self.graphsource.database.collection}
            FILTER TO_NUMBER(n._key) IN @nodes
            RETURN n
        """
        node_ids = list(self.graph.nodes())
        arango_nodes = self.graphsource.database.get_raw_value(query, nodes=node_ids)
        node_dict = dict()
        for node in arango_nodes:
            node_id = get_id(node)
            node_dict[node_id] = node
            self.graph.add_node(
                    node_id,
                    **{
                        **{ **node },
                        **{
                            "labels": list(node['labels']),
                            "label": node.get("displayName")
                        },
                        **({

                            "displayProperties": [
                                {
                                    "type": "URL",
                                    "href": node['url'],
                                    "title": "BioCyc",
                                    "description": node.get('biocyc_id', '')
                                }
                            ]
                        } if 'url' in node else {})
                    }
            )
        self.graphsource.set_nodes_description(arango_nodes, self.graph)

        if self.graph.is_multigraph():
            for u, v, k, d in self.graph.edges(data=True, keys=True):
                node1 = node_dict[u]
                node2 = node_dict[v]
                self.graphsource.set_edge_description(self.graph, node1, node2, d['label'], k)
        else:
            for u, v, d in self.graph.edges(data=True):
                node1 = node_dict[u]
                node2 = node_dict[v]
                self.graphsource.set_edge_description(self.graph, node1, node2, d['label'])

    def write_to_sankey_file(self, filename=None):
        """
        run personalized pagerank using source nodes
        @param filename sankey file name, ending with '.graph'
        """
        self.clean_graph()
        self.load_graph_detail()
        os.makedirs(self.datadir, exist_ok=True)
        write_sankey_file(os.path.join(self.datadir, filename), self.graph)

    def write_cytoscape_json(self, filename=None):
        self.clean_graph()
        self.load_graph_detail()
        write_cytoscape_file(os.path.join(self.datadir, filename), self.graph)

    def get_nodes_detail_as_dataframe(self, node_ids):
        """
        Get node properties from both tracegraph and arango database. The dataframe is used for excel export
        """
        df1 = pd.DataFrame(self.graph.get(nodes=node_ids), index=node_ids)
        df2 = self.graphsource.get_node_data_for_excel(node_ids)
        df2.set_index('id', inplace=True)
        return df2.join(df1)

    def get_most_weighted_nodes(self, weighted_prop_name, num_nodes, include_nodes: [] = None, exclude_nodes: [] = None):
        """
        get top weighted nodes based on the given weighted_prop_name.
        Example: to get top 10 pagerank nodes for given sources(node set name), the parameters will be as follows:
        weighted_prop_name = {sources}_pagerank
        num_nodes = 10
        exclude_nodes = D.node_set(sources)

        @param weighted_prop_name: node property name
        @param num_nodes: num of nodes to return
        @param include_nodes: return nodes should be in this list
        @param exclude_nodes: node list to exclude from return nodes
        """
        if include_nodes:
            nodes = set(include_nodes)
        else:
            nodes = set(self.graph.nodes)
        if exclude_nodes:
            nodes = nodes - set(exclude_nodes)
        prop_dict = self.graph.getd(weighted_prop_name, nodes=nodes)
        best_nodes = dict_max_ties(prop_dict, num_nodes)
        # list(<numpy>) !== <numpy>.tolist() - first one preserves numpy object and 2nd cast them to corresponding Python types
        return best_nodes.tolist()

    def get_least_weighted_nodes(self, weighted_prop_name, num_nodes, include_nodes: [] = None, exclude_nodes: [] = None):
        """
        get min weighted nodes based on the given weighted_prop_name.
        @param weighted_prop_name: node property name
        @param num_nodes: num of nodes to return
        @param include_nodes: return nodes should be in this list
        @param exclude_nodes: node list to exclude from return nodes
        """
        if include_nodes:
            nodes = set(include_nodes)
        else:
            nodes = set(self.graph.nodes)
        if exclude_nodes:
            nodes = nodes - set(exclude_nodes)
        prop_dict = self.graph.getd(weighted_prop_name, nodes=nodes)
        best_nodes = dict_min_ties(prop_dict, num_nodes)
        return list(best_nodes)

    def add_selected_nodes_traces_combined_network(self, selected_nodes_key: str, weight_property, sources: str,
                                          targets: str, trace_name, shortest_paths_plus_n=0):
        """
        Add traces from sources to all selected nodes, and all selected nodes to targets
        """
        if sources:
            logging.info(f"Adding trace network from {sources} to {selected_nodes_key}")
            add_trace_network(
                self.graph,
                sources,
                selected_nodes_key,
                name=f"{trace_name}. Highest influence",
                maxsum=weight_property,
                query=sources,
                shortest_paths_plus_n=shortest_paths_plus_n
            )
            add_trace_network(
                self.graph,
                sources,
                selected_nodes_key,
                name=f"{trace_name}. Shortest paths",
                query=sources,
                shortest_paths_plus_n=shortest_paths_plus_n
            )
        if targets:
            logging.info(f"Adding trace network from {selected_nodes_key} to {targets}")
            add_trace_network(
                self.graph,
                selected_nodes_key,
                targets,
                name=f"{trace_name}. Highest influence",
                maxsum=weight_property,
                query=targets,
                shortest_paths_plus_n=shortest_paths_plus_n
            )
            add_trace_network(
                self.graph,
                selected_nodes_key,
                targets,
                name=f"{trace_name}. Shortest paths",
                query=targets,
                shortest_paths_plus_n=shortest_paths_plus_n
            )

    def add_selected_nodes_trace_networks(self, selected_nodes, weight_property, trace_name_prefix,
                                          sources:str=None, targets:str=None, include_allshortest_path=True,
                                          shortest_paths_plus_n=0):
        """
        Add traces from sources to each selected nodes (arango Node), or from each selected nodes (arango Node) to targets
        Args:
            selected_nodes: list of arango Node objects
            weight_property: node weight property, used for minmax, e.g. page rank property
            trace_name_prefix: e.g. forward, reverse
            sources: source node set name
            targets: target node set name
            include_allshortest_path: if true, the shortest path traces will also be added into the graph
        Returns:

        """
        for i in range(len(selected_nodes)):
            select_key = self.set_node_set_for_node(selected_nodes[i])
            select_name = self.graph.get_node_set_name(select_key)
            tracename = f"{trace_name_prefix} #{i+1} ({select_name}). High influence using {weight_property}"
            if sources:
                source_name = self.graph.get_node_set_name(sources)
                logging.info(f"Adding trace network {source_name} to {select_name} #{i+1}")
                add_trace_network(
                    self.graph,
                    sources,
                    select_key,
                    name=tracename,
                    maxsum=weight_property,
                    query=sources,
                    shortest_paths_plus_n=shortest_paths_plus_n
                )
                if include_allshortest_path:
                    add_trace_network(
                        self.graph,
                        sources,
                        select_key,
                        name=f"{trace_name_prefix} #{i + 1} ({select_name}). Shortest paths.",
                        query=sources,
                        shortest_paths_plus_n=shortest_paths_plus_n
                    )

            if targets:
                target_name = self.graph.get_node_set_name(targets)
                logging.info(f"Adding trace network from {select_name} #{i+1} to {target_name}")
                tracename = f"{trace_name_prefix} #{i + 1} ({select_name}). High influence using {weight_property}"
                add_trace_network(
                    self.graph,
                    select_key,
                    targets,
                    name=tracename,
                    maxsum=weight_property,
                    query=targets,
                    shortest_paths_plus_n=shortest_paths_plus_n
                )
                if include_allshortest_path:
                    add_trace_network(
                        self.graph,
                        select_key,
                        targets,
                        name=f"{trace_name_prefix} #{i + 1} ({select_name}). Shortest paths",
                        query=targets,
                        shortest_paths_plus_n=shortest_paths_plus_n
                    )

    def add_single_shortest_path(self, sources, targets):
        source_nodes = self.graph.node_set(sources)
        target_nodes = self.graph.node_set(targets)
        paths = single_shortest_paths(self.graph, source_nodes, target_nodes)
        source_name = self.graph.get_node_set_name(sources)
        source_desc = self.graph.get_node_set_description(sources)
        target_name = self.graph.get_node_set_name(targets)
        target_desc = self.graph.get_node_set_description(targets)
        logging.info(f'num single shortest paths from {sources} to {targets}: {len(paths)}')
        query = sources
        self.graph.add_trace_network(
            sources=sources,
            targets=targets,
            node_paths=paths,
            method="min(length)",
            name=f"All paths from {source_name} to {target_name}",
            description=f"All paths from {source_desc} to {target_desc}.",
            query=query
        )
        self.add_graph_description(f"single shortest paths from {sources} to {targets}")

    def add_k_shortest_simple_paths(self, sources, targets, k=100):
        paths = []
        source_nodes = self.graph.node_set(sources)
        target_nodes = self.graph.node_set(targets)
        for s in source_nodes:
            for t in target_nodes:
                k_paths = k_shortest_paths(self.graph, s, t, k)
                for path in k_paths:
                    paths.append(path)
        self.paths.append(paths)
        source_name = self.graph.get_node_set_name(sources)
        source_desc = self.graph.get_node_set_description(sources)
        target_name = self.graph.get_node_set_name(targets)
        target_desc = self.graph.get_node_set_description(targets)
        logging.info(f'num simple paths from {sources} to {targets}: {len(paths)}')
        query = sources
        self.graph.add_trace_network(
            sources=sources,
            targets=targets,
            node_paths=paths,
            method="min(length)",
            name=f"All paths from {source_name} to {target_name}",
            description=f"All paths from {source_desc} to {target_desc}.",
            query=query
        )
        self.add_graph_description(f"all paths from {sources} to {targets}")
