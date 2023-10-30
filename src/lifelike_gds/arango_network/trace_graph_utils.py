import logging

import pandas as pd

from lifelike_gds.network.graph_io import read_gpickle, serializable_node_link_data, write_json
from lifelike_gds.network.graph_algorithms import add_influence_contribution
import networkx as nx
import sys


def convert_gpickle_to_json(gpicklefile):
    D = read_gpickle(gpicklefile)
    jsonfile = gpicklefile.replace('.gpickle.gz', '.json')
    write_json(D, jsonfile)


def write_sankey_file(filename, D):
    """
    Use indexing in "link" list instead of (u, v) or (u, v, k)
    :param filename:
    :param D: TraceGraph
    :return:
    """
    data = serializable_node_link_data(D)
    link_index(data)
    write_json(data, filename)

def write_cytoscape_file(filename, D):
    data = nx.cytoscape_data(D)
    data['data'] = {}
    for e in data['elements']['edges']:
        e['data']['source'] = str(e['data']['source'])
        e['data']['target'] = str(e['data']['target'])
    write_json(data, filename)


def link_index(data):
    """
    Use indexing in "link" list instead of (u, v) or (u, v, k)
    :param data: node_link_data dict representation of graph
    :return:
    """
    if data["multigraph"]:
        edge2index = {
            (l["source"], l["target"], l["key"]): i for i, l in enumerate(data["links"])
        }
    else:
        edge2index = {
            (l["source"], l["target"]): i for i, l in enumerate(data["links"])
        }

    for tn in data["graph"]["trace_networks"]:
        for t in tn["traces"]:
            # they should be unique. Use list since it is json serializable
            t["edges"] = [edge2index[e] for e in t["edges"]]

    if data["multigraph"]:
        # the "key" entry is redundant now
        for link in data["links"]:
            del link["key"]


def add_pagerank(
    D,
    personalized_node_set: str,
    personalization:dict = None,
    pagerank_prop:str = None,
    reverse=False,
    contribution=False,
    tol=1e-7
):
    _D = D.reverse(copy=False) if reverse else D
    method = "scipy" if D.is_multigraph() else "iteration"
    if not pagerank_prop:
        pagerank_prop = f'pagerank_{personalized_node_set}'
        if reverse:
            pagerank_prop = 'rev_' + pagerank_prop
    df = pagerank_influence(_D, personalized_node_set, personalization, method=method, tol=tol)
    pageranks = {row['node']: row['pagerank'] for index, row in df.iterrows()}
    nstarts = {row['node']: row['nstart'] for index, row in df.iterrows()}

    nx.set_node_attributes(D, pageranks, pagerank_prop)
    filtered_nstart = {k: v for k, v in nstarts.items() if v > 0}
    nx.set_node_attributes(D, filtered_nstart, 'start_val')
    if contribution:
        add_influence_contribution(D, reverse=reverse, weight=pagerank_prop, **{f"{pagerank_prop}_contribution": pagerank_prop})


def pagerank_influence(D, sources_name: str, personalization: dict=None, weight=None, method="iteration", tol=1e-6):
    """
    Modified pagerank centrality measuring how much nodes are influenced by "start_nodes"
    :param D: DirectedGraph
    :param sources_name: node set key for personalized nodes
    :param personalization: The "personalization vector" consisting of a dictionary with a
      key some subset of graph nodes and personalization value each of those.
    :param weight: Edge data key to use as weight. If None weights are set to 1.
    :param method: one of 'iteration', 'numpy', or 'scipy'.
    :param tol: tolerance.
    :return: list of level of influence for each node
    """
    sources = D.node_set(sources_name) & D.nodes
    nstart = {v: 0 for v in D}
    for v in sources:
        nstart[v] = 1
    if personalization:
        for v, val in personalization.items():
            try:
                assert v in nstart, f"personalization node {v} not in graph"
            except AssertionError as e:
                logging.exception(e)
            nstart[v] = val
    if method.startswith("iter"):
        pageranks = nx.pagerank(
            D,
            personalization=nstart,
            nstart=nstart,
            weight=weight,
            max_iter=500,
            tol=tol,
        )
    elif method.startswith("num"):
         pageranks = nx.pagerank_numpy(D, personalization=nstart, weight=weight)
    elif method.startswith("sci"):
         pageranks = nx.pagerank_scipy(
            D, personalization=nstart, nstart=nstart, weight=weight, tol=tol
        )
    df1 = pd.DataFrame(list(pageranks.items()), columns=['node', 'pagerank'])
    df2 = pd.DataFrame(list(nstart.items()), columns=['node', 'nstart'])
    df = pd.merge(df1, df2, on='node', how='outer')
    # df['scaled_pagerank'] = df['pagerank'] * 100
    return df


def set_nReach(D, sources:str, reverse=False):
    """
    set a node property to indicate if number of source or target nodes that can be reached. either sources or targets must be given.
    :param D:
    :param sources: source node set key
    :return: node set
    """
    node_set_name = D.get_node_set_name(sources)
    node_set = D.node_set(sources)
    nReach_prop = f'nReach'
    if reverse:
        nReach_prop = 'rev_' + nReach_prop
    prop_name = f"number of {node_set_name} nodes that can "
    if reverse:
        prop_name += "be reached from this node"
    else:
        prop_name += "reach to this node"
    D.name_node_props(**{nReach_prop: prop_name})
    if reverse:
        D = D.reverse(copy=False)
    nReach = {n:0 for n in D}
    for s in set(D).intersection(node_set):
        reach = nx.single_source_shortest_path(D, s)
        for n in reach.keys():
            nReach[n] += 1
    D.set(**{nReach_prop: nReach})

def set_intersection_pagerank(D, source_pagerank_name, target_rev_pagerank_name, intersect_pagerank_name=None):
    """
    Calculate intersection pagerank using using formulas p1*p2/(p1+p2-p1*p2) where pi is source pagerank,
    p2 is target rev_pagerank.
    """
    source_pageranks = D.getd(source_pagerank_name)
    target_rev_pageranks = D.getd(target_rev_pagerank_name)
    assert len(source_pageranks) > 0, "No pagerank node property found for sources"
    assert (
        len(target_rev_pageranks) > 0
    ), "No reverse pagerank node property found for targets"
    df = pd.DataFrame(dict(
        source_pagerank=source_pageranks,
        target_pagerank=target_rev_pageranks,
    ))
    p1 = df['source_pagerank']
    p2 = df['target_pagerank']
    df['inter_pagerank'] = p1*p2/(p1+p2-(p1*p2))
    df.index.name = 'node'
    df.reset_index(inplace=True)
    if not intersect_pagerank_name:
        intersect_pagerank_name = 'inter_pagerank'
    nx.set_node_attributes(D, {row['node']:row["inter_pagerank"] for index, row in df.iterrows()}, intersect_pagerank_name)


def set_intersection_pagerank_christ(D, sources, targets):
    """
    If pagerank is already added you can add the "intersection" pagerank which is the max of pagerank rank personalized to sources and reverse personalized to targets.
    :param D: (Multi)DirectedGraph
    :param sources: str name of source set to name the new node properties and to look for the node property f"{sources}_pagerank"
    :param targets: str name of source set to name the new node properties and to look for the node property f"{targets}_pagerank"
    :return: keys for new node properties
    """
    source_pageranks = D.getd(f"{sources}_pagerank")
    target_rev_pageranks = D.getd(f"{targets}_rev_pagerank")
    assert len(source_pageranks) > 0, "No pagerank node property found for sources"
    assert (
        len(target_rev_pageranks) > 0
    ), "No reverse pagerank node property found for targets"


def k_shortest_paths(G, source:int, target:int, k:int, weight=None):
    paths = []
    n = 1
    try:
        for p in nx.shortest_simple_paths(G, source, target, weight):
            if n > k:
                break
            paths.append(p)
            n += 1
    except nx.exception.NetworkXNoPath:
        return []
    return paths

def all_node_maxsum_paths(D, sources, targets, node_weight):
    """
    Get all shortest paths weighted by a given node property along the path where a higher value in the property is a better connection.
    :param D: directed graph
    :param sources: source node set []
    :param targets: target node set []
    :param node_weight: the node property key
    :return:
    """
    # add temp edge weight. This temp property approach is way faster than using weight=lambda u, v, d: ...
    tempkey = node_weight + "_maxsum"
    _D = D.copy(as_view=False)
    set_edge_weight_by_source_node_weight(_D, tempkey)
    paths = all_shortest_paths(
        _D, sources, targets, weight=tempkey
    )
    # still necessary to delete the tempkey as the dicts might the same as for D
    remove_edge_prop(D, tempkey)
    return paths

def set_edge_weight_by_source_node_weight(D, node_weight_prop, edge_weight_prop, inverse=True):
    """
    Set all edge
    Args:
        D: nx.DiGraph
        node_weight_prop: node property name
        edge_weight_prop: edge property name for weight.
        inverse: if true, set edge_weight_prop to be the inversed source node weight
    Returns:
    """
    for u, v, d in D.edges(data=True):
        u_wt = D.nodes[u].get(node_weight_prop, 0)

        if inverse:
            d[edge_weight_prop] = (1 / u_wt) if u_wt > 0 else sys.maxsize
        else:
            d[edge_weight_prop] = u_wt

def set_edge_weight_by_nodes_weight(D, edge_weight_prop, start_node_prop, end_node_prop, inverse=True):
    """
        Set all edge
        Args:
            D: nx.DiGraph
            edge_weight_prop: edge property name for weight.
            start_node_prop: edge start node weight property
            end_node_prop: edge end node weight property
            inverse: if true, set edge_weight_prop to be the inversed source node weight
        Returns:
        """


def remove_edge_prop(D, edge_prop_name):
    for u, v, d in D.edges(data=True):
        del d[edge_prop_name]


def all_shortest_paths(D, sources, targets, weight=None):
    """
    :param D:
    :param sources: node set
    :param targets: node set
    :param weight:
    :return: list of node paths
    """
    return [
        p
        for s in sources
        for t in targets
        for p in _all_shortest_paths(D, s, t, weight=weight)
    ]


def single_shortest_paths(D, sources, targets):
    paths = []
    for s in sources:
        for t in targets:
            p = list(_all_shortest_paths(D, s, t))
            if len(p) > 0:
                paths.append(p[0])
    return paths


def _all_shortest_paths(D, source, target, weight=None):
    """
    :param D:
    :param source:
    :param target:
    :param weight:
    :return: generator or empty list
    """
    try:
        for p in nx.all_shortest_paths(D, source, target, weight=weight):
            yield p
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


def get_node_set_nodes(D):
    nodes = set()
    if "node_sets" in D.graph:
        for k, ns in D.graph["node_sets"].items():
            nodes.update(ns)
    return nodes




