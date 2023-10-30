from typing import List
import pandas as pd
from lifelike_gds.arango_network.trace_graph_nx import TraceGraphNx
from lifelike_gds.arango_network.trace_graph_utils import add_pagerank, set_nReach, set_intersection_pagerank
import logging, os


class RadiateTrace(TraceGraphNx):
    def __init__(self, graphsource, multigraph=True):
        super().__init__(graphsource, True, multigraph)

    @classmethod
    def get_pagerank_prop_name(cls, sources):
        return "pagerank"

    @classmethod
    def get_rev_pagerank_prop_name(cls, sources):
        return "rev_pagerank"

    @classmethod
    def get_intersection_rank_prop_name(cls, sources, targets):
        return "intersect_pagerank"


    def set_pagerank_and_numreach(self, sources, direction='both', personalization:dict=None, contribution=False):
        """
        Set personalized page ranks from sources, and num of reachable sources nodes for every nodes in the graph
        property names as {sources}_pagerank, {sources}_rev_pagerank,
        @param sources: node set name for sources
        @param personalization dict of node as key, initial weight of nodes as val
        @param direction: forward, reverse or both
        """
        logging.info(f"set pagerank and num reach for {sources}")
        source_set = self.graph.node_set(sources)
        has_out = self.graph.has_out(source_set)
        has_in = self.graph.has_in(source_set)
        if has_out and ((direction == 'forward') or (direction == 'both')):
            pagerank_prop = RadiateTrace.get_pagerank_prop_name(sources)
            add_pagerank(self.graph, sources, pagerank_prop=pagerank_prop, personalization=personalization, contribution=contribution)
            set_nReach(self.graph, sources)
        if has_in and ((direction == 'reverse') or (direction == 'both')):
            rev_pagerank_prop = RadiateTrace.get_rev_pagerank_prop_name(sources)
            add_pagerank(self.graph, sources, pagerank_prop=rev_pagerank_prop, personalization=personalization, reverse=True, contribution=contribution)
            set_nReach(self.graph, sources, reverse=True)
        return has_in, has_out

    def set_pagerank(self, sources, pagerank_prop:str, reverse=False, personalization:dict=None, contribution=True):
        """
        Calculate personalized pagerank from soruces. Set pagerank values to the pagerank_prop.
        Args:
            sources:  source node set name.
            pagerank_prop: pagerank property name
            reverse: if True, calculate reverse pagerank
            personalization: used for weighted pagerank analysis.
            contribution: if true, edge property is set. This is important for sankey display
        Returns:

        """
        add_pagerank(self.graph, sources, pagerank_prop=pagerank_prop,
                     personalization=personalization, reverse=reverse, contribution=contribution)

    def export_pagerank_data(
            self,
            sources,
            filename,
            sources_personalization: dict[int, int] = None,
            direction="both",
            num_nodes=3000,
            exclude_sources=True
    ):
        """
        Run personalized pagerank for given sources, and export the pagerank data to excel file
        @param sources: sources set name.  The pagerank sources name
        @param filename:
        @param sources_personalization: node id to initial weight dict
        @param direction: both, forward or reverse
        @param num_nodes: top nodes to return. if both direction, the returned nodes are the joining of both direction nodes, and could be more than 1000
        @param exclude_sources:
        """
        if sources_personalization is None:
            sources_personalization = dict()

        has_in, has_out = self.set_pagerank_and_numreach(sources, personalization=sources_personalization, direction=direction)
        pr_name = self.get_pagerank_prop_name(sources)
        pr_reach = 'nReach'
        rev_pr_name = self.get_rev_pagerank_prop_name(sources)
        rev_reach = 'rev_' + pr_reach

        excludes = []
        if exclude_sources:
            excludes = self.graph.node_set(sources)
        best_forward_nodes = []
        best_reverse_nodes = []

        if has_out and direction != 'reverse':
            # process forward pageranks
            includes = [n for n, p in self.graph.nodes(data=True) if p[pr_name] > 0]
            if len(includes)>0:
                best_forward_nodes = self.get_most_weighted_nodes(pr_name, 
                                                                  num_nodes, 
                                                                  include_nodes=includes, 
                                                                  exclude_nodes=excludes)
        if has_in and direction != 'forward':
            # process reverse pageranks
            includes = [n for n, p in self.graph.nodes(data=True) if p[rev_pr_name] > 0]
            if len(includes) > 0:
                best_reverse_nodes = self.get_most_weighted_nodes(rev_pr_name, 
                                                                  num_nodes, 
                                                                  include_nodes=includes,
                                                                  exclude_nodes=excludes)
        all_nodes = set(best_forward_nodes + best_reverse_nodes)
        df = self.get_nodes_detail_as_dataframe(list(all_nodes))
        df['select'] = ''

        filepath = os.path.join(self.datadir, filename)
        logging.info(f"export top {num_nodes} pagerank data into {filepath}")
        with pd.ExcelWriter(filepath, engine='xlsxwriter') as writer:
            if len(best_forward_nodes) > 0:
                df_forward = df[df.index.isin(set(best_forward_nodes))]
                if has_in and direction != 'forward':
                    df_forward.drop(columns=[rev_pr_name, rev_reach], inplace=True)
                df_forward.sort_values(by=[pr_name], ascending=False, inplace=True)
                df_forward.to_excel(writer, sheet_name="pageranks")
            if len(best_reverse_nodes) > 0:
                df_reverse = df[df.index.isin(set(best_reverse_nodes))]
                if has_out and direction != 'reverse':
                    df_reverse.drop(columns=[pr_name, pr_reach], inplace=True)
                df_reverse.sort_values(by=[rev_pr_name], ascending=False, inplace=True)
                df_reverse.to_excel(writer, sheet_name="reverse pageranks")

    def add_traces_from_sources_to_each_selected_nodes(self, selected_nodes: List, sources: str, weighted_prop=None,
                                                       selected_nodes_name: str = None,
                                                       include_shortest_paths=True, shortest_paths_plus_n=0):
        """
        Add individual traces from each selected node to source node sets
        Args:
            selected_nodes: list of selected arango nodes
            sources: source node set name
            weighted_prop: pagerank (or rev_pagerank) property name
            selected_nodes_name: user-defined name for the selected node set
            include_shortest_paths: if True, add all shortest paths traces, otherwise, only add influence traces
        Returns:
        """
        if not weighted_prop:
            weighted_prop = self.get_pagerank_prop_name(sources)
        prefix = 'Forward'
        if selected_nodes_name:
            prefix += ' ' + selected_nodes_name
        self.add_selected_nodes_trace_networks(selected_nodes, weighted_prop, prefix,
                                               sources, include_allshortest_path=include_shortest_paths,
                                               shortest_paths_plus_n=shortest_paths_plus_n)
        if selected_nodes_name:
            self.add_graph_description(f"Traces from {sources} to each of the {len(selected_nodes)} {selected_nodes_name} nodes;")
        else:
            self.add_graph_description(f"Traces from {sources} to each of the {len(selected_nodes)} selected nodes;")

    def add_trace_from_sources_to_all_selected_nodes(self, selected_nodeset: str,
                                                     sources: str, weighted_prop=None, 
                                                     trace_name='Forward combined',
                                                     shortest_paths_plus_n=0):
        if not weighted_prop:
            weighted_prop = self.get_pagerank_prop_name(sources)
        self.add_selected_nodes_traces_combined_network(selected_nodeset, weighted_prop, sources, None,
                                                        trace_name=trace_name, 
                                                        shortest_paths_plus_n=shortest_paths_plus_n)
        self.add_graph_description(f"Traces from {sources} to all {selected_nodeset};")

    def add_traces_from_each_selected_nodes_to_targets(self, selected_nodes: List, targets: str, weighted_prop=None,
                                                       selected_nodes_name: str = None,
                                                       include_allshortest_path=True,
                                                       shortest_paths_plus_n=0):
        """
        Add traces from each selected nodes to all the target nodes
        Args:
            selected_nodes: list of arango Node objects
            targets: target node set name
            weighted_prop: node weight property for minmax,e.g. pagerank property
            selected_nodes_name: user-defined name for the selected node set
            include_allshortest_path: if True, add shortest path traces to the graph
        Returns:

        """
        if not weighted_prop:
            weighted_prop = self.get_rev_pagerank_prop_name(targets)
        prefix = 'Reverse'
        if selected_nodes_name:
            prefix += ' ' + selected_nodes_name
        self.add_selected_nodes_trace_networks(selected_nodes, weighted_prop, prefix,
                                               targets=targets,
                                               include_allshortest_path=include_allshortest_path,
                                               shortest_paths_plus_n=shortest_paths_plus_n)
        if selected_nodes_name:
            self.add_graph_description(f"traces from each of the {len(selected_nodes)} {selected_nodes_name} nodes to {targets}")
        else:
            self.add_graph_description(f"traces from each of the {len(selected_nodes)} selected nodes to {targets}")

    def add_trace_from_all_selected_nodes_to_targets(self, selected_nodeset: str, targets: str, weighted_prop=None,
                                                     trace_name='Reverse combined', 
                                                     shortest_paths_plus_n=0):
        if not weighted_prop:
            weighted_prop = self.get_rev_pagerank_prop_name(targets)
        self.add_selected_nodes_traces_combined_network(selected_nodeset, weighted_prop, sources=None, targets=targets,
                                                        trace_name=trace_name, shortest_paths_plus_n=shortest_paths_plus_n)
        self.add_graph_description(f"traces from all {selected_nodeset} to {targets}")

    def set_intersection_pagerank(self, source_pr:str, target_rev_pr: str, intersect_pr:str):
        set_intersection_pagerank(self.graph, source_pr, target_rev_pr, intersect_pr)

    def export_intersection_pageranks(self, excel_filename, source_set: str, target_set:str, source_personalization={},
                                      target_personalization={}, num_nodes=3000, exclude_sources=True):
        self.set_pagerank_and_numreach(source_set, direction='forward', personalization=source_personalization)
        self.set_pagerank_and_numreach(target_set, direction='reverse', personalization=target_personalization)
        pr_prop = self.get_pagerank_prop_name(source_set)
        rev_pr_prop = self.get_rev_pagerank_prop_name(target_set)
        inter_pr_prop = self.get_intersection_rank_prop_name(source_set, target_set)
        self.set_intersection_pagerank(pr_prop, rev_pr_prop, inter_pr_prop)

        excluded = []
        if exclude_sources:
            excluded = list(self.graph.node_set(source_set)) + list(self.graph.node_set(target_set))

        inter_nodes = [n for n, d in self.graph.nodes(data=True) if (d[pr_prop] > 0) and (d[rev_pr_prop] > 0)]
        nodes = self.get_most_weighted_nodes(inter_pr_prop, num_nodes, include_nodes=inter_nodes, exclude_nodes=excluded)
        df = self.get_nodes_detail_as_dataframe(nodes)
        df.sort_values(by=[inter_pr_prop], ascending=False, inplace=True)
        df['select'] = ''
        filepath = os.path.join(self.datadir, excel_filename)
        print('export intersection pagerank to file ', filepath)
        df.to_excel(filepath, index=False)







