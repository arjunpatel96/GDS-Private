import os
import pandas as pd

# Functions to get nodes in different ways
def get_nodes_by_identity_from_file(database, csv_file_ref, id_name, id_column, sep=','):
    df = pd.read_csv(csv_file_ref, sep=sep, dtype='str')
    ids = [n for n in df[id_column]]
    nodes = database.get_nodes_by_attr(ids, id_name)
    print('file_rows:', len(df), ', nodes matched:', len(nodes))
    return nodes


def get_chemical_nodes_by_chebi(database, csv_file_ref, chebi_id_column, sep=','):
    df = pd.read_csv(csv_file_ref, sep=sep, dtype='str')
    ids = [n for n in df[chebi_id_column]]
    nodes = database.get_entity_nodes_by_chebi_ids(ids)
    print('file_rows:', len(df), ', nodes matched:', len(nodes))
    return nodes


def get_protein_nodes_by_gene_id(database, csv_file_ref, gene_id_column, sep=','):
    df = pd.read_csv(csv_file_ref, sep=sep, dtype='str')
    ids = [n for n in df[gene_id_column]]
    nodes = database.get_entity_nodes_by_gene_ids(ids)
    print('file_rows:', len(df), ', nodes matched:', len(nodes))
    return nodes


def get_reference_nodes_by_chebi(database, csv_file_ref, chebi_id_column, sep=','):
    df = pd.read_csv(csv_file_ref, sep=sep, dtype='str')
    ids = [n for n in df[chebi_id_column]]
    nodes = database.get_reference_nodes_by_chebi_ids(ids)
    print('file_rows:', len(df), ', nodes matched:', len(nodes))
    return nodes


def get_reference_nodes_by_gene_id(database, csv_file_ref, gene_id_column, sep=','):
    df = pd.read_csv(csv_file_ref, sep=sep, dtype='str')
    ids = [n for n in df[gene_id_column]]
    nodes = database.get_reference_nodes_by_gene_ids(ids)
    print('file_rows:', len(df), ', nodes matched:', len(nodes))
    return nodes

def get_nodes_by_stId(database, stIds):
    return database.get_nodes_by_attr(stIds, 'stId')

# Function to get the summation text

def write_summation_text_for_graph(tracegraph, outfilename):
    """
    extract summation text from given graph, and write to outfile (text file).
    """
    summation_file = os.path.join(tracegraph.datadir, outfilename)
    D = tracegraph.graph
    with open(summation_file, 'w', encoding="utf-8") as f:
        for n in D.nodes():
            if "summation" in D.nodes[n]:
                f.write(D.nodes[n]['displayName'] + "\n")
                f.write(D.nodes[n]['summation'] + "\n\n")

# Funcrions for shortest paths.       

def write_shortest_paths(
    tracegraph, 
    source_name, 
    source_nodes, 
    target_name, 
    target_nodes,
    summation=True
    ):
    """
    Simplest way to generate shortest paths. 
    Return all paths from all source nodes to all target_nodes in one trace graph
    """
    tracegraph.graph = tracegraph.orig_graph.copy()
    tracegraph.set_node_set_from_arango_nodes(
        source_nodes, source_name, source_name
    )
    tracegraph.set_node_set_from_arango_nodes(
        target_nodes, target_name, target_name
    )
    tracegraph.add_graph_description('Reactome')
    source_as_query = len(source_nodes) > len(target_nodes)
    ok = tracegraph.add_shortest_paths(source_name, target_name, source_as_query)
    if ok:
        graphfile = f"Shortest_paths_from_{source_name}_to_{target_name}.graph"
        summation_file = f"Summation_from_{source_name}_to_{target_name}.txt"
        tracegraph.write_to_sankey_file(graphfile)
        if summation:
            write_summation_text_for_graph(tracegraph, summation_file)
    else:
        print(f"No paths found from {source_name} to {target_name}")

# Functions for radiate traces



def get_selected_nodes(file_ref, sheet_name):
    """
    Read file for nodes selection.  The file was generated from radiate analysis but contains user's node selection. 
    Any columns after 'nReach' or 'rev_nReach' will be scanned for value '1' as selected rows.
    Users can use the column name to specific the selected nodes, such as 'selected_genes', 'selected_compounds' 

    file_ref: file path or url with nodes selection based on radiate analysis
    sheet_name: 'pageranks' or 'reverse pageranks'
    return dict with column name as key, and selected node eids as value
    """
    df = pd.read_excel(file_ref, sheet_name)
    colnames = [c for c in df.columns]
    if 'nReach' in colnames:
        select_cols = colnames[colnames.index('nReach') + 1 :]
    else:
        select_cols = colnames[colnames.index('rev_nReach') + 1 :]
    selected_nodes = dict()
    for c in select_cols:
        mydf = df[df[c] == 1]
        selected_nodes[c] = [id for id in mydf['stId']]
    print(f'selected {sheet_name} nodes:\n', selected_nodes)
    return selected_nodes


def export_radiate_analysis(
    tracegraph,
    source_name,
    source_nodes,
    exclude_sources_from_file=False,
    rows_export=4000,
):
    """
    Perform radiate analysis from the given source_nodes, and export pageranks and rev_pageranks
    into excel file. The excel file contains two tabs, one for pageranks and one for reverse pageranks.
    The data are sorted by pagerank (or rev_pagerank)
    rows_export: define the top ranked rows exported into file
    """
    tracegraph.graph = tracegraph.orig_graph.copy()
    tracegraph.set_node_set_from_arango_nodes(
        source_nodes, source_name, source_name
    )
    outfile_name = f"Radiate_analysis_for_{source_name}.xlsx"
    tracegraph.export_pagerank_data(
        source_name,
        outfile_name,
        direction='both',
        num_nodes=rows_export,
        exclude_sources=exclude_sources_from_file,
    )




def export_radiate_traces(
    database,
    tracegraph,
    source_name,
    source_nodes,
    forward_selection: dict,
    reverse_selection: dict,
    summation=True
):
    """
    Exports reverse and forward traces to a file
    source_name: the data set name for radiate analysis
    source_nodes: list of source nodes used for radiate analysis
    forward_selection: dict for col_name:eids for nodes selected based on pageranks
    reverse_selection: dict for col_name:eids for nodes selected based on rev_pageranks
    """
    tracegraph.graph = tracegraph.orig_graph.copy()
    tracegraph.set_node_set_from_arango_nodes(
        source_nodes, source_name, source_name
    )

    # set pagerank or rev_pagerank property
    pagerank_prop = 'pagerank'
    rev_pagerank_prop = 'rev_pagerank'

    if forward_selection:
        tracegraph.set_pagerank(source_name, pagerank_prop, False)

    if reverse_selection:
        tracegraph.set_pagerank(source_name, rev_pagerank_prop, True)

    # add graph description
    tracegraph.add_graph_description('Reactome')

    # add forward traces
    if forward_selection:
        for k, v in forward_selection.items():
            selected_nodes = database.get_nodes_by_attr(v, 'stId')
            nodeset_name = 'forward ' + k
            tracegraph.set_node_set_from_arango_nodes(
                selected_nodes, nodeset_name, nodeset_name
            )

            # add traces from sources to each selected nodes
            tracegraph.add_traces_from_sources_to_each_selected_nodes(
                selected_nodes,
                source_name,
                weighted_prop=pagerank_prop,
                selected_nodes_name=nodeset_name,
            )

            # add traces from sources to all selected nodes
            tracegraph.add_trace_from_sources_to_all_selected_nodes(
                nodeset_name,
                source_name,
                weighted_prop=pagerank_prop,
                trace_name=f'Forward combined {k}',
            )

    # add reverse traces
    if reverse_selection:
        for k, v in reverse_selection.items():
            selected_nodes = database.get_nodes_by_attr(v, 'stId')
            nodeset_name = 'reverse ' + k
            tracegraph.set_node_set_from_arango_nodes(
                selected_nodes, nodeset_name, nodeset_name
            )

            # add traces from each selected nodes to SOURCE_SET genes
            tracegraph.add_traces_from_each_selected_nodes_to_targets(
                selected_nodes,
                source_name,
                weighted_prop=rev_pagerank_prop,
                selected_nodes_name=nodeset_name,
            )

            # add traces from all reverse-selected nodes to SOURCE_SET
            tracegraph.add_trace_from_all_selected_nodes_to_targets(
                nodeset_name,
                source_name,
                weighted_prop=rev_pagerank_prop,
                trace_name=f"Reverse combined {k}",
            )
    # If the trace contains only one direction we can get the summation
    # write all traces into one graph file
    if forward_selection and reverse_selection:
        graph_file = f'Radiate_traces_for_{source_name}.graph'
        tracegraph.write_to_sankey_file(graph_file)
    elif forward_selection:
        graph_file = f'Forward_traces_for_{source_name}.graph'
        tracegraph.write_to_sankey_file(graph_file)
        if summation:
            summation_file = f'Forward_summation_for_{source_name}.txt'
            write_summation_text_for_graph(tracegraph, summation_file)
    elif reverse_selection:
        graph_file = f'Reverse_traces_for_{source_name}.graph'
        tracegraph.write_to_sankey_file(graph_file)
        if summation:
            summation_file = f'Forward_summation_for_{source_name}.txt'
            write_summation_text_for_graph(tracegraph, summation_file) 
    else:
        print('No selected nodes')    
