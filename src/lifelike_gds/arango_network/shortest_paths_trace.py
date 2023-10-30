from lifelike_gds.arango_network.trace_graph_nx import TraceGraphNx
from lifelike_gds.network.trace_utils import add_trace_network
import logging


class ShortestPathTrace(TraceGraphNx):
    def __init__(self, graphsource, multigraph=True):
        super().__init__(graphsource, True, multigraph)

    def add_shortest_paths(self, sources: str, targets: str, sources_as_query=True, shortest_paths_plus_n=0):
        source_name = self.graph.get_node_set_name(sources)
        target_name = self.graph.get_node_set_name(targets)
        if sources_as_query:
            query = sources
        else:
            query = targets
        has_paths = False
        for n in range(0, shortest_paths_plus_n+1):
            plus_n = ''
            if n > 0:
                plus_n = f'+{n}'
            network_name = f"Shortest{plus_n} paths from {source_name} to {target_name}"
            networkIdx, num_paths = add_trace_network(
                self.graph,
                sources,
                targets,
                name=network_name,
                query=query,
                shortest_paths_plus_n=n
            )
            if num_paths > 0:
                has_paths = True
            logging.info(f'add {network_name}: {num_paths} paths')
            if networkIdx is not None:
                self.add_graph_description(f'{network_name}: {num_paths} paths')
        return has_paths
