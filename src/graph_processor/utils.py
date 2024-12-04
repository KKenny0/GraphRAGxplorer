import json
import networkx as nx


def graphml_to_json(graphml_file):
    """
    Returns
        graph: NetworkX graph
        graph_data: {..., "graph": {"nodes": [...], "links": [...]}}
            - node: {"entity_type": ..., "description": ..., "id", ..., "source_id": ..., "clusters": ...}
            - link: {"source":..., "target":..., "description":...,  "weight":...,, "order": ..., "source_id": ...}
    """
    G = nx.read_graphml(graphml_file)
    data = nx.node_link_data(G, edges="edges")
    print(f"{data.get('graph')=}")
    return G, data


if __name__ == '__main__':
    print(graphml_to_json("graphrag_cache/graph_chunk_entity_relation.graphml"))
