from collections import defaultdict
import networkx as nx
import random


def filter_graph_by_attribute(graph, attribute_info, attribute_name, verbose=False):
    """
    Filters a graph by adding attributes to nodes or removing nodes if attribute info is missing.

    Args:
        graph (networkx.Graph): The input graph to filter.
        attribute_info (dict): A dictionary mapping nodes to their attributes (e.g., subfields or traditions).
        attribute_name (str): The name of the attribute to add to the graph nodes.
        verbose (bool): Whether to print details about the filtering process.

    Returns:
        networkx.Graph: A filtered copy of the graph with attributes added or nodes removed.
    """
    filtered_graph = graph.copy()
    nodes_to_remove = []  # Collect nodes to remove

    for node in list(filtered_graph.nodes):
        if node in attribute_info:
            filtered_graph.nodes[node][attribute_name] = attribute_info[node]
        else:
            # Mark node for removal if no attribute info
            nodes_to_remove.append(node)

    filtered_graph.remove_nodes_from(nodes_to_remove)

    if verbose:
        print(
            f"Did not find {attribute_name} for: {len(nodes_to_remove)} philosophers (therefore not included in the filtered graph)")
        print(f" -> Example of removed nodes: {list(nodes_to_remove[:3])}")
        print(
            f"Original graph: {len(graph.nodes)} nodes and {len(graph.edges)} edges")
        print(
            f"Filtered '{attribute_name}' graph: {len(filtered_graph.nodes)} nodes and {len(filtered_graph.edges)} edges")

    return filtered_graph


def count_nodes_by_attribute(graph, attribute_name, verbose=False):
    """
    Counts the number of nodes in the graph for each category of a given attribute.

    Args:
        graph (networkx.Graph): The graph to analyze.
        attribute_name (str): The attribute name to group and count nodes by.
        verbose (bool): Whether to print details about the counts.

    Returns:
        dict: A dictionary with categories as keys and node counts as values.
    """
    attribute_counts = defaultdict(int)

    for _, attributes in graph.nodes(data=True):
        if attribute_name in attributes:
            attribute = attributes[attribute_name]
            for attribute_value in attribute:
                attribute_counts[attribute_value] += 1

    if verbose:
        print(f"Number of nodes by {attribute_name}:")
        for attribute, count in attribute_counts.items():
            print(f" -> {attribute}: {count}")

    return dict(attribute_counts)


def grow_Barabasi_Albert_graph(n=5000, E=None):
    if E is None:
        raise ValueError("Total number of edges E must be specified.")
    if n < 2:
        raise ValueError("Number of nodes n must be at least 2.")

    # Function to approximate m from n and E
    def approximate_m(n, E):
        best_m = None
        min_diff = float('inf')
        for m in range(1, n):
            total_edges = m * n - (m * (m + 1)) // 2
            diff = abs(total_edges - E)
            if diff < min_diff:
                min_diff = diff
                best_m = m
            if diff == 0:
                break
        return best_m

    m = approximate_m(n, E)
    print(f"Using m = {m} to approximate E = {E}")

    # Initialize the graph with a complete graph of m nodes
    F_BA = nx.complete_graph(m)

    # Function to get the degree list for preferential attachment
    def get_degree_list(G):
        degree_list = []
        for node in G.nodes:
            degree_list.extend([node] * G.degree[node])
        return degree_list

    # Function to add a new node connected to m existing nodes preferentially
    def add_node_proportional_to_degree(G, new_node, m):
        degree_list = get_degree_list(G)
        targets = set()
        while len(targets) < m:
            target_node = random.choice(degree_list)
            targets.add(target_node)
        for target_node in targets:
            G.add_edge(new_node, target_node)

    # Add the remaining n - m nodes
    for new_node in range(m, n):
        add_node_proportional_to_degree(F_BA, new_node, m)

    return F_BA


def compute_network_metrics(graph):
    """Computes and returns key metrics for a graph."""
    metrics = {}

    # Degree Distribution
    degrees = [d for _, d in graph.degree()]
    metrics['degree_histogram'] = nx.degree_histogram(graph)

    # Clustering Coefficient
    metrics['global_clustering'] = nx.transitivity(graph)
    metrics['average_clustering'] = nx.average_clustering(graph)

    # Assortativity
    metrics['assortativity'] = nx.degree_assortativity_coefficient(graph)

    # Connected Components
    components = list(nx.connected_components(graph))
    metrics['num_connected_components'] = len(components)
    metrics['largest_component_size'] = len(max(components, key=len))

    # Shortest Path Lengths (only for the largest connected component)
    largest_component = graph.subgraph(max(components, key=len)).copy()
    metrics['average_shortest_path'] = nx.average_shortest_path_length(
        largest_component)
    metrics['diameter'] = nx.diameter(largest_component)

    # Degree Centrality
    metrics['degree_centrality'] = nx.degree_centrality(graph)

    return metrics
