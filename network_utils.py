from collections import defaultdict

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
            nodes_to_remove.append(node)  # Mark node for removal if no attribute info

    filtered_graph.remove_nodes_from(nodes_to_remove)

    if verbose:
        print(f"Did not find {attribute_name} for: {len(nodes_to_remove)} philosophers (therefore not included in the filtered graph)")
        print(f" -> Example of removed nodes: {list(nodes_to_remove[:3])}")
        print(f"Original graph: {len(graph.nodes)} nodes and {len(graph.edges)} edges")
        print(f"Filtered '{attribute_name}' graph: {len(filtered_graph.nodes)} nodes and {len(filtered_graph.edges)} edges")

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