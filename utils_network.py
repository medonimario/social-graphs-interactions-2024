from collections import defaultdict
import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson, expon, lognorm


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

###################
# Network Metrics #
###################


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

# Function to plot empirical data and fitted model on log-log scale


def plot_degree_distribution_loglog(degrees, model_name, fitted_params):
    plt.figure(figsize=(8, 6))

    # Empirical degree distribution
    degree_counts = np.bincount(degrees)
    degrees_unique = np.nonzero(degree_counts)[0]
    counts = degree_counts[degrees_unique]
    prob = counts / counts.sum()

    plt.scatter(degrees_unique, prob, color='blue',
                marker='o', label='Empirical', alpha=0.6)

    # Generate degrees for plotting fitted model
    x = np.arange(degrees_unique.min(), degrees_unique.max()+1)

    if model_name == 'Poisson':
        pmf = poisson.pmf(x, fitted_params['lambda'])
        plt.plot(x, pmf, 'r-', label='Poisson Fit')
    elif model_name == 'Exponential':
        # For discrete plotting, use the CDF to approximate
        # Alternatively, treat as continuous
        pdf = expon.pdf(x, scale=fitted_params['scale'])
        plt.plot(x, pdf, 'g-', label='Exponential Fit')
    elif model_name == 'Log-Normal':
        # Avoid zero by starting from 1
        pdf = lognorm.pdf(
            x, fitted_params['sigma'], loc=fitted_params['loc'], scale=fitted_params['scale'])
        plt.plot(x, pdf, 'm-', label='Log-Normal Fit')
    elif model_name == 'Power-Law':
        # Power-law PDF: C * x^{-alpha}
        pdf = (x ** (-fitted_params['alpha']))
        # Normalize the PDF over the range xmin to max(x)
        pdf = pdf / pdf.sum()
        plt.plot(x, pdf, 'k-', label='Power-Law Fit')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Degree (log scale)')
    plt.ylabel('Probability (log scale)')
    plt.title(f'Degree Distribution with {model_name} Fit')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_all_models_loglog(degrees, fitted_models):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    models = ['Poisson', 'Exponential', 'Log-Normal', 'Power-Law']
    colors = ['r', 'g', 'm', 'k']

    for ax, model, color in zip(axes.flatten(), models, colors):
        # Empirical degree distribution
        degree_counts = np.bincount(degrees)
        degrees_unique = np.nonzero(degree_counts)[0]
        counts = degree_counts[degrees_unique]
        prob = counts / counts.sum()

        ax.scatter(degrees_unique, prob, color='blue',
                   marker='o', label='Empirical', alpha=0.6)

        # Generate degrees for plotting fitted model
        x = np.arange(degrees_unique.min(), degrees_unique.max()+1)

        if model == 'Poisson':
            pmf = poisson.pmf(x, fitted_models['Poisson']['lambda'])
            ax.plot(x, pmf, color=color, linestyle='-', label='Poisson Fit')
        elif model == 'Exponential':
            pdf = expon.pdf(x, scale=fitted_models['Exponential']['scale'])
            ax.plot(x, pdf, color=color, linestyle='-',
                    label='Exponential Fit')
        elif model == 'Log-Normal':
            pdf = lognorm.pdf(x, fitted_models['Log-Normal']['sigma'],
                              loc=fitted_models['Log-Normal']['loc'], scale=fitted_models['Log-Normal']['scale'])
            ax.plot(x, pdf, color=color, linestyle='-', label='Log-Normal Fit')
        elif model == 'Power-Law':
            pdf = (x ** (-fitted_models['Power-Law']['alpha']))
            pdf = pdf / pdf.sum()
            ax.plot(x[x >= fitted_models['Power-Law']['xmin']], pdf[x >= fitted_models['Power-Law']
                    ['xmin']], color=color, linestyle='-', label='Power-Law Fit')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Degree (log scale)')
        ax.set_ylabel('Probability (log scale)')
        ax.set_title(f'Degree Distribution with {model} Fit')
        ax.legend()

    plt.tight_layout()
    plt.show()


def plot_all_models_linear(degrees, fitted_models):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    models = ['Poisson', 'Exponential', 'Log-Normal', 'Power-Law']
    colors = ['r', 'g', 'm', 'k']

    for ax, model, color in zip(axes.flatten(), models, colors):
        # Empirical degree distribution
        degree_counts = np.bincount(degrees)
        degrees_unique = np.nonzero(degree_counts)[0]
        counts = degree_counts[degrees_unique]
        prob = counts / counts.sum()

        ax.scatter(degrees_unique, prob, color='blue',
                   marker='o', label='Empirical', alpha=0.6)

        # Generate degrees for plotting fitted model
        x = np.arange(degrees_unique.min(), degrees_unique.max() + 1)

        if model == 'Poisson':
            pmf = poisson.pmf(x, fitted_models['Poisson']['lambda'])
            ax.plot(x, pmf, color=color, linestyle='-', label='Poisson Fit')
        elif model == 'Exponential':
            pdf = expon.pdf(x, scale=fitted_models['Exponential']['scale'])
            ax.plot(x, pdf, color=color, linestyle='-',
                    label='Exponential Fit')
        elif model == 'Log-Normal':
            pdf = lognorm.pdf(x, fitted_models['Log-Normal']['sigma'],
                              loc=fitted_models['Log-Normal']['loc'], scale=fitted_models['Log-Normal']['scale'])
            ax.plot(x, pdf, color=color, linestyle='-', label='Log-Normal Fit')
        elif model == 'Power-Law':
            pdf = (x ** (-fitted_models['Power-Law']['alpha']))
            pdf = pdf / pdf.sum()
            ax.plot(x[x >= fitted_models['Power-Law']['xmin']], pdf[x >= fitted_models['Power-Law']
                    ['xmin']], color=color, linestyle='-', label='Power-Law Fit')

        ax.set_xlabel('Degree')
        ax.set_ylabel('Probability')
        ax.set_title(f'Degree Distribution with {model} Fit')
        ax.legend()

    plt.tight_layout()
    plt.show()


def effective_size(graph, node):
    """Calculate the effective size of a node."""
    neighbors = set(graph.neighbors(node))  # Immediate neighbors
    total_neighbors = len(neighbors)

    if total_neighbors <= 1:
        return total_neighbors  # No structural hole if only 1 neighbor

    redundancy = 0
    for neighbor in neighbors:
        shared_neighbors = set(graph.neighbors(neighbor)) & neighbors
        redundancy += len(shared_neighbors) / total_neighbors

    return total_neighbors - redundancy
