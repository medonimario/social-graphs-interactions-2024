from collections import defaultdict
import networkx as nx
import random
import matplotlib.pyplot as plt
import plotly.express as px
from fa2_modified import ForceAtlas2
import numpy as np
import pandas as pd
from scipy.stats import poisson, expon, lognorm
import powerlaw
import warnings
# Ignore warnings from powerlaw package
warnings.filterwarnings("ignore")

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
    # print(f"Using m = {m} to approximate E = {E}")

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

def model_fitting(degrees):
	# Step 1: Extract the degree sequence
    print(f"Minimum degree: {np.min(degrees)}")
    print(f"Maximum degree: {np.max(degrees)}")

		# Step 2: Fit candidate distributions using MLE

		# 2a. Poisson Distribution
    lambda_poisson = np.mean(degrees)
    log_likelihood_poisson = np.sum(poisson.logpmf(degrees, lambda_poisson))

		# 2b. Exponential Distribution (continuous approximation)
    params_exp = expon.fit(degrees, floc=0)  # Fix loc=0
    lambda_exp = 1 / params_exp[1]  # Scale parameter is 1/lambda
    log_likelihood_exp = np.sum(expon.logpdf(degrees, *params_exp))

		# 2c. Log-Normal Distribution
    params_lognorm = lognorm.fit(degrees[degrees > 0], floc=0)  # Exclude zeros
    sigma_lognorm, loc_lognorm, scale_lognorm = params_lognorm
    log_likelihood_lognorm = np.sum(lognorm.logpdf(degrees[degrees > 0], *params_lognorm))

		# 2d. Power-Law Distribution
    fit = powerlaw.Fit(degrees, xmin=1)
    alpha_powerlaw = fit.power_law.alpha
    xmin_powerlaw = fit.power_law.xmin
    log_likelihood_powerlaw = fit.power_law.loglikelihoods(degrees).sum()

		# Step 3: Compute AIC and BIC for each model
    n = len(degrees)

		# Poisson
    k_poisson = 1  # lambda
    AIC_poisson = 2 * k_poisson - 2 * log_likelihood_poisson
    BIC_poisson = k_poisson * np.log(n) - 2 * log_likelihood_poisson

		# Exponential
    k_exp = 1  # lambda
    AIC_exp = 2 * k_exp - 2 * log_likelihood_exp
    BIC_exp = k_exp * np.log(n) - 2 * log_likelihood_exp

		# Log-Normal
    n_lognorm = len(degrees[degrees > 0])
    k_lognorm = 2  # sigma and scale
    AIC_lognorm = 2 * k_lognorm - 2 * log_likelihood_lognorm
    BIC_lognorm = k_lognorm * np.log(n_lognorm) - 2 * log_likelihood_lognorm

		# Power-Law
    n_powerlaw = len(degrees[degrees >= xmin_powerlaw])
    k_powerlaw = 2  # alpha and xmin
    AIC_powerlaw = 2 * k_powerlaw - 2 * log_likelihood_powerlaw
    BIC_powerlaw = k_powerlaw * np.log(n_powerlaw) - 2 * log_likelihood_powerlaw

		# Step 4: Select the best-fitting model based on AIC and BIC
    AICs = {
    'Poisson': AIC_poisson,
    'Exponential': AIC_exp,
    'Log-Normal': AIC_lognorm,
    'Power-Law': AIC_powerlaw
		}

    BICs = {
    'Poisson': BIC_poisson,
    'Exponential': BIC_exp,
    'Log-Normal': BIC_lognorm,
    'Power-Law': BIC_powerlaw
		}

    best_fit_aic = min(AICs, key=AICs.get)
    best_fit_bic = min(BICs, key=BICs.get)

		# Output results
    print("\nModel fitting results:")
    print("-----------------------")
    print("Log-Likelihoods:")
    print(f"-Poisson: {log_likelihood_poisson:.1f}")
    print(f"-Exponential: {log_likelihood_exp:.1f}")
    print(f"-Log-Normal: {log_likelihood_lognorm:.1f}")
    print(f"-Power-Law: {log_likelihood_powerlaw:.1f}")

    print(f"\nAIC values:")
    for dist, aic in AICs.items():
        print(f"-{dist}: {aic:.2f} {'<- Best fit' if dist == best_fit_aic else ''}")

    print(f"\nBIC Values:")
    for dist, bic in BICs.items():
        print(f"-{dist}: {bic:.1f} {'<- Best fit' if dist == best_fit_bic else ''}")

    print("\nEstimated parameters:")
    print("---------------------")
    print(f"Poisson lambda: {lambda_poisson:.2f}")
    print(f"Exponential lambda: {lambda_exp:.4f}")
    print(f"Log-Normal sigma: {sigma_lognorm:.4f}, scale: {scale_lognorm:.2f}")
    print(f"Power-Law alpha: {alpha_powerlaw:.2f}, xmin: {xmin_powerlaw}")

    fitted_models = {
        'Poisson': {'lambda': lambda_poisson},
        'Exponential': {'scale': params_exp[1]},  # scale = 1/lambda
        'Log-Normal': {'sigma': sigma_lognorm, 'loc': loc_lognorm, 'scale': scale_lognorm},
        'Power-Law': {'alpha': alpha_powerlaw, 'xmin': xmin_powerlaw}
    }
    return fitted_models


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


def plot_centrality_comparison(graph, centrality_x, centrality_y, x_label, y_label, title):
    """
    Creates an interactive scatter plot to compare two centrality measures.

    Parameters:
    - graph: NetworkX graph
    - centrality_x: Dictionary of centrality values for x-axis
    - centrality_y: Dictionary of centrality values for y-axis
    - x_label: Label for x-axis
    - y_label: Label for y-axis
    - title: Title of the plot
    """
    # Prepare data for the plot
    data = pd.DataFrame({
        'Node': list(centrality_x.keys()),
        x_label: list(centrality_x.values()),
        y_label: list(centrality_y.values())
    })

    # Create an interactive scatter plot
    fig = px.scatter(
        data,
        x=x_label,
        y=y_label,
        hover_name='Node',  # Display node name on hover
        hover_data={
            x_label: ':.4f',  # Format centrality values
            y_label: ':.4f'
        },
        title=title,
        labels={x_label: x_label, y_label: y_label},
        template="plotly_white"
    )
    
    fig.show()


    # Function to create subgraphs for each subfield
def create_subfield_subgraphs(graph):
    subfield_subgraphs = {}
    for subfield in set(subfield for subfields in nx.get_node_attributes(graph, 'subfields').values() for subfield in subfields):
        nodes_in_subfield = [
            node for node, data in graph.nodes(data=True) if subfield in data.get('subfields', [])
        ]
        subfield_subgraphs[subfield] = graph.subgraph(nodes_in_subfield).copy()
    return subfield_subgraphs


# Function to compute centrality measures
def compute_centrality_measures(subgraph):
    degree_centrality = nx.degree_centrality(subgraph)
    closeness_centrality = nx.closeness_centrality(subgraph)
    betweenness_centrality = nx.betweenness_centrality(subgraph)
    eigenvector_centrality = nx.eigenvector_centrality(subgraph)
    page_rank = nx.pagerank(subgraph)
    clustering = nx.clustering(subgraph)
    return {
        "degree": degree_centrality,
        "closeness": closeness_centrality,
        "betweenness": betweenness_centrality,
        "eigenvector": eigenvector_centrality,
        "pagerank": page_rank,
        "clustering": clustering
    }


# Function to find the top nodes by each centrality measure
def get_top_nodes_by_centrality(centrality_measures, top_n=5):
    top_nodes = {}
    for measure_name, centrality_dict in centrality_measures.items():
        sorted_nodes = sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_nodes[measure_name] = sorted_nodes
    return top_nodes


# Function to visualize a subgraph
def plot_subgraph(subgraph, centrality_measures, subfield_name, top_n=5):
    # Use degree centrality for node size
    degree_centrality = centrality_measures["degree"]
    node_sizes = [1000 * degree_centrality[node] for node in subgraph.nodes()]

    # Use closeness centrality for node color
    closeness_centrality = centrality_measures["closeness"]
    closeness_values = list(closeness_centrality.values())
    cmap = plt.cm.plasma  # Updated colormap
    norm = plt.Normalize(vmin=min(closeness_values), vmax=max(closeness_values))
    node_colors = [cmap(norm(closeness_centrality[node])) for node in subgraph.nodes()]

    # Compute layout
    forceatlas2 = ForceAtlas2(
        outboundAttractionDistribution=True,
        edgeWeightInfluence=1.0,
        jitterTolerance=0.05,
        barnesHutOptimize=True,
        barnesHutTheta=1.2,
        scalingRatio=0.1,
        strongGravityMode=False,
        gravity=0.1,
        verbose=False
    )
    positions = forceatlas2.forceatlas2_networkx_layout(subgraph, pos=None, iterations=2000)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 14))
    nx.draw_networkx_nodes(
        subgraph,
        positions,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.8,
        ax=ax
    )
    nx.draw_networkx_edges(
        subgraph,
        positions,
        edge_color='lightgray',
        alpha=0.5,
        ax=ax
    )

    # Add labels for the top `n` nodes by degree centrality
    top_degree_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]
    for node, _ in top_degree_nodes:
        x, y = positions[node]
        last_name = node.split('_')[-1]  # Use last name for labeling
        ax.text(
            x, y,
            last_name,
            fontsize=8,
            fontweight='bold',
            color='black',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2', alpha=0.6)
        )

    # Add color bar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Closeness Centrality', fontsize=12)

    plt.title(f"Subgraph for Subfield: {subfield_name}", fontsize=16)
    plt.axis('off')
    plt.show()


# Function to process and visualize all subfields
def process_and_visualize_subfields(graph, top_n=5):
    subfield_subgraphs = create_subfield_subgraphs(graph)
    for subfield_name, subgraph in subfield_subgraphs.items():
        print(f"Processing subfield: {subfield_name}")
        
        # Compute centrality measures
        centrality_measures = compute_centrality_measures(subgraph)
        
        # Get top nodes by centrality
        top_nodes = get_top_nodes_by_centrality(centrality_measures, top_n)
        
        # Print top nodes
        print(f"Top {top_n} nodes by centrality for subfield '{subfield_name}':")
        for measure, nodes in top_nodes.items():
            print(f"  {measure.capitalize()} Centrality:")
            for node, value in nodes:
                print(f"    {node}: {value:.4f}")
        
        # Plot subgraph
        plot_subgraph(subgraph, centrality_measures, subfield_name, top_n=top_n)