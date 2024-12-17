import re
import math
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from community import community_louvain
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from fa2_modified import ForceAtlas2

### Document processing functions ###

def clean_wikipedia_text(text):
    """Clean Wikipedia syntax and irrelevant content from text."""
    # Remove comments <!-- ... -->
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    # Remove content inside double curly braces {{...}}
    text = re.sub(r'\{\{.*?\}\}', '', text, flags=re.DOTALL)
    # Remove references <ref>...</ref>
    text = re.sub(r'<ref.*?>.*?</ref>', '', text, flags=re.DOTALL)
    # Remove self-closing references <ref .../>
    text = re.sub(r'<ref.*?/>', '', text)
    # Remove file links [[File:...]] or [[Image:...]]
    text = re.sub(r'\[\[(File|Image):.*?\]\]', '', text, flags=re.IGNORECASE)
    # Replace internal links [[Link|Text]] with 'Text' or 'Link' if 'Text' is missing
    text = re.sub(r'\[\[([^|\]]*\|)?([^\]]+)\]\]', r'\2', text)
    # Remove any remaining brackets and braces
    text = text.replace('[', '').replace(']', '').replace('{', '').replace('}', '')
    # Remove excess whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_text(text, stop_words , lemmatizer, exclusion_list):
    """Process text with optional exclusion of specific terms."""
    if exclusion_list is None:
        exclusion_list = set()  # Default to an empty set
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]  # Convert to lowercase and filter alphabetic tokens
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    tokens = [word for word in tokens if word not in exclusion_list]  # Remove excluded terms
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatize words
    return Counter(tokens)  # Count word frequencies

def compute_tf_idf(word_counts_dict):
    # Calculate document frequency (df) for each term
    all_terms = set(term for counts in word_counts_dict.values() for term in counts.keys())
    df = {term: sum(1 for counts in word_counts_dict.values() if term in counts) for term in all_terms}

    # Total number of subfields
    N = len(word_counts_dict)

    # Calculate IDF for each term
    idf = {term: math.log(N / df[term]) for term in df}

    # Calculate TF-IDF for each subfield
    tf_idf_scores = {}
    for subfield, word_counts in word_counts_dict.items():
        total_words = sum(word_counts.values())
        tf_idf_scores[subfield] = {term: (count / total_words) * idf[term] for term, count in word_counts.items()}

    return tf_idf_scores


### Functions for modularity calculation and greedy optimization ###
def calculate_modularity(G, partition):
    m = G.size(weight='weight')
    modularity = 0.0
    for community in set(partition.values()):
        nodes_in_community = [node for node in G.nodes if partition[node] == community]
        subgraph = G.subgraph(nodes_in_community)
        l_c = subgraph.size(weight='weight')
        d_c = sum(G.degree(n, weight='weight') for n in nodes_in_community)
        modularity += (l_c / m) - (d_c / (2 * m)) ** 2
    return modularity

def greedy_modularity_optimization(G, initial_subfields, random_seed=42):
    random.seed(random_seed)
    partition = {}
    for node in G.nodes:
        partition[node] = random.choice(initial_subfields[node])
    improved = True
    while improved:
        improved = False
        for node in G.nodes:
            best_subfield = partition[node]
            best_modularity = calculate_modularity(G, partition)
            for subfield in initial_subfields[node]:
                if subfield == partition[node]:
                    continue
                partition[node] = subfield
                new_modularity = calculate_modularity(G, partition)
                if new_modularity > best_modularity:
                    best_modularity = new_modularity
                    best_subfield = subfield
                    improved = True
            partition[node] = best_subfield
    return partition

def find_unassigned_subfields(initial_subfields, partition):
    all_subfields = set(subfield for sublist in initial_subfields.values() for subfield in sublist)
    assigned_subfields = set(partition.values())
    unassigned_subfields = all_subfields - assigned_subfields
    return unassigned_subfields

def get_louvain_modularity(graph, partition_dict):
    return f"{community_louvain.modularity(partition_dict, graph):.4f}"

### Functions for TF-IDF weighted embeddings and visualization ###

def load_glove_embeddings(glove_file):
    embeddings = {}
    with open(glove_file, "r", encoding="utf-8") as file:
        for line in file:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

def compute_tf_idf_weighted_embeddings(tf_idf_scores, embeddings):
    subfield_embeddings = {}
    for subfield, tf_idf in tf_idf_scores.items():
        word_vectors = []
        weights = []
        for word, weight in tf_idf.items():
            if word in embeddings:
                word_vectors.append(embeddings[word] * weight)
                weights.append(weight)
        if word_vectors:
            subfield_embeddings[subfield] = np.sum(word_vectors, axis=0) / np.sum(weights)
        else:
            print(f"No embeddings found for subfield {subfield}")
            subfield_embeddings[subfield] = np.zeros(300)  # Assume 300D GloVe
    return subfield_embeddings

def plot_confusion_matrix(subfields, similarity_matrix):
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_matrix, xticklabels=subfields, yticklabels=subfields, annot=True, fmt=".2f", cmap="Blues")
    plt.title("Subfield Cosine Similarity Matrix (TF-IDF Weighted)")
    plt.xlabel("Subfields")
    plt.ylabel("Subfields")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


### Functions for similarity graph creation and visualization ###

def create_similarity_graph(subfields, similarity_matrix, threshold):
    G = nx.Graph()
    for subfield in subfields:
        G.add_node(subfield)
    for i in range(len(subfields)):
        for j in range(len(subfields)):
            if i != j and similarity_matrix[i, j] >= threshold:
                G.add_edge(subfields[i], subfields[j], weight=similarity_matrix[i, j])
    return G

def filter_top_n_edges(G, n):
    new_G = nx.Graph()
    new_G.add_nodes_from(G.nodes(data=True))
    for node in G.nodes():
        edges = [(neighbor, G[node][neighbor]['weight']) for neighbor in G.neighbors(node)]
        edges = sorted(edges, key=lambda x: x[1], reverse=True)
        for neighbor, weight in edges[:n]:
            if not new_G.has_edge(node, neighbor):
                new_G.add_edge(node, neighbor, weight=weight)
    return new_G

def visualize_forceatlas2_graph(G, threshold, gamma=2.0):
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
    positions = forceatlas2.forceatlas2_networkx_layout(G, pos=None, iterations=2000)
    edge_weights = nx.get_edge_attributes(G, 'weight')
    edges, weights = zip(*edge_weights.items())

    max_weight = max(weights)
    transformed_weights = [((w - threshold) / (max_weight - threshold)) ** gamma for w in weights]
    transformed_weights = np.clip(transformed_weights, 0, 1)

    cmap = plt.cm.get_cmap('Reds')
    fig, ax = plt.subplots(figsize=(14, 7))
    for i, edge in enumerate(edges):
        color = cmap(transformed_weights[i])
        nx.draw_networkx_edges(
            G, positions, edgelist=[edge],
            edge_color=[color], width=weights[i]*3, alpha=0.7
        )
    nx.draw_networkx_nodes(G, positions, node_size=1000, node_color='purple', alpha=0.9)
    for node in G.nodes():
        x, y = positions[node]
        last_name = node.split('_')[-1]
        ax.text(x, y, last_name, fontsize=10, fontweight='bold', color='black',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2', alpha=0.6))

    norm = plt.Normalize(vmin=threshold, vmax=max_weight)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = plt.colorbar(sm, ax=ax, label='Embedding cossine similarity', fraction=0.03, pad=0.1)
    cbar.ax.tick_params(labelsize=10)
    ax.axis('off')
    plt.show()