import re
import csv
import os
import math
import numpy as np
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from utils_globals import *

# This function reads the wordlist and creates a dictionary
def load_labmt_wordlist(file_path):
    labmt_dict = {}
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        # Skip the first four lines
        for _ in range(4):
          next(reader)
        for row in reader:
            # Check if the row has the expected number of columns
            if len(row) < 3:
                print(f"Skipping row due to insufficient columns: {row}")
                continue
            word = row[0].lower()  # Lowercase the word for consistency
            try:
                score = float(row[2])  # Happiness average is in the third column
                labmt_dict[word] = score
            except ValueError:
                # Handle case where score is not a valid float
                print(f"Skipping row due to invalid score: {row}")
                continue
    return labmt_dict

# This function will return a list of words for each philosopher's page, ready for sentiment scoring
def clean_and_tokenize(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    tokens = text.split()  # Tokenize (split into words)
    return tokens

# Calculate sentiment based on the LabMT wordlist
def calculate_sentiment(tokens, labmt_dict):
    scores = [labmt_dict[word] for word in tokens if word in labmt_dict]
    return sum(scores) / len(scores) if scores else None

# Calculate the sentiment SCORE and store it as a node ATTRIBUTE
def process_graph(graph, labmt_wordlist):
    processed_count = 0  # Initialize counter for processed philosopher

    for node in graph.nodes:
        philosopher_name = node
        philosopher_file_name = philosopher_name.replace(' ', '_') + ".txt"  # Format the philosopher file name
        philosopher_file_path = os.path.join(DOWNLOADS_DIR, philosopher_file_name)  # Prepend directory path

        # Check if the file exists for this philosopher
        if os.path.exists(philosopher_file_path):
            try:
                # Read the text file content
                with open(philosopher_file_path, 'r', encoding='utf-8') as f:
                    text = f.read()

                    # Tokenize the text
                    tokens = clean_and_tokenize(text)

                    # Calculate sentiment
                    sentiment = calculate_sentiment(tokens, labmt_wordlist)

                    # Store sentiment in the graph node
                    graph.nodes[node]["sentiment"] = sentiment
                    processed_count += 1  # Increment processed count

            except Exception as e:
                print(f"Error processing file for {philosopher_name}: {e}")
        else:
            print(f"File not found for artist: {philosopher_name} ({philosopher_file_name})")

    print(f"\nTotal artists processed: {processed_count} out of {len(graph.nodes)}")

# Function to calculate the average sentiment for a specific community
def calculate_community_sentiment(graph, community_id, partition):
    community_nodes = [node for node, community in partition.items() if community == community_id]
    # Collect the pre-calculated sentiment for each node in the community
    sentiments = []
    for node in community_nodes:
      sentiment = graph.nodes[node].get("sentiment")
      if sentiment is not None:
        sentiments.append(sentiment)

    # Calculate the average sentiment for the community
    if sentiments:
      return np.mean(sentiments)
    else:
      return None

# Get the three most connected nodes (philosophers) for each community
def get_top_connected_characters(graph, community_id, partition):
    community_nodes = [node for node, community in partition.items() if community == community_id]
    # Get degrees of nodes in the community
    degrees = {node: graph.degree(node) for node in community_nodes}
    # Sort by degree
    top_3_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:3]
    return [node for node, degree in top_3_nodes]


# Compile and clean texts for all subfields
def compile_and_clean_all_subfields(partition, text_dir):
    subfield_texts = {}

    for subfield in set(partition.values()):
        compiled_text = ""
        # Find all philosophers in this subfield
        philosophers = [node for node, sf in partition.items() if sf == subfield]
        
        for philosopher in philosophers:
            filename = os.path.join(text_dir, f"{philosopher}.txt")
            if os.path.exists(filename):
                with open(filename, "r", encoding="utf-8") as file:
                    text = file.read()
                    compiled_text += text + "\n"
            else:
                print(f"Warning: File not found for philosopher {philosopher}")
        
        # Clean compiled text
        subfield_texts[subfield] = clean_wikipedia_text(compiled_text)

    return subfield_texts



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
