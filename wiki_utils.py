import urllib.request
import urllib.parse  # to handle special characters in the title
import json
import shutil
import re
import os
import networkx as nx
import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pickle

# Set the directory to downloads
DOWNLOADS_DIR = "downloads"
TITLE_LINKS_FILE = "title_links.json"

def getJsonResponse(title):
  # Define the components of the query
  baseurl = "https://en.wikipedia.org/w/api.php?"
  action = "action=query"
  title = f"titles={urllib.parse.quote(title)}"
  content = "prop=revisions&rvprop=content"
  dataformat = "format=json"
  rvslots = "rvslots=main"

  # Construct the query URL
  query = "{}{}&{}&{}&{}&{}".format(baseurl, action, content, title, dataformat, rvslots)

  try:
    # Make the request to Wikipedia API
    wikiresponse = urllib.request.urlopen(query)

    # Check if the HTTP status is OK (200)
    if wikiresponse.getcode() != 200:
      print(f"Error: Received non-200 HTTP status code {wikiresponse.getcode()}")
      return None

    wikidata = wikiresponse.read()

    # Parse the JSON response
    try:
      wikiJson = json.loads(wikidata)
    except json.JSONDecodeError:
      print("Error: Failed to decode JSON response")
      return None

    # Get the page from the JSON response
    page = next(iter(wikiJson['query']['pages'].values()))  # extract the single page

    # Check if the page has revisions and extract the latest wikitext content
    if 'revisions' in page and len(page['revisions']) > 0:
      wikitext = page['revisions'][0]['slots']['main']['*']  # extract wikitext from "main" slot
      return wikitext
    else:
      #print(f"Error: Page '{title}' does not contain revisions.")
      return None

  except urllib.error.URLError as e:
    print(f"Network error: {e.reason}")
    return None
  except Exception as e:
    print(f"Unexpected error: {str(e)}")
    return None

## Convert the list to link titles e.g. John McCain (fictional) => John_McCain_(fictional)
def extract_title_link(match):
  # Regular expression to match the content between [[ and | (the first part of the link)
  title = re.search(r'\[\[([^\|\]]+)', match)
  if title:
    # Replace all whitespaces in the title with underscores
    return title.group(1).replace(" ", "_")
  else:
    print("ERROR FINDING ", match)
    return None

def findLinks(wikipage):
  pattern = r'\[{2}[\w\-\s\(\)]*\|?[\w\s\-\(\)]*\]{2}' ## regex for finding links e.g.: [[John McCain (fictional)|John McCain]]
  matches = re.findall(pattern, wikipage)
  # Convert the list to a set to keep only unique matches
  unique_matches = set(matches)

  links = [extract_title_link(unique_match) for unique_match in unique_matches]
  return links

def build_graph_from_files(path):
    files = os.listdir(path)
    outgoing_links = {}
    pages = set()
    
    # Process each file in the directory to collect outgoing links and all pages
    for file in files:
        if not file.endswith(".txt"): 
            continue
        
        filepath = os.path.join(path, file)
        with open(filepath, "r", encoding="utf-8") as f:
            wikipage = f.read()
            wikipage_links = findLinks(wikipage)
            withoutExtension = os.path.splitext(file)[0]
            pages.add(withoutExtension)  # Add the page to the set of all pages
            
            for link in wikipage_links:
                if link + ".txt" in files:  # Only consider links that exist as files
                    outgoing_links.setdefault(withoutExtension, []).append(link)
                    pages.add(link)  # Add the linked page to the set of all pages

    G = nx.DiGraph()

    # Add all pages to the graph with the 'contentlength' attribute
    for page in pages:
        filename = os.path.join(path, f"{page}.txt")
        with open(filename, "r", encoding="utf-8") as f:
            content = f.read()
        word_count = len(content.split())
        G.add_node(page, contentlength=word_count)
    
    # Add edges based on outgoing links
    for page, links in outgoing_links.items():
        for link in links:
            G.add_edge(page, link)

    # Remove isolated nodes
    isolated_nodes = list(nx.isolates(G))
    if isolated_nodes:
        G.remove_nodes_from(isolated_nodes)

    # Get the largest connected component
    if nx.is_weakly_connected(G):
        S = G.copy()
    else:
        largest_cc = max(nx.weakly_connected_components(G), key=len)
        S = G.subgraph(largest_cc).copy()
    
    return S

def get_category_data(links, replace_terms, verbose=False):
    """
    Processes a list of links, fetches philosopher data, and organizes it into a clean format.
    
    Args:
        links (list): List of Wikipedia links to process.
        replace_terms (list of tuples): Terms to replace in the keys for cleaning data (e.g., "List of").
        verbose (bool): Whether to print detailed output during processing.
        
    Returns:
        tuple: A dictionary mapping cleaned categories to philosophers and a dictionary mapping philosophers to categories.
    """
    # Step 1: Fetch and organize raw data
    raw_data = {}
    for link in links:
        wiki_markup = getJsonResponse(link)
        philosopher_list = findLinks(wiki_markup)
        if link not in raw_data:
            raw_data[link] = []
        raw_data[link].extend(philosopher_list)

    # Step 2: Clean the data
    clean_data = {}
    for category in raw_data:
        cleaned_category = category
        for old, new in replace_terms:
            cleaned_category = cleaned_category.replace(old, new).strip()
        clean_data[cleaned_category] = raw_data[category]

    # Step 3: Convert to philosopher-to-category mapping
    philosopher_info = {}
    for category, philosopher_list in clean_data.items():
        if verbose:
            print(f"- {category}: {len(philosopher_list)}")
        for philosopher in philosopher_list:
            if philosopher not in philosopher_info:
                philosopher_info[philosopher] = []
            philosopher_info[philosopher].append(category)

    return clean_data, philosopher_info

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