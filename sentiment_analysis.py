import urllib.request
import csv
import re
import os
import pickle

# Load graph:
S = pickle.load(open("graph.pkl", "rb"))
DOWNLOADS_DIR = "downloads"

# Download the LabMT wordlist (with happiness scores)
file_id = "1fEW8gxKEfwiNRgpeqQ1S9qbATyrNftoE"
url = f"https://drive.google.com/uc?export=download&id={file_id}"
local_filename = "labmt_wordlist.txt"
urllib.request.urlretrieve(url, local_filename)

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

# Load the LabMT wordlist into memory
labmt_wordlist = load_labmt_wordlist(local_filename)

# Check to ensure the list is loaded correctly
print(list(labmt_wordlist.items())[:10])

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

# Calculate the sentiment score and store it as a node attribute
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


# Example of running the function
process_graph(S, labmt_wordlist)
