import re
import math
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


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


# def process_text(text, stop_words, lemmatizer):
#     tokens = word_tokenize(text)
#     tokens = [word.lower() for word in tokens if word.isalpha()]  # Only keep alphabetic tokens
#     tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
#     tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatize words
#     return Counter(tokens)  # Count word frequencies


def process_text(text, lemmatizer, exclusion_list=None):
    """Process text with optional exclusion of specific terms."""
    if exclusion_list is None:
        exclusion_list = set()  # Default to an empty set
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]  # Convert to lowercase and filter alphabetic tokens
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
