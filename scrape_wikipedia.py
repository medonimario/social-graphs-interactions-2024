import urllib.request
import urllib.parse  # to handle special characters in the title
import json
import re
import os


def getJsonResponse(title):
    baseurl = "https://en.wikipedia.org/w/api.php?"
    action = "action=query"
    title = f"titles={urllib.parse.quote(title)}"
    content = "prop=revisions&rvprop=content"
    dataformat = "format=json"
    rvslots = "rvslots=main"

    query = "{}{}&{}&{}&{}&{}".format(baseurl, action, content, title, dataformat, rvslots)

    try:
        wikiresponse = urllib.request.urlopen(query)
        if wikiresponse.getcode() != 200:
            print(f"Error: Received non-200 HTTP status code {wikiresponse.getcode()}")
            return None

        wikidata = wikiresponse.read()
        wikiJson = json.loads(wikidata)
        page = next(iter(wikiJson['query']['pages'].values()))

        if 'revisions' in page and len(page['revisions']) > 0:
            wikitext = page['revisions'][0]['slots']['main']['*']
            return wikitext
        else:
            print(f"Error: Page '{title}' does not contain revisions.")
            return None
    except urllib.error.URLError as e:
        print(f"Network error: {e.reason}")
        return None
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return None

def extract_title_link(match):
    title = re.search(r'\[\[([^\|\]]+)', match)
    if title:
        return title.group(1).replace(" ", "_")
    else:
        return None

def findLinks(wikipage):
    pattern = r'\[{2}[\w\-\s\(\)]*\|?[\w\s\-\(\)]*\]{2}'
    matches = re.findall(pattern, wikipage)
    unique_matches = set(matches)
    links = [extract_title_link(unique_match) for unique_match in unique_matches]
    return links

def save_wikipedia_pages_to_files(title_links, path):
    if not os.path.exists(path):
        os.makedirs(path)
    
    invalid_links = []  # Track titles that could not be saved
    for title_link in title_links:
        all_wikitext = getJsonResponse(title_link)
        if not all_wikitext:
            print(f"Skipping '{title_link}' as it has no content.")
            invalid_links.append(title_link)  # Track invalid pages without modifying the list directly
            continue
        filename = os.path.join(path, f"{title_link}.txt")
        with open(filename, "w", encoding="utf-8") as file:
            file.write(all_wikitext)


wiki_links = ["List of philosophers (A–C)", "List of philosophers (D–H)", "List of philosophers (I–Q)", "List of philosophers (R–Z)"]
title_links = []
for wiki_link in wiki_links:
  wiki_markup = getJsonResponse(wiki_link)
  title_links.extend(findLinks(wiki_markup))

# Remove irrelevant links if they exist
for unwanted in ["List_of_philosophers", "Philosopher", "Stanford_Encyclopedia_of_Philosophy", "Encyclopedia_of_Philosophy", "Routledge_Encyclopedia_of_Philosophy", "The_Cambridge_Dictionary_of_Philosophy", "The_Oxford_Companion_to_Philosophy"]:
    if unwanted in title_links:
        title_links.remove(unwanted)

path = "Philosophers/"
save_wikipedia_pages_to_files(title_links, path)