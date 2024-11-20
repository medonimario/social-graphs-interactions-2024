import urllib.request
import csv


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
