import os
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from pathlib import Path
import re
import pickle
import nltk
from gensim.models.phrases import Phraser
import csv
from spacy.tokenizer import Tokenizer
import spacy

#from pandas.api.types import SparseDtype

# Ensure necessary NLTK resources are downloaded
nltk.download('stopwords')

# Initialize necessary variables
stemmer = SnowballStemmer("english")
stopwords = set(nltk.corpus.stopwords.words('english'))
Tokenizer = spacy.load('en_core_web_sm').tokenizer
# Path setup
PROJECT_ROOT = Path("C:\\Users\\mitha\\Documents\\empirical_evidence_corren")

def project_path(*args):
    return PROJECT_ROOT.joinpath(*args)

def preprocess_wsw(text, dont_stem, tokenizer):
    """A function to preprocess text."""
    text = text.lower()
    text = re.sub('-', ' ', text)
    text = re.sub('[^A-Za-z0-9 ]', '', text)
    text = re.sub(r'\s+', ' ', text)
    tokenized = tokenizer(text)
    tokens = (t.text for t in tokenized)
    new_tokens = [stemmer.stem(w) if w not in dont_stem else w for w in tokens]
    all_text = ' '.join(new_tokens)
    return re.sub(r'\s+', ' ', all_text)

def create_dict(MAX, bigram, dont_stem):
    """Create the_con_dict and supporting structures."""
    all_con_dicts = []
    con_dict_trans = {}
    for i in range(1, MAX):
        con_dict_new = []
        dict_path = project_path("scripts", "con_dicts", f"dict_{i}.csv")
        with open(dict_path, 'r', encoding="ISO-8859-1") as inputfile:
            for row in csv.reader(inputfile):
                term_original = row[0].strip()
                term = ' '.join(bigram[term_original.split(' ')])
                term = preprocess_wsw(term, dont_stem, Tokenizer)
                if term:
                    con_dict_new.append(' ' + term + ' ')
                    con_dict_trans[' ' + term + ' '] = term_original
        all_con_dicts.append(list(set(con_dict_new)))

    the_con_dict = list(set(term for con_dict in all_con_dicts for term in con_dict))
    return the_con_dict, all_con_dicts, con_dict_trans

def process_hearings(hearing_list, main_df, the_con_dict, con_dict_trans):
    """Process hearings to compute term frequencies."""
    all_data = []  # List to store data for the DataFrame

    counter = 0  # Initialize the counter
    for file_name in hearing_list:
        counter += 1  # Increment the counter
        print(f"Processing hearing {counter} of {len(hearing_list)}: {file_name}")

        # Subset the DataFrame to only include rows from the current file
        temp_df = main_df[main_df['file_name'] == file_name + ".txt"]
        speeches = " ".join(temp_df['speech'].dropna())
        all_nwords = len(speeches.split()) or 1  # Avoid division by zero

        # Dictionary to hold term data for the current hearing
        hearing_data = {"hearing": file_name}

        for term in the_con_dict:
            term_count = speeches.count(term)
            relative_freq = round(term_count / all_nwords * 1000000, 3)
            hearing_data[con_dict_trans[term]] = (term_count, relative_freq)

        all_data.append(hearing_data)
        print(f"Completed hearing {file_name}. {counter} out of {len(hearing_list)} processed.")
    
    # Convert the list of dictionaries to a DataFrame
    df_result = pd.DataFrame(all_data)
    return df_result

def main():
    # Set paths and initialize variables
    hearing_list = [
        f[:-4] for f in os.listdir(project_path("data"))
        if f.startswith("CHRG") and f.endswith(".csv") and "_sentcontext" not in f
    ]

    # Load and merge data
    dfs = []
    for file_name in hearing_list:
        file_path = project_path("data", f"{file_name}.csv")
        df = pd.read_csv(file_path)
        dfs.append(df)
    merged_df = pd.concat(dfs, ignore_index=True)

    # Load bigram model
    bigram = Phraser.load(str(project_path("scripts", "con_dicts", "bigram_phraser.pkl")))

    # Create dictionaries
    dont_stem = ['constitute', 'constituting', 'constitutes', 'takings', 'presentment', 'originalism', 'originalist', 'federalism']
    the_con_dict, all_con_dicts, con_dict_trans = create_dict(6, bigram, dont_stem)

    # Process hearings
    df_freq = process_hearings(hearing_list, merged_df, the_con_dict, con_dict_trans)

    # Save the DataFrame
    pickle_file = project_path("scripts", "con_dicts", "df_freq_con.pkl")
    with open(pickle_file, "wb") as file:
        pickle.dump(df_freq, file)
    print("Processing complete. DataFrame saved.")

if __name__ == "__main__":
    main()
