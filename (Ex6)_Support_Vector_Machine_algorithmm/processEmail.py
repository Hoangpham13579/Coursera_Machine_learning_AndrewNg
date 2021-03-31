import regex as re
import string
import pandas as pd
from nltk.stem import PorterStemmer
import numpy as np


# TODO: Preprocessing email before training
def email_processing(content):
    # Processing actions
    # 1) Lower case
    # 2) Strip all HTML: Any expression start with ">" and end with "<" & replace them with space
    # 3) Handle number: Replace number with text "number"
    # 4) Handle URL: Replace URLs with the text "httpaddr"
    # 5) Handle email addresses: Replace email address with the text "emailaddr"
    # 6) Handle dollars $: Replace \$ with "dollar"
    # 7) Remove non-words: Remove non-alphanumeric characters (e.g: "?", ".", ...) and punctuation
    # 8) Word stemming Porter algorithm

    process = content.lower()
    # Means between "<" and ">", NOT exist any "<" or ">"
    process = re.sub(r"<[^<>]>+", " ", process)
    process = re.sub(r"\d+", "number", process)
    process = re.sub(r"(https|http):\S*", "httpaddr", process)
    process = re.sub(r"\S+@\S+", "emailaddr", process)
    process = re.sub(r"[$]", "dollar", process)

    # 7) Remove non-words: Remove non-alphanumeric characters (e.g: "?", ".", ...) and punctuation
    for punctuation in string.punctuation:
        process = process.replace(punctuation, " ")

    # Stemming words using Porter stemmer algorithm
    stemmer = PorterStemmer()
    # Change all only alphanumeric word to Porter shortcut word (as list)
    process = " ".join([stemmer.stem(re.sub('[^a-zA-Z0-9]', '', word)) for word in process.split()])
    # Join each ele inside list with " " (whitespace)
    process = " ".join(process.split())

    # Return string removing heading and trailing characters
    return process.strip()


# TODO: Preprocessing AND filtering index of each word in content from Porter dictionary vocab
def email_processing_and_index_filtering(content, vocab_df):
    # Preprocessing email
    # content = email_processing(content)

    # Filtering index of each word in content from Porter dictionary vocab
    indices = [vocab_df[vocab_df.vocab == word]["index"].values[0] for word in content.split()
            if len(vocab_df[vocab_df.vocab == word]["index"].values > 0)]

    return indices


# TODO: Change all the existing Porter's vocab in email to 1 as in the list of zeros vocab
# Ex return: [0, 0, 0, 1, 0, ...]: word with index 4 in vocab exist in email (4: abl)
def email_features(indices):
    n = 1899
    X = np.zeros((n, 1))
    X[indices] = 1
    return X

