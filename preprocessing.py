import re

import nltk
from nltk.corpus import stopwords

# Download stopwords if not already
nltk.download("stopwords")


# Preprocess function
stop_words = set(stopwords.words("english"))


def preprocess(text):
    """Remove the useless words from the question"""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)
    return " ".join(words)


# Example usage of this library
#
# from preprcessing import preprocess
# train_df["processed_column"] = train_df["some_column"].apply(preprocess)
