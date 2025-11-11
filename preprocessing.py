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

def clean_text(s: str,
               keep_numbers: bool = True,
               expand_contractions: bool = False) -> str:
    if not isinstance(s, str):
        return ""

    # unicode normalize (fix smart quotes/ligatures)
    s = unicodedata.normalize("NFKD", s)

    # optional contractions expansion
    if expand_contractions:
        try:
            import contractions
            s = contractions.fix(s)
        except Exception:
            pass  # if library not installed, skip silently

    # lowercase
    s = s.lower()

    # strip HTML tags
    s = re.sub(r"<[^>]+>", " ", s)

    # keep numbers or not
    if keep_numbers:
        s = re.sub(r"[^a-z0-9\s]", " ", s)   # letters + digits
    else:
        s = re.sub(r"[^a-z\s]", " ", s)      # letters only

    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s
