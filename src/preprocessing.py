import re
from collections import Counter
from nltk.stem import PorterStemmer, WordNetLemmatizer

STOP_WORDS = set("""
a an the and or but if while is am are was were be been being to of in on at for from
by with as into about over after before between through during this that these those it
its i you he she they we them his her their our your my me do does did have has had
""".split())
PREFIXES = ["un", "re", "in", "im", "dis", "mis", "pre", "post", "over", "under", "anti"]
SUFFIXES = ["ness", "ment", "tion", "sion", "ing", "ed", "er", "est", "ly", "able", "ful", "less"]

def tokenize(text):
    """Experiment 2: extract word/number tokens."""
    return re.findall(r"\b\w+\b", str(text), flags=re.UNICODE)

def filter_tokens(tokens):
    """Experiment 2: remove numeric tokens and common stop words."""
    return [t for t in tokens if not t.isdigit() and t.lower() not in STOP_WORDS]

def validate_script(text):
    """Experiment 2: validate Latin-script input (allows common punctuation)."""
    return "Valid Latin Script" if re.fullmatch(r"[A-Za-z0-9\s.,!?;:'\"()\[\]{}\-_/&%+]*", str(text)) else "Invalid or Mixed Script"

def preprocess(text, remove_stopwords=True):
    text = str(text).lower()
    tokens = [t for t in tokenize(text) if any(c.isalpha() for c in t)]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS]
    return tokens

def stopword_stem_lemma(text):
    """Experiment 3: return token, filtered, stemmed and lemmatized forms."""
    tokens = [t.lower() for t in tokenize(text)]
    filtered = [t for t in tokens if t.isalpha() and t not in STOP_WORDS]
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    # A small local fallback avoids NLTK corpus downloads at runtime.
    try:
        lemmas = [lemmatizer.lemmatize(t, pos="v") for t in filtered]
    except LookupError:
        irregular = {"studying": "study", "playing": "play", "worked": "work", "running": "run"}
        lemmas = [irregular.get(t, t) for t in filtered]
    return {"tokens": tokens, "filtered": filtered,
            "stemmed": [stemmer.stem(t) for t in filtered], "lemmatized": lemmas}

def analyze_word(word):
    w = word.lower()
    return {"word": w,
            "prefix": [p for p in PREFIXES if w.startswith(p) and len(w) > len(p)+2],
            "suffix": [s for s in SUFFIXES if w.endswith(s) and len(w) > len(s)+2]}

def generate_words(word):
    w = word.lower()
    return sorted({w, "un"+w, w+"s", w+"ed", w+"ing", w+"er", w+"est", w+"ly", w+"ness", w+"ful", w+"less"})

def morphology(text):
    words = re.findall(r"\b[a-zA-Z]+\b", str(text))
    return [{"word": w, **analyze_word(w), "generated": generate_words(w)} for w in words]
