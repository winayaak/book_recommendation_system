from collections import Counter, defaultdict
import random, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.preprocessing import tokenize, filter_tokens, validate_script, stopword_stem_lemma, morphology, preprocess

def ngram_tokenize(text):
    return re.findall(r"\b\w+\b", str(text).lower())

def build_ngram_model(text, n=2):
    words = ngram_tokenize(text)
    model = defaultdict(Counter)
    if n < 1 or len(words) < n:
        return {}
    for i in range(len(words)-n+1):
        context = tuple(words[i:i+n-1])
        model[context][words[i+n-1]] += 1
    return dict(model)

def predict_next_word(model, context):
    toks = tuple(ngram_tokenize(context))
    if not toks:
        return None
    # For unigram model, empty context maps to all tokens; normal UI uses n >= 2.
    return model[toks].most_common(1)[0][0] if toks in model else None

def generate_text(model, seed, num_words=10, seed_value=7):
    generated = ngram_tokenize(seed)
    if not generated or not model:
        return " ".join(generated)
    n_minus_one = len(next(iter(model.keys())))
    rng = random.Random(seed_value)
    for _ in range(max(0, int(num_words))):
        context = tuple(generated[-n_minus_one:]) if n_minus_one else tuple()
        if context not in model:
            break
        choices = model[context]
        generated.append(rng.choices(list(choices), weights=list(choices.values()))[0])
    return " ".join(generated)

def compare_texts(doc1, doc2):
    try:
        matrix = TfidfVectorizer().fit_transform([doc1, doc2])
        score = float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])
    except ValueError:
        score = 0.0
    if score >= .75: level = "Highly Similar"
    elif score >= .50: level = "Moderately Similar"
    elif score >= .25: level = "Slightly Similar"
    else: level = "Not Similar"
    return score, level

def word_frequencies(text):
    return Counter(w for w in re.findall(r"[a-zA-Z]+", str(text).lower()) if w not in {"a","an","the","and","or","is","are","of","to","in","for"})
