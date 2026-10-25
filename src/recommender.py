from pathlib import Path
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "sample_books.csv"

def load_books(path=None):
    p = Path(path) if path else DEFAULT_DATA
    df = pd.read_csv(p).fillna("")
    aliases = {"author":"authors", "authors":"authors", "category":"genre", "summary":"description", "overview":"description"}
    for old, new in aliases.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    for col in ["title","authors","genre","description"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str).fillna("")
    df = df[df["title"].str.strip().ne("")].drop_duplicates(subset=["title","authors"]).reset_index(drop=True)
    df["combined_text"] = (df["title"]+" "+df["authors"]+" "+df["genre"]+" "+df["description"]).str.lower()
    df["book_id"] = [f"BK{i+1:05d}" for i in range(len(df))]
    return df

class BookRecommender:
    def __init__(self, books=None):
        self.books = (books.copy() if books is not None else load_books()).reset_index(drop=True)
        if "combined_text" not in self.books:
            self.books["combined_text"] = self.books[["title","authors","genre","description"]].astype(str).agg(" ".join, axis=1)
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=30000, sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(self.books["combined_text"].astype(str))
        self.similarity = cosine_similarity(self.matrix)

    def search(self, query="", genre=None, author=None, min_score=0.0, limit=100, sort="Relevance"):
        df = self.books.copy()
        q = str(query).strip().lower()
        if q:
            # fuzzy-ish search across fields; ranking by TF-IDF query similarity
            qv = self.vectorizer.transform([q])
            scores = cosine_similarity(qv, self.matrix).ravel()
            df["search_score"] = scores
            literal = df.apply(lambda r: q in " ".join(str(r[c]) for c in ["title","authors","genre","description"]).lower(), axis=1)
            df = df[(df["search_score"] >= min_score) | literal]
        else:
            df["search_score"] = 0.0
        if genre and genre != "All genres":
            df = df[df["genre"].str.casefold() == str(genre).casefold()]
        if author and author != "All authors":
            df = df[df["authors"].str.casefold() == str(author).casefold()]
        if sort == "Title A–Z":
            df = df.sort_values("title")
        elif sort == "Author A–Z":
            df = df.sort_values("authors")
        elif sort == "Highest match" and "search_score" in df:
            df = df.sort_values("search_score", ascending=False)
        else:
            df = df.sort_values(["search_score","title"], ascending=[False,True])
        return df.head(limit).copy()

    def recommend(self, title, limit=10, genre=None, exclude_read=None):
        matches = self.books.index[self.books["title"].str.casefold() == str(title).strip().casefold()].tolist()
        if not matches:
            matches = self.books.index[self.books["title"].str.contains(str(title).strip(), case=False, regex=False, na=False)].tolist()
        if not matches:
            return self.books.iloc[0:0].assign(similarity=[], rationale=[])
        idx = matches[0]
        scores = self.similarity[idx].copy()
        if genre and genre != "All genres":
            mask = self.books["genre"].str.casefold().eq(str(genre).casefold()).to_numpy()
            scores[~mask] = -1
        excluded = set(exclude_read or [])
        scores[idx] = -1
        for i, r in self.books.iterrows():
            if r["title"] in excluded: scores[i] = -1
        order = np.argsort(scores)[::-1]
        ids = [i for i in order if scores[i] >= 0][:limit]
        result = self.books.iloc[ids].copy()
        result["similarity"] = [float(scores[i]) for i in ids]
        result["rationale"] = [
            "Matched on " + ", ".join([x for x in [str(r["genre"]), "author" if r["authors"] == self.books.iloc[idx]["authors"] else "", "description themes"] if x])
            for _, r in result.iterrows()
        ]
        return result

    def recommend_for_preferences(self, text, limit=10, genre=None, exclude_titles=None):
        q = str(text).strip()
        if not q:
            return self.books.iloc[0:0].assign(similarity=[], rationale=[])
        scores = cosine_similarity(self.vectorizer.transform([q]), self.matrix).ravel()
        if genre and genre != "All genres":
            scores[~self.books["genre"].str.casefold().eq(str(genre).casefold()).to_numpy()] = -1
        excluded = set(exclude_titles or [])
        for i, title in enumerate(self.books["title"]):
            if title in excluded: scores[i] = -1
        ids = [i for i in np.argsort(scores)[::-1] if scores[i] >= 0][:limit]
        result = self.books.iloc[ids].copy()
        result["similarity"] = [float(scores[i]) for i in ids]
        result["rationale"] = ["Semantic keyword match across title, author, genre, and description" for _ in ids]
        return result

    def similar_books(self, book_index, limit=10):
        scores = self.similarity[int(book_index)].copy()
        scores[int(book_index)] = -1
        ids = [i for i in np.argsort(scores)[::-1] if scores[i] >= 0][:limit]
        result = self.books.iloc[ids].copy()
        result["similarity"] = [float(scores[i]) for i in ids]
        return result
