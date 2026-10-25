from src.recommender import load_books, BookRecommender
from src.preprocessing import tokenize, filter_tokens, validate_script, morphology
from src.experiment_tools import compare_texts, build_ngram_model

def test_sample_dataset_loads():
    books = load_books()
    assert len(books) >= 10
    assert {"title", "authors", "genre", "description"}.issubset(books.columns)

def test_recommendations_exclude_selected_book():
    engine = BookRecommender()
    title = engine.books.iloc[0]["title"]
    result = engine.recommend(title, 3)
    assert title not in result["title"].tolist()

def test_preprocessing_functions():
    assert "book" in tokenize("Book 123")
    assert "the" not in filter_tokens(["the", "book"])
    assert validate_script("hello world").startswith("Valid")
    assert morphology("unhappy")[0]["prefix"] == ["un"]

def test_similarity_and_ngram():
    score, _ = compare_texts("space science fiction", "space science")
    assert 0 <= score <= 1
    assert build_ngram_model("i like books i like stories", 2)
