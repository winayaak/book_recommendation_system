"""Experiment 1: Study of NLP applications and formulate a mini-project use case."""
def get_project_use_case():
    return {
        "project": "NLP-Based Book Recommendation System",
        "problem_statement": "Readers face difficulty finding books matching their interests across large catalogs.",
        "nlp_applications": {
            "Text categorization": "Use genre and category metadata to filter books.",
            "Text similarity": "Compare book descriptions using TF-IDF and cosine similarity.",
            "Information retrieval": "Search title, author, genre, and description.",
            "Text preprocessing": "Normalize and filter book text before feature extraction."
        },
        "techniques": ["text preprocessing", "tokenization", "TF-IDF", "cosine similarity", "word frequency analysis"]
    }
