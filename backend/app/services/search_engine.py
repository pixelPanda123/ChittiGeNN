from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SearchEngine:
    def __init__(self, max_features: int = 20000):
        """
        Initialize the SearchEngine with a TF-IDF vectorizer.
        :param max_features: Maximum number of features for TF-IDF.
        """
        self.vectorizer = TfidfVectorizer(max_features=max_features)

    def search(
        self, 
        db: Session, 
        query: str, 
        top_k: Optional[int] = 10
    ) -> List[Dict[str, Any]]:
        """
        Perform a search on ContentChunk texts in the database.
        :param db: SQLAlchemy DB session.
        :param query: Search query string.
        :param top_k: Number of top results to return.
        :return: List of results with chunk_id, document_id, text, and score.
        """
        # Lazy import to avoid circular dependencies
        from app.models.document import ContentChunk

        # Fetch all content chunks
        chunks = db.query(ContentChunk).all()
        corpus = [chunk.content or "" for chunk in chunks]

        if not corpus:
            return []

        # Compute TF-IDF vectors
        X = self.vectorizer.fit_transform(corpus)
        q_vec = self.vectorizer.transform([query])

        # Compute cosine similarity between query and corpus
        sims = cosine_similarity(q_vec, X)[0]

        # Sort by similarity and select top_k
        idxs = sims.argsort()[::-1][: top_k]

        results: List[Dict[str, Any]] = []
        for i in idxs:
            ch = chunks[i]
            results.append({
                "chunk_id": ch.id,
                "document_id": ch.document_id,
                "text": ch.content[:1000],
                "score": float(sims[i]),
            })

        return results


# Singleton instance for app use
search_engine = SearchEngine()
