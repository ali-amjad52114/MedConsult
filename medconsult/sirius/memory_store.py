"""
ChromaDB vector store for persistent medical lessons.
Stores distilled reasoning patterns; retrieved at inference for ICL injection.
"""

import chromadb
from datetime import datetime


class MemoryStore:
    """ChromaDB vector store: stores/retrieves medical reasoning lessons."""

    def __init__(self, persist_dir="experience_library/chroma_db"):
        self.persist_dir = persist_dir
        self.client = chromadb.PersistentClient(path=persist_dir)
        # Uses ChromaDB's default embedding function (all-MiniLM-L6-v2, CPU-friendly)
        self.collection = self.client.get_or_create_collection(
            name="medical_lessons"
        )

    def store_lessons(self, lessons: list) -> int:
        """
        Store lesson dicts. Each lesson has: topic, rule, input_type, example_values, etc.
        Each lesson must have: topic, input_type, rule, example_values, confidence,
        source_score, chain_id.
        Returns the number of lessons successfully stored.
        """
        stored = 0
        for lesson in lessons:
            try:
                rule = lesson.get("rule", "").strip()
                if not rule:
                    continue

                timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
                topic_slug = lesson.get("topic", "unknown").replace(" ", "_").lower()[:30]
                chain_id = str(lesson.get("chain_id", "unknown"))[:20]
                lesson_id = f"{chain_id}_{topic_slug}_{timestamp}"

                metadata = {
                    "topic": str(lesson.get("topic", "")),
                    "input_type": str(lesson.get("input_type", "general")),
                    "example_values": str(lesson.get("example_values", "")),
                    "confidence": str(lesson.get("confidence", "medium")),
                    "source_score": int(lesson.get("source_score", 3)),
                    "chain_id": str(lesson.get("chain_id", "")),
                }

                self.collection.add(
                    ids=[lesson_id],
                    documents=[rule],
                    metadatas=[metadata],
                )
                stored += 1
            except Exception as e:
                print(f"Warning: Failed to store lesson: {e}")

        return stored

    def retrieve_lessons(
        self, query_text: str, input_type: str = None, top_k: int = 5
    ) -> list:
        """
        Query ChromaDB for relevant lessons.
        If input_type is provided, filters by that type.
        Returns a list of dicts with rule + metadata fields.
        """
        count = self.collection.count()
        if count == 0:
            return []

        n = min(top_k, count)
        query_kwargs = {
            "query_texts": [query_text],
            "n_results": n,
            "include": ["documents", "metadatas"],
        }

        if input_type is not None:
            query_kwargs["where"] = {"input_type": {"$eq": input_type}}

        try:
            results = self.collection.query(**query_kwargs)
        except Exception:
            # Filter may yield no matches — retry without filter
            query_kwargs.pop("where", None)
            try:
                results = self.collection.query(**query_kwargs)
            except Exception as e:
                print(f"Warning: Failed to retrieve lessons: {e}")
                return []

        lessons = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        for doc, meta in zip(docs, metas):
            lesson = {"rule": doc}
            lesson.update(meta)
            lessons.append(lesson)

        return lessons

    def get_lesson_count(self) -> int:
        """Total number of lessons stored."""
        return self.collection.count()

    def clear(self):
        """Delete all lessons (for testing)."""
        self.client.delete_collection("medical_lessons")
        self.collection = self.client.get_or_create_collection(
            name="medical_lessons"
        )
