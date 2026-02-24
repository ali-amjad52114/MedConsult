"""
ChromaDB vector store for persistent medical lessons.
Stores distilled reasoning patterns; retrieved at inference for ICL injection.
"""

import chromadb
from datetime import datetime
import uuid

class MemoryStore:
    """ChromaDB vector store: stores/retrieves medical reasoning lessons."""

    def __init__(self, persist_dir="experience_library/chroma_db"):
        self.persist_dir = persist_dir
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="medical_lessons"
        )

    def add_lesson(self, lesson_text, metadata=None):
        """Store a lesson with optional metadata dict."""
        if not lesson_text:
            return

        lesson_id = str(uuid.uuid4())
        
        # Ensure metadata is a dict and values are compatible with Chroma
        valid_metadata = {}
        if metadata:
            for k, v in metadata.items():
                if v is not None:
                    valid_metadata[k] = str(v)

        try:
            self.collection.add(
                ids=[lesson_id],
                documents=[lesson_text],
                metadatas=[valid_metadata] if valid_metadata else None,
            )
        except Exception as e:
            print(f"Warning: Failed to add lesson: {e}")

    def query_for_agent(self, agent_name, query_text, n=5):
        """Query lessons filtered by target_agent. Falls back to unfiltered."""
        count = self.collection.count()
        if count == 0:
            return []

        search_n = min(n, count)
        
        try:
            # Try with filter
            results = self.collection.query(
                query_texts=[query_text],
                n_results=search_n,
                where={"target_agent": agent_name},
                include=["documents", "metadatas"]
            )
            
            # If no results or not enough, fallback
            docs = results.get("documents", [[]])[0]
            if len(docs) == 0:
                print(f"No lessons found for {agent_name}, falling back to unfiltered")
                fallback_n = min(3, count)
                results = self.collection.query(
                    query_texts=[query_text],
                    n_results=fallback_n,
                    include=["documents", "metadatas"]
                )
        except Exception as e:
            print(f"Query error: {e}. Falling back to unfiltered.")
            fallback_n = min(3, count)
            try:
                results = self.collection.query(
                    query_texts=[query_text],
                    n_results=fallback_n,
                    include=["documents", "metadatas"]
                )
            except Exception:
                return []

        lessons = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        
        for doc, meta in zip(docs, metas):
            lesson = {"rule": doc}
            if meta:
                lesson.update(meta)
            lessons.append(lesson)

        return lessons

    def get_lesson_count(self) -> int:
        """Total number of lessons stored."""
        return self.collection.count()

    def clear(self):
        """Delete all lessons (for testing)."""
        try:
            self.client.delete_collection("medical_lessons")
            self.collection = self.client.get_or_create_collection(
                name="medical_lessons"
            )
        except Exception:
            pass
