"""
Retrieves relevant lessons from ChromaDB at runtime.
Injects learned reasoning patterns into Analyst, Clinician, Critic prompts.
"""


class MemoryRetriever:
    """Fetches lessons from MemoryStore; formats as prompt context for agents."""

    def __init__(self, memory_store):
        self.memory_store = memory_store

    def get_context_for_agent(
        self, agent_name: str, user_input: str, input_type: str
    ) -> str | None:
        """
        Retrieve relevant lessons for the given agent and format as context string.
        agent_name: "analyst", "clinician", or "critic"
        Returns a formatted context string, or None if no lessons found.
        """
        # ChromaDB semantic search: user_input as query; optionally filter by input_type
        lessons = self.memory_store.retrieve_lessons(
            query_text=user_input,
            input_type=input_type,
            top_k=5,
        )

        if not lessons:
            return None

        context = "LEARNED MEDICAL KNOWLEDGE (from previous successful analyses):\n\n"
        for lesson in lessons:
            topic = lesson.get("topic", "")
            confidence = lesson.get("confidence", "medium")
            rule = lesson.get("rule", "")
            example_values = lesson.get("example_values", "")

            context += f'<lesson topic="{topic}" confidence="{confidence}">\n'
            context += f"  {rule}\n"
            if example_values:
                context += f"  Example values: {example_values}\n"
            context += "</lesson>\n\n"

        context += "Apply these lessons where relevant to the current analysis.\n"

        # Truncate to top 3 lessons if over 2000 chars
        if len(context) > 2000:
            context = "LEARNED MEDICAL KNOWLEDGE (from previous successful analyses):\n\n"
            for lesson in lessons[:3]:
                topic = lesson.get("topic", "")
                confidence = lesson.get("confidence", "medium")
                rule = lesson.get("rule", "")
                example_values = lesson.get("example_values", "")

                context += f'<lesson topic="{topic}" confidence="{confidence}">\n'
                context += f"  {rule}\n"
                if example_values:
                    context += f"  Example values: {example_values}\n"
                context += "</lesson>\n\n"
            context += "Apply these lessons where relevant to the current analysis.\n"

        return context
