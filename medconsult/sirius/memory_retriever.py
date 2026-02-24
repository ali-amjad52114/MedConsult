"""
Retrieves relevant lessons from ChromaDB at runtime.
Injects learned reasoning patterns into Analyst, Clinician, Critic prompts.
"""


class MemoryRetriever:
    """Fetches lessons from MemoryStore; formats as prompt context for agents."""

    def __init__(self, memory_store):
        self.memory_store = memory_store
        self.hits = {"analyst": 0, "clinician": 0, "critic": 0}

    def get_relevant_lessons(self, query_text: str, agent_name=None, n=5) -> str | None:
        """Retrieve lessons, optionally filtered by agent_name."""
        
        if agent_name:
            lessons = self.memory_store.query_for_agent(agent_name, query_text, n)
            if agent_name in self.hits:
                self.hits[agent_name] += 1
        else:
            # Fallback if no agent name is provided
            lessons = self.memory_store.query_for_agent("general", query_text, n)

        if not lessons:
            return None

        headers = {
            "analyst": "EXTRACTION PATTERNS FROM PAST CASES",
            "clinician": "REASONING GUIDANCE FROM PAST CASES",
            "critic": "COMMUNICATION LESSONS FROM PAST CASES"
        }
        
        icons = {
            "extraction_pattern": "📌",
            "reasoning_chain": "🔗",
            "communication_tip": "💡",
            "pitfall_warning": "⚠️ AVOID:"
        }

        header_text = headers.get(agent_name, "LEARNED MEDICAL KNOWLEDGE")
        context = f"{header_text}:\n\n"
        
        for lesson in lessons:
            l_type = lesson.get("lesson_type", "")
            icon = icons.get(l_type, "•")
            rule = lesson.get("rule", "").strip()
            topic = lesson.get("topic", "")
            
            if rule:
                context += f"{icon} [{topic}] {rule}\n"

        context += "\nApply these lessons where relevant to the current analysis.\n"
        return context

    def get_hit_stats(self) -> dict:
        """Return the number of times lessons were retrieved for each agent."""
        return self.hits
