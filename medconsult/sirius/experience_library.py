"""
Experience library: stores raw analysis chains as JSON.
Good/ (score ≥ 3) and bad/ chains; stats; input type classification.
"""

import json
import datetime
from pathlib import Path


class ExperienceLibrary:
    """Stores raw chains as JSON; good/ (score≥3) vs bad/; classifies input type."""

    def __init__(self, base_dir="experience_library"):
        self.base_dir = Path(base_dir)
        self.good_dir = self.base_dir / "good"
        self.bad_dir = self.base_dir / "bad"
        self.stats_file = self.base_dir / "stats.json"

        self.good_dir.mkdir(parents=True, exist_ok=True)
        self.bad_dir.mkdir(parents=True, exist_ok=True)

    def save_chain(self, chain_data: dict, score: int) -> str:
        """Save a full chain to good/ (score >= 3) or bad/. Returns filepath."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        input_type = chain_data.get("metadata", {}).get("input_type", "unknown")
        filename = f"{timestamp}_{input_type}_score{score}.json"

        folder = self.good_dir if score >= 3 else self.bad_dir
        filepath = folder / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(chain_data, f, indent=2, default=str)

        self._update_stats(score)
        return str(filepath)

    def get_good_chains(self, input_type: str = None, limit: int = 10) -> list:
        """Return good chains sorted by score descending."""
        chains = []
        for filepath in sorted(
            self.good_dir.glob("*.json"),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        ):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    chain = json.load(f)
                if input_type is None or chain.get("metadata", {}).get("input_type") == input_type:
                    chains.append(chain)
                if len(chains) >= limit:
                    break
            except (json.JSONDecodeError, IOError):
                continue

        return sorted(
            chains,
            key=lambda x: x.get("evaluation", {}).get("score", 0),
            reverse=True,
        )

    def get_stats(self) -> dict:
        """Return saved stats dict."""
        try:
            with open(self.stats_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError, FileNotFoundError):
            return {}

    def classify_input_type(self, text: str) -> str:
        """Keyword-based: lab_report, clinical_note, imaging, or general."""
        text_lower = text.lower()

        if any(kw in text_lower for kw in [
            "x-ray", "xray", "ct scan", "mri", "ultrasound",
            "chest x", "radiograph", "imaging",
        ]):
            return "imaging"

        if any(kw in text_lower for kw in [
            "discharge summary", "clinical note", "progress note", "h&p",
            "chief complaint", "history of present illness",
            "admission date", "discharge diagnosis",
        ]):
            return "clinical_note"

        if any(kw in text_lower for kw in [
            "cbc", "blood count", "lab report", "reference range",
            "hemoglobin", "hematocrit", "wbc", "rbc", "platelet",
            "glucose", "creatinine", "bmp", "cmp", "lab result",
        ]):
            return "lab_report"

        return "general"

    def _update_stats(self, score: int):
        stats = self.get_stats()
        total = stats.get("total_chains", 0) + 1
        stats["total_chains"] = total
        stats["good_chains"] = stats.get("good_chains", 0) + (1 if score >= 3 else 0)
        stats["bad_chains"] = stats.get("bad_chains", 0) + (1 if score < 3 else 0)
        old_avg = stats.get("average_score", 0.0)
        stats["average_score"] = round(((old_avg * (total - 1)) + score) / total, 2)

        with open(self.stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
