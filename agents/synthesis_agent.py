import json
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class SynthesisAgent:
    """
    Synthesis Agent: Reads ranked papers and writes a structured research report.
    This is the final step in the pipeline — the main value delivered to the user.
    """

    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"
        self.name = "SynthesisAgent"

    def synthesize(self, query: str, papers: list) -> str:
        """
        Generate a structured markdown research report from ranked papers.
        """
        papers_text = json.dumps(papers, indent=2)

        prompt = f"""You are an expert research synthesizer helping a user understand the academic literature on: "{query}"

Based on these top papers:
{papers_text}

Write a comprehensive research synthesis report in Markdown. Use exactly this structure:

# Research Synthesis: {query}

## Overview
(2-3 sentences: what is this field about, based on the papers)

## Key Findings
(5-7 bullet points of the most important findings across all papers)

## Paper Summaries
(For each paper: bold title, authors, date, and 2-3 sentence summary of its contribution)

## Common Themes
(What approaches or ideas appear across multiple papers?)

## Research Gaps
(What questions remain unanswered? What should future work explore?)

## Recommended Reading Order
(Which paper to read first and why — write for a beginner)

Be specific. Cite paper titles when making claims. Write clearly for an intelligent non-expert."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=2000
        )

        return response.choices[0].message.content

    def run(self, query: str, papers: list) -> str:
        """Main entry point — called by Orchestrator."""
        print(f"[{self.name}] Synthesizing {len(papers)} papers...")
        report = self.synthesize(query, papers)
        print(f"[{self.name}] Report generated.")
        return report
