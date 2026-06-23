import arxiv
import json
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class SearchAgent:
    """
    Search Agent: Finds and ranks research papers from ArXiv.
    Uses Groq (Llama 3.3 70B) to score papers by relevance.
    This agent is called by the Orchestrator.
    """

    def __init__(self):
        # Initialize Groq client with API key from .env
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"
        self.name = "SearchAgent"
        # Initialize arxiv client (new API style)
        self.arxiv_client = arxiv.Client()

    def search_arxiv(self, query: str, max_results: int = 8) -> list:
        """Search ArXiv — completely free, no API key needed."""
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )

            papers = []
            for paper in self.arxiv_client.results(search):
                papers.append({
                    "id": paper.entry_id.split("/")[-1],
                    "title": paper.title,
                    "authors": [str(a) for a in paper.authors[:3]],
                    "abstract": paper.summary[:600],
                    "published": str(paper.published.date()),
                    "url": paper.entry_id
                })

            return papers

        except Exception as e:
            print(f"[{self.name}] ArXiv search error: {e}")
            return []

    def rank_papers(self, query: str, papers: list) -> list:
        """
        Use Groq/Llama to rank papers by relevance.
        Returns top 5 papers with relevance scores and reasons.
        """
        papers_text = json.dumps(papers, indent=2)

        prompt = f"""You are a research assistant. A user is researching: "{query}"

Here are {len(papers)} papers found on ArXiv:
{papers_text}

Rank these papers from most to least relevant to the user's query.
For each paper give a relevance score from 1-10 and a one-sentence reason.

Return ONLY a valid JSON array like this, nothing else:
[
  {{
    "id": "paper_id",
    "title": "paper title",
    "authors": ["author1", "author2"],
    "abstract": "abstract text",
    "published": "date",
    "url": "url",
    "relevance_score": 9,
    "reason": "one sentence explaining relevance"
  }}
]"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        text = response.choices[0].message.content.strip()

        try:
            # Strip markdown code fences if present
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            ranked = json.loads(text.strip())
            ranked.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
            return ranked[:5]
        except Exception as e:
            print(f"[{self.name}] Ranking parse error: {e}")
            return papers[:5]

    def run(self, query: str) -> list:
        """Main entry point — called by Orchestrator."""
        print(f"[{self.name}] Searching ArXiv for: {query}")
        papers = self.search_arxiv(query)

        # If no papers found return empty list
        if not papers:
            print(f"[{self.name}] No papers found for query: {query}")
            return []

        print(f"[{self.name}] Found {len(papers)} papers. Ranking with Groq...")
        ranked = self.rank_papers(query, papers)
        print(f"[{self.name}] Done. Top {len(ranked)} papers selected.")
        return ranked

