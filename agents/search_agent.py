import arxiv
import json
import os
from datetime import datetime
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class SearchAgent:
    """
    Search Agent: Finds and ranks research papers from ArXiv.
    Uses Groq (Llama 3.3 70B) to score papers by relevance.
    Falls back to smaller models if rate limit is hit.
    Supports year filtering and field category filtering.
    This agent is called by the Orchestrator.
    """

    def __init__(self):
        # Initialize Groq client with API key from .env
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        # Try main model first, fall back to smaller if rate limited
        self.models = [
            "llama-3.3-70b-versatile",
            "llama3-8b-8192",
            "gemma2-9b-it"
        ]
        self.name = "SearchAgent"
        # Initialize arxiv client (new API style)
        self.arxiv_client = arxiv.Client()

    def search_arxiv(self, query: str, max_results: int = 8, year_start: int = 2000, year_end: int = 2026, category: str = "") -> list:
        """
        Search ArXiv and filter by year range and category.
        Fetches extra results to account for year filtering.
        """
        try:
            # Build query with category filter if provided
            if category:
                # Combine user query with category filter
                cats = category.strip().split()
                cat_filter = " OR ".join([f"cat:{c}" for c in cats])
                full_query = f"({query}) AND ({cat_filter})"
            else:
                full_query = query

            print(f"[{self.name}] Full ArXiv query: {full_query}")

            # Fetch more results to account for year filtering
            search = arxiv.Search(
                query=full_query,
                max_results=max_results * 3,
                sort_by=arxiv.SortCriterion.Relevance
            )

            papers = []
            for paper in self.arxiv_client.results(search):
                # Filter by year range
                pub_year = paper.published.year
                if pub_year < year_start or pub_year > year_end:
                    continue

                papers.append({
                    "id": paper.entry_id.split("/")[-1],
                    "title": paper.title,
                    "authors": [str(a) for a in paper.authors[:3]],
                    "abstract": paper.summary[:600],
                    "published": str(paper.published.date()),
                    "url": paper.entry_id
                })

                if len(papers) >= max_results:
                    break

            return papers

        except Exception as e:
            print(f"[{self.name}] ArXiv search error: {e}")
            return []

    def rank_papers(self, query: str, papers: list) -> list:
        """
        Use Groq/Llama to rank papers by relevance.
        Returns top 5 papers with relevance scores and reasons.
        Tries multiple models if rate limit is hit.
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

        response = None
        for model in self.models:
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                )
                print(f"[{self.name}] Using model: {model}")
                break
            except Exception as e:
                print(f"[{self.name}] Model {model} failed: {e}. Trying next...")
                continue

        if not response:
            print(f"[{self.name}] All models failed, returning unranked papers.")
            return papers[:5]

        text = response.choices[0].message.content.strip()

        try:
            
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

    def run(self, query: str, max_results: int = 8, year_start: int = 2000, year_end: int = 2026, category: str = "") -> list:
        """Main entry point — called by Orchestrator."""
        print(f"[{self.name}] Searching ArXiv for: {query} ({year_start}-{year_end}) category: {category}")
        papers = self.search_arxiv(query, max_results=max_results, year_start=year_start, year_end=year_end, category=category)

        if not papers:
            print(f"[{self.name}] No papers found for query: {query}")
            return []

        print(f"[{self.name}] Found {len(papers)} papers. Ranking with Groq...")
        ranked = self.rank_papers(query, papers)
        print(f"[{self.name}] Done. Top {len(ranked)} papers selected.")
        return ranked
