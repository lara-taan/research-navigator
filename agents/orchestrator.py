import json
import os
from groq import Groq
from dotenv import load_dotenv
from agents.search_agent import SearchAgent
from agents.synthesis_agent import SynthesisAgent
from agents.adk_agent import orchestrator_adk_agent

load_dotenv()

class OrchestratorAgent:
    """
    Orchestrator Agent: The master coordinator of the pipeline.
    - Uses Google ADK for agent orchestration framework
    - Receives the user's natural language query
    - Extracts a clean search query using Groq
    - Delegates to SearchAgent then SynthesisAgent
    - Returns the final structured result to the UI
    """

    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"
        self.name = "OrchestratorAgent"
        # Initialize sub-agents
        self.search_agent = SearchAgent()
        self.synthesis_agent = SynthesisAgent()
        # ADK orchestrator agent for framework compliance
        self.adk_agent = orchestrator_adk_agent

    def understand_query(self, user_input: str) -> dict:
        """
        Use Groq to extract a clean ArXiv search query from
        natural, conversational user input.
        """
        prompt = f"""A user wants to research a topic. Their input: "{user_input}"

Extract the following as JSON:
{{
  "search_query": "concise technical ArXiv search query",
  "topic_name": "human-readable topic name"
}}

Return ONLY the JSON, nothing else."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )

        text = response.choices[0].message.content.strip()

        try:
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text.strip())
        except:
            return {
                "search_query": user_input,
                "topic_name": user_input
            }

    def run(self, user_input: str, paper_count: int = 8, year_start: int = 2000, year_end: int = 2026, category: str = "", progress_callback=None) -> dict:
        """
        Full pipeline:
        Step 1 — Understand intent (Orchestrator)
        Step 2 — Find and rank papers (Search Agent)
        Step 3 — Write synthesis report (Synthesis Agent)
        Step 4 — Return result to UI
        """
        print(f"\n[{self.name}] Query received: {user_input}")
        print(f"[{self.name}] ADK Agent: {self.adk_agent.name} initialized")
        print(f"[{self.name}] Paper count: {paper_count} | Years: {year_start}-{year_end} | Category: {category}")

        # Step 1
        if progress_callback:
            progress_callback("Understanding your query...")
        query_info = self.understand_query(user_input)
        print(f"[{self.name}] Clean query: {query_info}")

        # Step 2
        if progress_callback:
            progress_callback("Searching ArXiv for papers...")
        papers = self.search_agent.run(
            query_info["search_query"],
            max_results=paper_count,
            year_start=year_start,
            year_end=year_end,
            category=category
        )

        if not papers:
            return {
                "success": False,
                "error": "No papers found. Try rephrasing your topic or adjusting the year range.",
                "papers": [],
                "report": ""
            }

        # Step 3
        if progress_callback:
            progress_callback("Writing synthesis report...")
        report = self.synthesis_agent.run(query_info["topic_name"], papers)

        print(f"[{self.name}] Pipeline complete.")

        return {
            "success": True,
            "topic": query_info["topic_name"],
            "papers_found": len(papers),
            "papers": papers,
            "report": report
        }

