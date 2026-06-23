from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from agents.search_agent import SearchAgent
from agents.synthesis_agent import SynthesisAgent
import json

# Initialize our existing agents
search_agent = SearchAgent()
synthesis_agent = SynthesisAgent()

# Define ADK Tools 

def search_and_rank_papers(query: str) -> str:
    """
    Search ArXiv for research papers on a given topic and rank them by relevance.
    
    Args:
        query: The research topic to search for
        
    Returns:
        A JSON string containing ranked papers with relevance scores
    """
    papers = search_agent.run(query)
    return json.dumps(papers, indent=2)

def synthesize_research_report(query: str, papers_json: str) -> str:
    """
    Synthesize a structured research report from a list of ranked papers.
    
    Args:
        query: The original research topic
        papers_json: JSON string of ranked papers from search_and_rank_papers
        
    Returns:
        A structured markdown research report
    """
    papers = json.loads(papers_json)
    report = synthesis_agent.run(query, papers)
    return report

# Wrap functions as ADK Tools 
search_tool = FunctionTool(search_and_rank_papers)
synthesis_tool = FunctionTool(synthesize_research_report)

# Define ADK Sub-Agents 
search_adk_agent = Agent(
    name="search_agent",
    model="gemini-2.0-flash",
    description="Searches ArXiv for research papers and ranks them by relevance to the user query.",
    instruction="""You are a research search specialist. 
    When given a research topic, use the search_and_rank_papers tool to find and rank relevant papers.
    Always return the complete JSON output from the tool.""",
    tools=[search_tool]
)

synthesis_adk_agent = Agent(
    name="synthesis_agent", 
    model="gemini-2.0-flash",
    description="Synthesizes ranked research papers into a structured literature review report.",
    instruction="""You are a research synthesis specialist.
    When given a topic and papers JSON, use the synthesize_research_report tool to generate a comprehensive report.
    Always return the complete markdown report from the tool.""",
    tools=[synthesis_tool]
)

# Define ADK Orchestrator Agent 
orchestrator_adk_agent = Agent(
    name="orchestrator_agent",
    model="gemini-2.0-flash",
    description="Orchestrates the research pipeline by coordinating search and synthesis agents.",
    instruction="""You are the master research orchestrator for ResearchNavigator.
    
    When a user provides a research topic, follow these steps:
    1. Use search_and_rank_papers to find relevant papers on the topic
    2. Use synthesize_research_report with the topic and papers JSON to generate a report
    3. Return both the ranked papers JSON and the synthesis report
    
    Always complete both steps before responding.""",
    tools=[search_tool, synthesis_tool],
    sub_agents=[search_adk_agent, synthesis_adk_agent]
)
