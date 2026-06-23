import arxiv
import json
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

app = Server("arxiv-search-server")
arxiv_client = arxiv.Client()

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    """Expose tools that agents can call through MCP."""
    return [
        types.Tool(
            name="search_papers",
            description="Search ArXiv for research papers on a given topic",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Research topic to search for"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of papers to return",
                        "default": 8
                    }
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="get_paper_details",
            description="Get full details of a paper by its ArXiv ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "paper_id": {
                        "type": "string",
                        "description": "ArXiv paper ID e.g. 2301.07041"
                    }
                },
                "required": ["paper_id"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict):
    """Handle tool calls from agents."""

    if name == "search_papers":
        query = arguments["query"]
        max_results = arguments.get("max_results", 8)

        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )

        papers = []
        for paper in arxiv_client.results(search):
            papers.append({
                "id": paper.entry_id.split("/")[-1],
                "title": paper.title,
                "authors": [str(a) for a in paper.authors[:3]],
                "abstract": paper.summary[:600],
                "published": str(paper.published.date()),
                "url": paper.entry_id
            })

        return [types.TextContent(type="text", text=json.dumps(papers, indent=2))]

    elif name == "get_paper_details":
        paper_id = arguments["paper_id"]
        search = arxiv.Search(id_list=[paper_id])
        paper = next(arxiv_client.results(search))

        details = {
            "title": paper.title,
            "authors": [str(a) for a in paper.authors],
            "abstract": paper.summary,
            "published": str(paper.published.date()),
            "url": paper.entry_id,
            "categories": paper.categories
        }

        return [types.TextContent(type="text", text=json.dumps(details, indent=2))]

async def run_server():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_server())
