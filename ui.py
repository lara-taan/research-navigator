import gradio as gr
from agents.orchestrator import OrchestratorAgent

# Load orchestrator once at startup
orchestrator = OrchestratorAgent()

def run_agent(user_query, progress=gr.Progress()):
    """Called when user clicks the button. Runs the full agent pipeline."""

    if not user_query.strip():
        return "⚠️ Please enter a research topic.", "", ""

    progress(0.1, desc="🧠 Understanding your query...")
    result = orchestrator.run(
        user_query,
        progress_callback=lambda msg: progress(0.5, desc=msg)
    )
    progress(1.0, desc="✅ Done!")

    if not result["success"]:
        return f"❌ {result['error']}", "", ""

    # Format ranked papers for display
    papers_md = ""
    for i, paper in enumerate(result["papers"], 1):
        score = paper.get("relevance_score", "N/A")
        reason = paper.get("reason", "")
        papers_md += f"""### {i}. {paper['title']}
- 👥 **Authors:** {', '.join(paper['authors'])}
- 📅 **Published:** {paper['published']}
- ⭐ **Relevance Score:** {score}/10
- 💡 **Why:** {reason}
- 🔗 [View on ArXiv]({paper['url']})

---
"""

    status = f"✅ Found and analyzed **{result['papers_found']} papers** on: **{result['topic']}**"
    return status, result["report"], papers_md


# Build UI
with gr.Blocks(title="ResearchNavigator", theme=gr.themes.Soft(), css="""
    .built-with { display: none !important; }
    footer { display: none !important; }
    .svelte-1ipelgc { display: none !important; }
""") as demo:

    gr.Markdown("""
# ResearchNavigator
### Multi-Agent Literature Review System

Enter any research topic → AI agents automatically find, rank, and synthesize the most relevant papers for you.
    """)

    with gr.Row():
        with gr.Column(scale=4):
            query_input = gr.Textbox(
                label="Research Topic",
                placeholder="e.g. 'reinforcement learning from human feedback'",
                lines=2
            )
        with gr.Column(scale=1):
            submit_btn = gr.Button("Research This", variant="primary", size="lg")

    status_out = gr.Markdown()

    with gr.Tabs():
        with gr.Tab("Synthesis Report"):
            report_out = gr.Markdown()
        with gr.Tab("Ranked Papers"):
            papers_out = gr.Markdown()

    gr.Examples(
        examples=[
            ["reinforcement learning from human feedback"],
            ["large language model reasoning"],
            ["diffusion models image generation"],
            ["graph neural networks drug discovery"],
            ["federated learning privacy"],
        ],
        inputs=query_input
    )

    submit_btn.click(
        fn=run_agent,
        inputs=query_input,
        outputs=[status_out, report_out, papers_out]
    )

if __name__ == "__main__":
    demo.launch(
        show_error=True,
        show_api=False,
        footer="",
    )
