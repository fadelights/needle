import json
from pathlib import Path

import gradio as gr
import requests

from balequeue.config import settings

API_BASE_URL = f"http://127.0.0.1:{settings.balequeue_port}/api"


def _upload(file, business_id: str):
    if not file:
        return "⚠️ Please select a file"

    if not business_id.strip():
        return "⚠️ Please enter a business ID"

    try:
        with open(file, "rb") as f:
            files = {"file": (Path(file).name, f, "text/plain")}
            response = requests.post(
                f"{API_BASE_URL}/upload/{business_id.strip()}",
                files=files,
                timeout=60,
            )

        if response.status_code == 201:
            data = response.json()
            return f"✅ {data['message']}"
        else:
            data = response.json()
            return f"❌ Error: {data.get('detail', 'Unknown error.')}"

    except requests.exceptions.ConnectionError:
        return "❌ Error: Cannot connect to API server. Is it running?"
    except requests.exceptions.Timeout:
        return "❌ Error: Request timed out. Please try again."
    except Exception as e:
        return f"❌ Error: {str(e)}"


def _query(query: str, business_id: str):
    if not query.strip():
        return "⚠️ Please enter a query", ""

    if not business_id.strip():
        return "⚠️ Please enter a business ID", ""

    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={
                "query": query.strip(),
                "business_id": business_id.strip(),
            },
            timeout=30,
        )

        if response.status_code == 200:
            data = response.json()
            answer = data.get("answer", "No answer generated.")

            source_chunks = data.get("source_chunks", [])
            if source_chunks:
                sources_text = "**📚 Source Documents:**\n\n"
                for chunk in source_chunks:
                    file_path = chunk.get("file_path", "Unknown")
                    score = chunk.get("score", 0)
                    content = chunk.get("content", "")
                    sources_text += (
                        f"**File:** {file_path} | **Similarity:** {score:.1%}\n\n"
                    )
                    sources_text += f"```\n{content}\n```\n\n---\n\n"
            else:
                sources_text = "No source documents found."

            return answer, sources_text

        else:
            data = response.json()
            error_msg = data.get("detail", "Unknown error.")
            return f"❌ Error: {error_msg}", ""

    except requests.exceptions.ConnectionError:
        return "❌ Error: Cannot connect to API server. Is it running?", ""
    except requests.exceptions.Timeout:
        return "❌ Error: Request timed out. Please try again.", ""
    except Exception as e:
        return f"❌ Error: {str(e)}", ""


def create_app():
    """Create and return the Gradio interface."""
    with gr.Blocks(title="Balequeue QA Service") as client:
        gr.Markdown("""
        # Balequeue QA Service
        Intelligent business QA powered by Haystack
        """)

        with gr.Tabs():
            with gr.TabItem("Upload Documents"):
                gr.Markdown(
                    "### Upload documents for your business to the knowledge base"
                )

                with gr.Group():
                    business_id = gr.Textbox(
                        label="Business ID",
                        placeholder="e.g., 42",
                        info="Unique identifier for your business",
                    )
                    file_upload = gr.File(
                        label="Upload Text Document",
                        file_types=[".txt"],
                        file_count="single",
                    )

                upload_btn = gr.Button("Upload", variant="primary")
                upload_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    value="Ready to upload...",
                )

                upload_btn.click(
                    fn=_upload,
                    inputs=[file_upload, business_id],
                    outputs=upload_status,
                )

            with gr.TabItem("Query Knowledge Base"):
                gr.Markdown("### Ask questions about your business documents")

                with gr.Group():
                    business_id_user = gr.Textbox(
                        label="Business ID",
                        placeholder="e.g., 42",
                        info="Unique identifier for your business",
                    )
                    query_input = gr.Textbox(
                        label="Your Question",
                        placeholder="When will I get paid?",
                        lines=2,
                        info="Enter your question here",
                    )

                query_btn = gr.Button("Search & Answer", variant="primary")

                answer_output = gr.Textbox(
                    label="Answer",
                    interactive=False,
                    lines=4,
                )

                sources_output = gr.Markdown(
                    label="Sources",
                    value="Nothing to see here yet!",
                )

                query_btn.click(
                    fn=_query,
                    inputs=[query_input, business_id_user],
                    outputs=[answer_output, sources_output],
                )

    return client


client = create_app()

if __name__ == "__main__":
    # For standalone testing
    client.launch(server_name="127.0.0.1", server_port=8000, share=False)
