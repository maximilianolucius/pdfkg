#!/usr/bin/env python3
"""
Gradio web application for PDF Knowledge Graph Q&A.

Usage:
    python app.py
"""

import os
import shutil
from pathlib import Path
from typing import Generator

import gradio as gr
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import pipeline modules
from pdfkg.parse_pdf import load_pdf, extract_pages, extract_toc
from pdfkg.topology import build_section_tree
from pdfkg.chunking import build_chunks
from pdfkg.embeds import embed_chunks, build_faiss_index
from pdfkg.figtables import index_figures_tables
from pdfkg.xrefs import extract_mentions, resolve_mentions
from pdfkg.graph import build_graph, export_graph
from pdfkg.report import generate_report
from pdfkg.query import answer_question
from pdfkg.storage import get_storage_backend

import faiss
import pandas as pd
import orjson


# Initialize storage backend (with error handling)
try:
    storage = get_storage_backend()
    storage_type = os.getenv("STORAGE_BACKEND", "arango")
    print(f"✅ Storage backend initialized: {storage_type}")
except Exception as e:
    print(f"⚠️  Storage backend initialization failed: {e}")
    print(f"⚠️  Using file storage as fallback")
    from pdfkg.storage import FileStorage
    storage = FileStorage()
    storage_type = "file"

# Global state
current_pdf_slug = None


def process_pdf(pdf_file, embed_model: str, max_tokens: int, progress=gr.Progress()) -> tuple:
    """
    Process uploaded PDF and build knowledge graph.

    Args:
        pdf_file: Uploaded PDF file
        embed_model: Embedding model name
        max_tokens: Max tokens per chunk
        progress: Gradio progress tracker

    Returns:
        Tuple of (status_message, pdf_dropdown_choices, pdf_dropdown_value, chatbot_visibility, chat_input_visibility)
    """
    global current_pdf_slug

    if pdf_file is None:
        return (
            "Please upload a PDF file.",
            gr.update(),
            None,
            gr.update(visible=False),
            gr.update(visible=False),
        )

    try:
        # Debug logging
        print(f"\n{'='*60}")
        print(f"DEBUG: Processing PDF upload")
        print(f"DEBUG: pdf_file type: {type(pdf_file)}")
        print(f"DEBUG: pdf_file value: {pdf_file}")

        # Get filename and create slug
        # Note: With type="filepath", pdf_file is the filepath string
        if isinstance(pdf_file, str):
            pdf_filepath = pdf_file
            original_filename = Path(pdf_file).name
        else:
            # Fallback for other Gradio file types
            pdf_filepath = pdf_file.name
            original_filename = Path(pdf_file.name).name

        print(f"DEBUG: pdf_filepath: {pdf_filepath}")
        print(f"DEBUG: original_filename: {original_filename}")

        # Create slug from filename
        pdf_slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in Path(original_filename).stem).lower()
        print(f"DEBUG: pdf_slug: {pdf_slug}")

        # Check if already processed
        print(f"DEBUG: Checking if PDF already processed...")
        pdf_info = storage.get_pdf_metadata(pdf_slug)
        if pdf_info:
            print(f"DEBUG: PDF already processed, returning cached version")
            current_pdf_slug = pdf_slug

            status = f"""ℹ️ **PDF already processed!**

📄 **Document:** {pdf_info['filename']}
📊 **Statistics:**
- Pages: {pdf_info['num_pages']}
- Sections: {pdf_info['num_sections']}
- Text chunks: {pdf_info['num_chunks']}
- Processed: {pdf_info['processed_date'][:10]}

🤖 **Ready to answer questions!** Select it from the dropdown and ask below.
"""
            # Update dropdown
            pdf_list = storage.list_pdfs()
            choices = [(f"{p['filename']} ({p['num_chunks']} chunks)", p['slug']) for p in pdf_list]

            return (
                status,
                gr.update(choices=choices, value=pdf_slug),
                pdf_slug,
                gr.update(visible=True),
                gr.update(visible=True),
            )

        # Load from uploaded file first
        print(f"DEBUG: Loading PDF from: {pdf_filepath}")
        progress(0.1, desc="Loading PDF...")
        doc = load_pdf(pdf_filepath)

        # Save to data/input/ for persistence
        print(f"DEBUG: Saving PDF to data/input/")
        input_dir = Path("data/input")
        input_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = input_dir / original_filename
        if not pdf_path.exists():
            shutil.copy(pdf_filepath, pdf_path)
            print(f"DEBUG: Saved PDF to: {pdf_path}")
        else:
            print(f"DEBUG: PDF already exists at: {pdf_path}")

        current_pdf_slug = pdf_slug

        progress(0.2, desc="Extracting pages...")
        pages = extract_pages(doc)

        progress(0.3, desc="Extracting table of contents...")
        toc = extract_toc(doc)

        progress(0.35, desc="Building section tree...")
        sections = build_section_tree(toc)

        progress(0.4, desc="Chunking text...")
        chunks = build_chunks(pages, sections, max_tokens=max_tokens)

        progress(0.5, desc="Generating embeddings (this may take a minute)...")
        embeddings = embed_chunks(chunks, model_name=embed_model)

        progress(0.7, desc="Building search index...")
        storage.save_embeddings(pdf_slug, embeddings)

        progress(0.75, desc="Indexing figures and tables...")
        figures, tables = index_figures_tables(pages)

        progress(0.8, desc="Extracting cross-references...")
        all_mentions = []
        for chunk in chunks:
            all_mentions.extend(extract_mentions(chunk))

        all_mentions = resolve_mentions(
            all_mentions, sections, figures, tables, n_pages=len(pages)
        )

        progress(0.85, desc="Building knowledge graph...")
        graph = build_graph(
            doc_id="document",
            pages=pages,
            sections=sections,
            chunks=chunks,
            mentions=all_mentions,
            figures=figures,
            tables=tables,
        )

        progress(0.9, desc="Saving to database...")

        # Register PDF FIRST (required before saving metadata)
        print(f"DEBUG: Registering PDF in database...")
        storage.save_pdf_metadata(
            slug=pdf_slug,
            filename=original_filename,
            num_pages=len(pages),
            num_chunks=len(chunks),
            num_sections=len(sections),
            num_figures=len(figures),
            num_tables=len(tables),
            metadata={
                "embedding_model": embed_model,
                "embedding_dim": int(embeddings.shape[1]),
            },
        )

        # Save chunks
        print(f"DEBUG: Saving chunks to database...")
        chunks_data = [
            {"chunk_id": c.id, "section_id": c.section_id, "page": c.page, "text": c.text}
            for c in chunks
        ]
        storage.save_chunks(pdf_slug, chunks_data)

        # Save graph
        if hasattr(storage, 'save_graph'):
            print(f"DEBUG: Saving graph to database...")
            # Convert NetworkX graph to node/edge lists
            nodes = []
            for node_id, attrs in graph.nodes(data=True):
                node_doc = {"node_id": node_id, "type": attrs.get("type", "Unknown"), "label": attrs.get("label", "")}
                for k, v in attrs.items():
                    if k not in ["type", "label"] and v is not None:
                        node_doc[k] = v
                nodes.append(node_doc)

            edges = []
            for u, v, attrs in graph.edges(data=True):
                edge_doc = {"from_id": u, "to_id": v, "type": attrs.get("type", "EDGE")}
                for k, v in attrs.items():
                    if k not in ["type"] and v is not None:
                        edge_doc[k] = v
                edges.append(edge_doc)

            storage.save_graph(pdf_slug, nodes, edges)

        # Save metadata (AFTER PDF is registered)
        print(f"DEBUG: Saving metadata to database...")
        storage.save_metadata(pdf_slug, "sections", sections)
        storage.save_metadata(pdf_slug, "toc", toc)
        storage.save_metadata(pdf_slug, "mentions", [
            {
                "source_chunk_id": m.source_chunk_id,
                "kind": m.kind,
                "raw_text": m.raw_text,
                "target_hint": m.target_hint,
                "target_id": m.target_id,
            }
            for m in all_mentions
        ])
        storage.save_metadata(pdf_slug, "figures", figures)
        storage.save_metadata(pdf_slug, "tables", tables)

        progress(0.95, desc="Complete...")

        progress(1.0, desc="Complete!")

        status = f"""✅ **PDF processed successfully!**

📄 **Document:** {original_filename}
📊 **Statistics:**
- Pages: {len(pages)}
- Sections: {len(sections)}
- Text chunks: {len(chunks)}
- Figures: {len(figures)}
- Tables: {len(tables)}
- Cross-references found: {len(all_mentions)}

🤖 **Ready to answer questions!** Select it from the dropdown and ask below.
"""

        # Update dropdown with all PDFs
        pdf_list = storage.list_pdfs()
        choices = [(f"{p['filename']} ({p['num_chunks']} chunks)", p['slug']) for p in pdf_list]

        return (
            status,
            gr.update(choices=choices, value=pdf_slug),
            pdf_slug,
            gr.update(visible=True),
            gr.update(visible=True),
        )

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"\n{'='*60}")
        print(f"ERROR: Exception during PDF processing")
        print(f"ERROR: {error_details}")
        print(f"{'='*60}\n")
        return (
            f"❌ **Error processing PDF:** {str(e)}\n\nCheck terminal for full traceback.",
            gr.update(),
            None,
            gr.update(visible=False),
            gr.update(visible=False),
        )


def chat_response(
    message: str, history: list, selected_pdf: str, use_gemini: bool, top_k: int, embed_model: str
) -> Generator:
    """
    Generate response to user question.

    Args:
        message: User question
        history: Chat history
        selected_pdf: Selected PDF slug
        use_gemini: Whether to use Gemini for answers
        top_k: Number of chunks to retrieve
        embed_model: Embedding model name

    Yields:
        Updated chat history
    """
    print(f"\n{'='*60}")
    print(f"DEBUG Q&A: New question received")
    print(f"DEBUG Q&A: Question: {message}")
    print(f"DEBUG Q&A: Selected PDF: {selected_pdf}")
    print(f"DEBUG Q&A: Use Gemini: {use_gemini}")
    print(f"DEBUG Q&A: Top K: {top_k}")
    print(f"DEBUG Q&A: Embed Model: {embed_model}")

    if not selected_pdf:
        print(f"DEBUG Q&A: ERROR - No PDF selected")
        history.append((message, "⚠️ Please select a PDF from the dropdown first."))
        yield history
        return

    try:
        # Get answer using storage backend
        print(f"DEBUG Q&A: Calling answer_question()...")
        result = answer_question(
            message,
            selected_pdf,  # Pass slug instead of directory
            model_name=embed_model,
            top_k=top_k,
            use_gemini=use_gemini,
            storage=storage,
        )
        print(f"DEBUG Q&A: answer_question() completed successfully")

        # Format response
        answer = result["answer"]

        # Add source references if verbose
        sources = result["sources"][:3]  # Top 3 sources
        if sources and not use_gemini:  # Gemini mode already includes context
            source_refs = "\n\n📚 **Sources:**\n"
            for i, src in enumerate(sources, 1):
                source_refs += f"{i}. Section {src['section_id']}, Page {src['page']} (score: {src['similarity_score']:.2f})\n"
            answer += source_refs

        # Add related content
        related = result["related"]
        if any(related.values()):
            answer += "\n\n🔗 **Related:**"
            if related["figures"]:
                answer += f"\n- Figures: {', '.join(related['figures'][:3])}"
            if related["tables"]:
                answer += f"\n- Tables: {', '.join(related['tables'][:3])}"
            if related["sections"]:
                answer += f"\n- Sections: {', '.join(related['sections'][:3])}"

        print(f"DEBUG Q&A: Formatting and returning answer")
        history.append((message, answer))
        yield history

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"\n{'='*60}")
        print(f"ERROR Q&A: Exception during question answering")
        print(f"ERROR Q&A: {error_details}")
        print(f"{'='*60}\n")
        history.append((message, f"❌ **Error:** {str(e)}\n\nCheck terminal for full traceback."))
        yield history


def clear_chat():
    """Clear chat history."""
    return []


def load_available_pdfs():
    """Load available PDFs for dropdown."""
    pdf_list = storage.list_pdfs()
    if not pdf_list:
        return gr.update(choices=[], value=None)

    choices = [(f"{p['filename']} ({p['num_chunks']} chunks)", p['slug']) for p in pdf_list]
    return gr.update(choices=choices, value=pdf_list[0]['slug'] if pdf_list else None)


def refresh_pdf_list():
    """Refresh the PDF dropdown."""
    return load_available_pdfs()


# Build Gradio interface
with gr.Blocks(title="PDF Knowledge Graph Q&A", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 📚 PDF Knowledge Graph Q&A

        Upload a technical PDF manual and ask questions about it!

        **How it works:**
        1. Upload a PDF document
        2. Wait for processing (builds knowledge graph with embeddings)
        3. Ask questions in natural language
        4. Get answers with source references
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## ⚙️ Configuration")

            gr.Markdown("### 📤 Upload New PDF")

            pdf_input = gr.File(
                label="Upload PDF",
                file_types=[".pdf"],
                type="filepath"
            )

            embed_model = gr.Dropdown(
                choices=[
                    "sentence-transformers/all-MiniLM-L6-v2",
                    "sentence-transformers/all-mpnet-base-v2",
                    "BAAI/bge-small-en-v1.5",
                    "BAAI/bge-base-en-v1.5",
                ],
                value="sentence-transformers/all-MiniLM-L6-v2",
                label="Embedding Model",
                info="Model for semantic search"
            )

            max_tokens = gr.Slider(
                minimum=100,
                maximum=1000,
                value=500,
                step=50,
                label="Max Tokens per Chunk",
                info="Smaller = more precise, Larger = more context"
            )

            process_btn = gr.Button("🚀 Process PDF", variant="primary", size="lg")

            status_output = gr.Markdown(label="Status")

            gr.Markdown("### 📚 Select PDF to Query")

            with gr.Row():
                pdf_selector = gr.Dropdown(
                    label="Select PDF",
                    choices=[],
                    value=None,
                    interactive=True,
                    scale=3,
                )
                refresh_btn = gr.Button("🔄", scale=1, size="sm")

            gr.Markdown("### 💬 Chat Options")

            use_gemini = gr.Checkbox(
                label="Use Gemini for Natural Language Answers",
                value=True if os.getenv("GEMINI_API_KEY") else False,
                info="Requires GEMINI_API_KEY in .env"
            )

            top_k = gr.Slider(
                minimum=1,
                maximum=10,
                value=5,
                step=1,
                label="Number of Sources to Retrieve",
                info="More sources = more context but slower"
            )

        with gr.Column(scale=2):
            gr.Markdown("## 💬 Ask Questions")

            chatbot = gr.Chatbot(
                height=500,
                visible=False,
                label="Q&A Assistant",
                show_copy_button=True,
                type='tuples'
            )

            with gr.Row(visible=False) as chat_input_row:
                msg = gr.Textbox(
                    label="Your Question",
                    placeholder="e.g., What are the operative temperature limits?",
                    scale=4
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1)

            clear_btn = gr.Button("🗑️ Clear Chat History", visible=False)

            gr.Markdown(
                """
                ### 💡 Example Questions
                - What are the operative temperature limits?
                - How do you mount this device?
                - What are the different models available?
                - What certifications does this product have?
                - What is the input voltage range?
                - Where can I find the wiring diagram?
                """,
                visible=True
            )

    # Event handlers
    process_btn.click(
        fn=process_pdf,
        inputs=[pdf_input, embed_model, max_tokens],
        outputs=[status_output, pdf_selector, pdf_selector, chatbot, chat_input_row]
    )

    refresh_btn.click(
        fn=refresh_pdf_list,
        outputs=pdf_selector
    )

    submit_btn.click(
        fn=chat_response,
        inputs=[msg, chatbot, pdf_selector, use_gemini, top_k, embed_model],
        outputs=chatbot
    ).then(
        lambda: "",
        outputs=msg
    )

    msg.submit(
        fn=chat_response,
        inputs=[msg, chatbot, pdf_selector, use_gemini, top_k, embed_model],
        outputs=chatbot
    ).then(
        lambda: "",
        outputs=msg
    )

    clear_btn.click(fn=clear_chat, outputs=chatbot)

    # Show clear button when chatbot is visible
    chatbot.change(
        lambda: gr.update(visible=True),
        outputs=clear_btn
    )

    # Load available PDFs on startup
    demo.load(
        fn=load_available_pdfs,
        outputs=pdf_selector
    )


if __name__ == "__main__":
    print("=" * 80)
    print("PDF Knowledge Graph Q&A - Gradio Web App")
    print("=" * 80)

    # Check Gemini
    if os.getenv("GEMINI_API_KEY"):
        gemini_model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        print(f"✅ Gemini enabled: {gemini_model}")
    else:
        print("⚠️  Gemini disabled: GEMINI_API_KEY not set in .env")

    print("=" * 80)
    print("\n🚀 Starting web server...\n")

    # Launch with share=True for public link
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=8016,
        show_error=True
    )
