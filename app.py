#!/usr/bin/env python3
"""
Gradio web application for PDF Knowledge Graph Q&A.

Usage:
    python app.py
"""

# Fix for macOS multiprocessing segmentation fault
import multiprocessing
import sys
if sys.platform == "darwin":  # macOS
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

import os

# Fix FAISS threading issues on macOS
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import shutil
from pathlib import Path
from typing import Generator

import gradio as gr
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import pipeline modules
from pdfkg.query import answer_question
from pdfkg.storage import get_storage_backend
from pdfkg.pdf_manager import ingest_pdf, auto_ingest_directory


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


def process_pdf(pdf_files, embed_model: str, max_tokens: int, use_gemini: bool, force_reprocess: bool, progress=gr.Progress()) -> tuple:
    """
    Process uploaded PDF(s) and build knowledge graph.

    Args:
        pdf_files: Uploaded PDF file(s) - can be single file or list of files
        embed_model: Embedding model name
        max_tokens: Max tokens per chunk
        use_gemini: Enable Gemini visual analysis
        force_reprocess: Force reprocessing even if PDF is cached
        progress: Gradio progress tracker

    Returns:
        Tuple of (status_message, pdf_dropdown_choices, pdf_dropdown_value, chatbot_visibility, chat_input_visibility)
    """
    global current_pdf_slug

    if pdf_files is None or (isinstance(pdf_files, list) and len(pdf_files) == 0):
        return (
            "Please upload at least one PDF file.",
            gr.update(),
            None,
            gr.update(visible=False),
            gr.update(visible=False),
        )

    # Normalize to list
    if not isinstance(pdf_files, list):
        pdf_files = [pdf_files]

    print(f"\n{'='*60}")
    print(f"DEBUG: Processing {len(pdf_files)} PDF(s)")
    print(f"DEBUG: Embed model: {embed_model}")
    print(f"DEBUG: Max tokens: {max_tokens}")
    print(f"DEBUG: Use Gemini: {use_gemini}")
    print(f"DEBUG: Force reprocess: {force_reprocess}")
    print(f"{'='*60}\n")

    processed_results = []
    cached_results = []
    failed_results = []
    last_successful_slug = None

    # Process each PDF
    for idx, pdf_file in enumerate(pdf_files, 1):
        try:
            # Get filepath
            if isinstance(pdf_file, str):
                pdf_filepath = pdf_file
            else:
                pdf_filepath = pdf_file.name

            pdf_name = Path(pdf_filepath).name

            # Update progress
            overall_progress = (idx - 1) / len(pdf_files)
            progress(overall_progress, desc=f"Processing {idx}/{len(pdf_files)}: {pdf_name}")

            print(f"\n[{idx}/{len(pdf_files)}] Processing: {pdf_name}")

            # Progress callback for individual PDF
            def pdf_progress(pct: float, desc: str):
                # Combine overall progress with PDF progress
                pdf_weight = 1.0 / len(pdf_files)
                combined_progress = overall_progress + (pct * pdf_weight)
                progress(combined_progress, desc=f"[{idx}/{len(pdf_files)}] {pdf_name}: {desc}")

            # Run unified ingestion pipeline
            result = ingest_pdf(
                pdf_path=Path(pdf_filepath),
                storage=storage,
                embed_model=embed_model,
                max_tokens=max_tokens,
                use_gemini=use_gemini,
                gemini_pages="",  # Process all pages if Gemini enabled
                force_reprocess=force_reprocess,
                save_to_db=True,
                save_files=True,
                output_dir=None,  # Use default
                progress_callback=pdf_progress,
            )

            last_successful_slug = result.pdf_slug
            summary = result.summary()

            if result.was_cached:
                cached_results.append(summary)
                print(f"  ✓ Cached: {pdf_name}")
            else:
                processed_results.append(summary)
                print(f"  ✓ Processed: {pdf_name}")

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"\n{'='*60}")
            print(f"ERROR: Exception processing {pdf_name}")
            print(f"ERROR: {error_details}")
            print(f"{'='*60}\n")

            failed_results.append({
                'filename': pdf_name,
                'error': str(e)
            })

    # Build status message
    total = len(processed_results) + len(cached_results) + len(failed_results)

    if total == 0:
        status = "❌ No PDFs were processed."
    else:
        status_parts = []

        # Header
        if len(pdf_files) == 1:
            if processed_results:
                status_parts.append("✅ **PDF processed successfully!**\n")
            elif cached_results:
                status_parts.append("ℹ️ **PDF already processed!** (loaded from cache)\n")
            else:
                status_parts.append("❌ **PDF processing failed!**\n")
        else:
            status_parts.append(f"📦 **Batch Processing Complete: {total} PDF(s)**\n")

        # Summary stats for batch
        if len(pdf_files) > 1:
            status_parts.append(f"📊 **Summary:**")
            status_parts.append(f"- ✅ Newly processed: {len(processed_results)}")
            status_parts.append(f"- ⊘ Cached (skipped): {len(cached_results)}")
            status_parts.append(f"- ❌ Failed: {len(failed_results)}\n")

        # Processed PDFs
        if processed_results:
            status_parts.append("**✅ Newly Processed:**")
            for s in processed_results:
                status_parts.append(f"\n📄 **{s['filename']}**")
                status_parts.append(f"- Pages: {s['num_pages']}, Chunks: {s['num_chunks']}, Sections: {s['num_sections']}")
                status_parts.append(f"- Figures: {s['num_figures']}, Tables: {s['num_tables']}")
                status_parts.append(f"- Cross-refs: {s['num_mentions']} ({s['num_resolved_mentions']} resolved)")
            status_parts.append("")

        # Cached PDFs
        if cached_results:
            status_parts.append("**⊘ Cached (Already Processed):**")
            for s in cached_results:
                status_parts.append(f"- {s['filename']} ({s['num_chunks']} chunks)")
            status_parts.append("")

        # Failed PDFs
        if failed_results:
            status_parts.append("**❌ Failed:**")
            for f in failed_results:
                status_parts.append(f"- {f['filename']}: {f['error']}")
            status_parts.append("")

        if processed_results or cached_results:
            status_parts.append("🤖 **Ready to answer questions!** Select a PDF from the dropdown and ask below.")

        status = "\n".join(status_parts)

    # Update dropdown
    pdf_list = storage.list_pdfs()
    choices = [(f"{p['filename']} ({p['num_chunks']} chunks)", p['slug']) for p in pdf_list]

    # Set current slug to last successful one
    if last_successful_slug:
        current_pdf_slug = last_successful_slug

    # Show chat UI if at least one PDF was processed successfully
    show_chat = len(processed_results) + len(cached_results) > 0

    return (
        status,
        gr.update(choices=choices, value=last_successful_slug if last_successful_slug else None),
        last_successful_slug,
        gr.update(visible=show_chat),
        gr.update(visible=show_chat),
    )


def chat_response(
    message: str, history: list, selected_pdf: str, llm_provider: str, top_k: int, embed_model: str
) -> Generator:
    """
    Generate response to user question.

    Args:
        message: User question
        history: Chat history
        selected_pdf: Selected PDF slug
        llm_provider: LLM provider ("none", "gemini", or "mistral")
        top_k: Number of chunks to retrieve
        embed_model: Embedding model name

    Yields:
        Updated chat history
    """
    print(f"\n{'='*60}")
    print(f"DEBUG Q&A: New question received")
    print(f"DEBUG Q&A: Question: {message}")
    print(f"DEBUG Q&A: Selected PDF: {selected_pdf}")
    print(f"DEBUG Q&A: LLM Provider: {llm_provider}")
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
            llm_provider=llm_provider,
            storage=storage,
        )
        print(f"DEBUG Q&A: answer_question() completed successfully")

        # Format response
        answer = result["answer"]

        # Add source references if verbose
        sources = result["sources"][:3]  # Top 3 sources
        if sources and llm_provider == "none":  # LLM modes already include context
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


def auto_ingest_on_startup():
    """Auto-ingest all PDFs from data/input/ on app startup."""
    print("\n" + "=" * 80)
    print("🔄 AUTO-INGEST: Checking data/input/ for new PDFs...")
    print("=" * 80)

    # Determine if Gemini should be used by default
    use_gemini_default = bool(os.getenv("GEMINI_API_KEY"))

    # Progress callback
    def progress_print(pdf_name: str, status: str):
        print(f"  [{pdf_name}] {status}")

    try:
        results = auto_ingest_directory(
            input_dir=Path("data/input"),
            storage=storage,
            embed_model="sentence-transformers/all-MiniLM-L6-v2",
            max_tokens=500,
            use_gemini=use_gemini_default,
            save_to_db=True,
            save_files=True,
            progress_callback=progress_print,
        )

        total = len(results['processed']) + len(results['skipped']) + len(results['failed'])

        if total == 0:
            print("\n⚠️  No PDFs found in data/input/")
        else:
            print(f"\n✓ Auto-ingest complete:")
            print(f"  - Newly processed: {len(results['processed'])}")
            print(f"  - Skipped (cached): {len(results['skipped'])}")
            print(f"  - Failed: {len(results['failed'])}")

        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n⚠️  Auto-ingest error: {e}")
        print("=" * 80 + "\n")

    # Return updated dropdown state
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
                label="Upload PDF(s)",
                file_types=[".pdf"],
                file_count="multiple",
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

            use_gemini_ingest = gr.Checkbox(
                label="Use Gemini for Visual Analysis During Ingestion",
                value=True if os.getenv("GEMINI_API_KEY") else False,
                info="Extract cross-references from diagrams and images (requires GEMINI_API_KEY)",
                interactive=True
            )

            force_reprocess = gr.Checkbox(
                label="Force Reprocess",
                value=False,
                info="Reprocess PDF even if already cached in database",
                interactive=True
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

            # Determine default LLM provider
            default_llm = "none"
            if os.getenv("GEMINI_API_KEY"):
                default_llm = "gemini"
            elif os.getenv("MISTRAL_API_KEY"):
                default_llm = "mistral"

            llm_provider = gr.Radio(
                choices=[
                    ("No LLM (keyword search only)", "none"),
                    ("Gemini", "gemini"),
                    ("Mistral", "mistral"),
                ],
                value=default_llm,
                label="LLM Provider for Answer Generation",
                info="Choose the AI model to generate natural language answers"
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
        inputs=[pdf_input, embed_model, max_tokens, use_gemini_ingest, force_reprocess],
        outputs=[status_output, pdf_selector, pdf_selector, chatbot, chat_input_row]
    )

    refresh_btn.click(
        fn=refresh_pdf_list,
        outputs=pdf_selector
    )

    submit_btn.click(
        fn=chat_response,
        inputs=[msg, chatbot, pdf_selector, llm_provider, top_k, embed_model],
        outputs=chatbot
    ).then(
        lambda: "",
        outputs=msg
    )

    msg.submit(
        fn=chat_response,
        inputs=[msg, chatbot, pdf_selector, llm_provider, top_k, embed_model],
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

    # Auto-ingest PDFs from data/input/ and load dropdown on startup
    demo.load(
        fn=auto_ingest_on_startup,
        outputs=pdf_selector
    )


if __name__ == "__main__":
    print("=" * 80)
    print("PDF Knowledge Graph Q&A - Gradio Web App")
    print("=" * 80)

    # Check LLM providers
    llm_status = []
    if os.getenv("GEMINI_API_KEY"):
        gemini_model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        llm_status.append(f"✅ Gemini enabled: {gemini_model}")
    else:
        llm_status.append("⚠️  Gemini disabled: GEMINI_API_KEY not set in .env")

    if os.getenv("MISTRAL_API_KEY"):
        mistral_model = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
        llm_status.append(f"✅ Mistral enabled: {mistral_model}")
    else:
        llm_status.append("⚠️  Mistral disabled: MISTRAL_API_KEY not set in .env")

    for status in llm_status:
        print(status)

    print("=" * 80)
    print("\n🚀 Starting web server...\n")

    # Launch with share=True for public link
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=8016,
        show_error=True
    )
