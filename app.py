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

import json
import os

# Fix FAISS threading issues on macOS
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import shutil
from pathlib import Path
from typing import Any, Dict, Generator, List

import gradio as gr
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import pipeline modules
from pdfkg.query import answer_question
from pdfkg.storage import get_storage_backend
from pdfkg.pdf_manager import ingest_pdf, auto_ingest_directory
from pdfkg.cross_document import CrossDocumentAnalyzer, cross_doc_ref_to_dict, semantic_link_to_dict, version_relation_to_dict
from pdfkg.ner import TechnicalNER, extract_entities_from_chunks, entity_to_dict
from pdfkg.aas_classifier import classify_pdfs_to_aas
from pdfkg.aas_extractor import extract_aas_data
from pdfkg.aas_validator import validate_aas_data
from pdfkg.aas_xml_generator import generate_aas_xml
from pdfkg.template_extractor import extract_submodels, available_submodels
from pdfkg.submodel_templates import get_template
from pdfkg import llm_stats


# System logs buffer (thread-safe)
import threading
import io
from datetime import datetime

class SystemLogger:
    """Thread-safe system logger for UI display."""
    def __init__(self, max_lines=1000):
        self.logs = []
        self.max_lines = max_lines
        self.lock = threading.Lock()

    def log(self, message, level="INFO"):
        """Add a log entry."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"

        with self.lock:
            self.logs.append(log_entry)
            if len(self.logs) > self.max_lines:
                self.logs.pop(0)

        # Also print to console
        print(log_entry)

    def get_logs(self):
        """Get all logs as formatted string."""
        with self.lock:
            return "\n".join(self.logs)

    def clear(self):
        """Clear all logs."""
        with self.lock:
            self.logs = []

# Initialize global logger
system_logger = SystemLogger()

def verify_services():
    """
    Verify that Milvus and ArangoDB are running and accessible.
    Returns (success: bool, error_message: str)
    """
    errors = []

    # Check ArangoDB
    try:
        from arango import ArangoClient
        arango_host = os.getenv("ARANGO_HOST", "localhost")
        arango_port = os.getenv("ARANGO_PORT", "8529")
        arango_user = os.getenv("ARANGO_USER", "root")
        arango_password = os.getenv("ARANGO_PASSWORD", "")

        client = ArangoClient(hosts=f"http://{arango_host}:{arango_port}")
        sys_db = client.db('_system', username=arango_user, password=arango_password)
        version = sys_db.version()
        system_logger.log(f"✅ ArangoDB connected: version {version}", "INFO")
    except Exception as e:
        error_msg = f"❌ ArangoDB connection failed: {e}"
        system_logger.log(error_msg, "ERROR")
        errors.append(error_msg)

    # Check Milvus
    try:
        from pymilvus import connections, utility
        milvus_host = os.getenv("MILVUS_HOST", "localhost")
        milvus_port = os.getenv("MILVUS_PORT", "19530")

        connections.connect(
            alias="verify",
            host=milvus_host,
            port=milvus_port
        )

        # Try to list collections as a connection test
        collections = utility.list_collections(using="verify")
        connections.disconnect("verify")
        system_logger.log(f"✅ Milvus connected: {len(collections)} collections found", "INFO")
    except Exception as e:
        error_msg = f"❌ Milvus connection failed: {e}"
        system_logger.log(error_msg, "ERROR")
        errors.append(error_msg)

    if errors:
        return False, "\n".join(errors)

    return True, ""

# Verify services on startup
system_logger.log("=" * 80, "INFO")
system_logger.log("PDFKG Web Application Starting...", "INFO")
system_logger.log("=" * 80, "INFO")
system_logger.log("Verifying database services...", "INFO")

services_ok, error_message = verify_services()

if not services_ok:
    system_logger.log("=" * 80, "ERROR")
    system_logger.log("CRITICAL: Required services are not available!", "ERROR")
    system_logger.log(error_message, "ERROR")
    system_logger.log("=" * 80, "ERROR")
    system_logger.log("Please ensure ArangoDB and Milvus are running:", "ERROR")
    system_logger.log("  docker-compose up -d", "ERROR")
    system_logger.log("=" * 80, "ERROR")
    print("\n" + system_logger.get_logs())
    sys.exit(1)

# Initialize storage backend (with error handling)
try:
    storage = get_storage_backend()
    storage_type = os.getenv("STORAGE_BACKEND", "arango")
    system_logger.log(f"✅ Storage backend initialized: {storage_type}", "INFO")
except Exception as e:
    system_logger.log(f"⚠️  Storage backend initialization failed: {e}", "ERROR")
    system_logger.log(f"⚠️  Using file storage as fallback", "WARNING")
    from pdfkg.storage import FileStorage
    storage = FileStorage()
    storage_type = "file"

# Global state
current_pdf_slug = None

AVAILABLE_SUBMODELS = available_submodels()
SUBMODEL_CHOICES = [(get_template(key).display_name, key) for key in AVAILABLE_SUBMODELS]


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
                # Display AAS classification results
                if s.get('aas_classification'):
                    submodels = s['aas_classification'].get('submodels', [])
                    if submodels:
                        scores = s['aas_classification'].get('confidence_scores', {})
                        tags = [f"{sm} ({scores.get(sm, 0):.2f})" for sm in submodels]
                        status_parts.append(f"- **Submodels**: {', '.join(tags)}")
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


def reset_project_data(progress=gr.Progress(track_tqdm=False)) -> tuple:
    """Wipe project artifacts and rebuild empty ArangoDB/Milvus state."""
    global current_pdf_slug

    status_lines = ["### ♻️ Project Reset", ""]

    # 1. Clear local output directory
    progress(0.1, desc="Clearing data/output/")
    output_dir = Path("data/output")
    try:
        if output_dir.exists():
            for item in output_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            status_lines.append("- ✅ Cleared `data/output/`")
            system_logger.log("Cleared data/output/", "INFO")
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            status_lines.append("- ℹ️ Created empty `data/output/`")
            system_logger.log("Created data/output/ directory", "INFO")
    except Exception as exc:
        error_msg = f"Failed to clear data/output/: {exc}"
        status_lines.append(f"- ❌ {error_msg}")
        system_logger.log(error_msg, "ERROR")

    # 2. Reset ArangoDB database
    progress(0.4, desc="Resetting ArangoDB")
    arango_reset_ok = False
    if hasattr(storage, "db_client"):
        try:
            storage.db_client.reset_database()
            arango_reset_ok = True
            status_lines.append(f"- ✅ Reset ArangoDB database `{storage.db_client.db_name}`")
            system_logger.log(f"Reset ArangoDB database {storage.db_client.db_name}", "INFO")
        except Exception as exc:
            error_msg = f"Failed to reset ArangoDB: {exc}"
            status_lines.append(f"- ❌ {error_msg}")
            system_logger.log(error_msg, "ERROR")
    else:
        status_lines.append("- ℹ️ No ArangoDB backend detected; skipped")
        system_logger.log("Reset skipped: storage backend lacks ArangoDB client", "WARNING")

    # 3. Reset Milvus collection
    progress(0.7, desc="Resetting Milvus")
    if hasattr(storage, "milvus_client") and storage.milvus_client:
        try:
            default_dim = os.getenv("DEFAULT_EMBED_DIM")
            dimension = int(default_dim) if default_dim else None
            storage.milvus_client.reset_collection(dimension=dimension)
            status_lines.append(f"- ✅ Reset Milvus collection `{storage.milvus_client.collection_name}`")
            system_logger.log(f"Reset Milvus collection {storage.milvus_client.collection_name}", "INFO")
        except Exception as exc:
            error_msg = f"Failed to reset Milvus: {exc}"
            status_lines.append(f"- ❌ {error_msg}")
            system_logger.log(error_msg, "ERROR")
    else:
        status_lines.append("- ℹ️ No Milvus client configured; skipped")
        system_logger.log("Reset skipped: Milvus client unavailable", "WARNING")

    # 4. Re-ingest PDFs from data/input
    progress(0.85, desc="Reprocessing data/input/")
    ingest_processed = ingest_cached = ingest_failed = 0
    ingest_errors = []
    try:
        # Ensure fresh connections
        if hasattr(storage, "db_client"):
            storage.db_client.connect()
        if hasattr(storage, "milvus_client") and storage.milvus_client:
            storage.milvus_client.connect()

        use_gemini_default = bool(os.getenv("GEMINI_API_KEY"))

        ingest_events: list[str] = []

        def log_progress(pdf_name: str, status: str):
            message = f"[RESET INGEST] {pdf_name}: {status}"
            ingest_events.append(message)
            system_logger.log(message, "INFO")

        ingest_results = auto_ingest_directory(
            input_dir=Path("data/input"),
            storage=storage,
            embed_model="sentence-transformers/all-MiniLM-L6-v2",
            max_tokens=500,
            use_gemini=use_gemini_default,
            save_to_db=True,
            save_files=True,
            progress_callback=log_progress,
            force_reprocess=True,
        )

        ingest_processed = len(ingest_results.get('processed', []))
        ingest_cached = len(ingest_results.get('skipped', []))
        ingest_failed = len(ingest_results.get('failed', []))

        status_lines.append("- ✅ Reprocessed PDFs from `data/input/`")
        status_lines.append(f"  • Processed: {ingest_processed}")
        status_lines.append(f"  • Cached: {ingest_cached}")
        status_lines.append(f"  • Failed: {ingest_failed}")

        if ingest_events:
            status_lines.extend(["", "#### Detailed progress:"] + ingest_events)

        system_logger.log("Reprocessed data/input/ after reset", "INFO")
    except Exception as exc:
        ingest_errors.append(str(exc))
        status_lines.append(f"- ❌ Failed to reprocess data/input/: {exc}")
        system_logger.log(f"Failed to reprocess data/input/: {exc}", "ERROR")

    progress(1.0, desc="Reset complete")

    pdf_list = storage.list_pdfs()

    # Build cross-document relationships if multiple PDFs present
    if len(pdf_list) >= 2:
        try:
            build_cross_document_relationships(
                storage=storage,
                similarity_threshold=0.85,
                top_k=10
            )
            system_logger.log("Rebuilt cross-document relationships after reset", "INFO")
        except Exception as exc:
            system_logger.log(f"Failed to rebuild cross-document relationships: {exc}", "ERROR")
    selected_slug = pdf_list[0]['slug'] if pdf_list else None
    current_pdf_slug = selected_slug

    dropdown_update = load_available_pdfs()

    has_pdfs = bool(pdf_list)
    chatbot_update = gr.update(value=[], visible=has_pdfs)
    chat_row_update = gr.update(visible=has_pdfs)

    status_message = "\n".join(status_lines)

    return (
        status_message,
        dropdown_update,
        selected_slug,
        chatbot_update,
        chat_row_update,
    )


SUBMODEL_EDITOR_ORDER = AVAILABLE_SUBMODELS


def format_metadata_display(metadata: Dict[str, Dict[str, Any]]) -> str:
    if not metadata:
        return 'No evidence captured yet.'

    lines = ['**Confidence & Evidence**']
    for path_key in sorted(metadata.keys()):
        info = metadata.get(path_key, {})
        confidence = info.get('confidence')
        if isinstance(confidence, (int, float)):
            confidence_str = f"{confidence:.2f}"
        else:
            confidence_str = 'N/A'
        lines.append(f"- `{path_key}` (confidence {confidence_str})")
        sources = info.get('sources') or []
        for source in sources[:3]:
            lines.append(f"    • {source}")
        notes = info.get('notes')
        if notes:
            lines.append(f"    • Notes: {notes}")
    return '\n'.join(lines)


def extract_submodels_handler(selected_submodels, llm_provider, state, progress=gr.Progress(track_tqdm=True)):
    state = dict(state or {})
    selected_submodels = selected_submodels or []

    editor_updates: List[Any] = []
    metadata_updates: List[Any] = []
    accordion_updates: List[Any] = []
    status_lines: List[str] = []

    if not selected_submodels:
        status_lines.append('Select at least one submodel to extract.')
        for key in SUBMODEL_EDITOR_ORDER:
            editor_updates.append(gr.update(visible=False))
            metadata_updates.append(gr.update(visible=False))
            accordion_updates.append(gr.update(visible=False))
        download_update = gr.update(value=None, visible=False)
        return '\n'.join(status_lines), state, download_update, *editor_updates, *metadata_updates, *accordion_updates

    try:
        system_logger.log(f"Extracting submodels: {', '.join(selected_submodels)}", 'INFO')

        def progress_callback(pct, desc):
            progress(pct, desc=desc)

        extracted = extract_submodels(
            storage=storage,
            submodels=selected_submodels,
            llm_provider=llm_provider,
            progress_callback=progress_callback
        )
        status_lines.append('Extraction complete.')
        for key in selected_submodels:
            template = get_template(key)
            result = extracted.get(key)
            if result:
                data = result.get('data', template.schema)
                metadata = result.get('metadata', {})
            else:
                data = template.schema
                metadata = {}
            json_text = json.dumps(data, indent=2)
            state[key] = {"json": json_text, "metadata": metadata}
            status_lines.append(f"- {template.display_name}: captured {len(json_text)} characters")
            metadata_summary = format_metadata_display(metadata)
            status_lines.append(metadata_summary)
    except Exception as exc:
        system_logger.log(f"Extraction failed: {exc}", 'ERROR')
        status_lines.append(f"Extraction failed: {exc}")
        for key in SUBMODEL_EDITOR_ORDER:
            is_selected = key in selected_submodels
            editor_updates.append(gr.update(visible=is_selected))
            metadata_updates.append(gr.update(visible=is_selected))
            accordion_updates.append(gr.update(visible=is_selected))
        download_update = gr.update(value=None, visible=False)
        return '\n'.join(status_lines), state, download_update, *editor_updates, *metadata_updates, *accordion_updates

    for key in SUBMODEL_EDITOR_ORDER:
        entry = state.get(key)
        if isinstance(entry, str):
            entry = {"json": entry, "metadata": {}}
            state[key] = entry
        if key in selected_submodels:
            template = get_template(key)
            if entry is None:
                entry = {"json": template.default_text, "metadata": {}}
                state[key] = entry
            editor_updates.append(gr.update(value=entry["json"], visible=True))
            metadata_updates.append(gr.update(value=format_metadata_display(entry.get("metadata", {})), visible=True))
            accordion_updates.append(gr.update(visible=True, open=False))
        else:
            editor_updates.append(gr.update(visible=False))
            metadata_updates.append(gr.update(visible=False))
            accordion_updates.append(gr.update(visible=False))

    summary_lines = list(llm_stats.summary_lines())
    for line in summary_lines:
        system_logger.log(line, 'INFO')
    status_lines.extend(summary_lines)

    download_update = gr.update(value=None, visible=False)
    return '\n'.join(status_lines), state, download_update, *editor_updates, *metadata_updates, *accordion_updates



def update_submodel_editor(value, state, submodel_key):
    state = dict(state or {})
    entry = dict(state.get(submodel_key, {}))
    entry['json'] = value
    entry.setdefault('metadata', {})
    state[submodel_key] = entry
    return state


def generate_selected_aasx(llm_provider, selected_submodels, state, progress=gr.Progress(track_tqdm=True)):
    state = dict(state or {})
    selected_submodels = selected_submodels or []

    if not selected_submodels:
        message = 'Select at least one submodel before generating the AASX.'
        return message, gr.update(value=None, visible=False)

    compiled: Dict[str, Any] = {}
    metadata_map: Dict[str, Dict[str, Any]] = {}
    for key in selected_submodels:
        entry = state.get(key)
        if isinstance(entry, str):
            entry = {"json": entry, "metadata": {}}
            state[key] = entry
        if not entry:
            system_logger.log(f'No data available for submodel {key}.', 'WARNING')
            continue
        raw_json = entry.get('json')
        if not raw_json:
            system_logger.log(f'No JSON content for submodel {key}.', 'WARNING')
            continue
        try:
            compiled[key] = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            message = f"Invalid JSON for submodel {get_template(key).display_name}: {exc}"
            system_logger.log(message, 'ERROR')
            return message, gr.update(value=None, visible=False)
        metadata_map[key] = entry.get('metadata', {})

    if not compiled:
        message = 'No submodel data available. Extract or edit templates first.'
        return message, gr.update(value=None, visible=False)

    progress(0.5, desc='Generating AAS XML...')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path('data/output')
    output_dir.mkdir(parents=True, exist_ok=True)
    xml_path = output_dir / f'aas_output_{timestamp}.xml'

    xml_output = generate_aas_xml(
        storage=storage,
        llm_provider=llm_provider,
        output_path=xml_path,
        data=compiled
    )

    if not xml_output:
        message = 'XML generation returned no output. Check logs for details.'
        system_logger.log(message, 'ERROR')
        return message, gr.update(value=None, visible=False)

    progress(1.0, desc='Complete')
    system_logger.log(f'Generated XML at {xml_path}', 'INFO')
    summary_lines = list(llm_stats.summary_lines())
    for line in summary_lines:
        system_logger.log(line, 'INFO')

    status_lines = [
        'AAS XML generated successfully.',
        f"Submodels: {', '.join(compiled.keys())}",
        f'Saved to: {xml_path}'
    ]

    for key in compiled.keys():
        template = get_template(key)
        status_lines.append(f"\n### {template.display_name}")
        status_lines.append(format_metadata_display(metadata_map.get(key, {})))

    status_lines.extend(summary_lines)

    return '\n'.join(status_lines), gr.update(value=str(xml_path), visible=True)



def chat_response(
    message: str, history: list, selected_pdf: str, llm_provider: str, top_k: int, embed_model: str
) -> Generator:
    """
    Generate response to user question using the Query Planner and Orchestrator.
    """
    if not selected_pdf:
        history.append((message, "⚠️ Please select a PDF from the dropdown first."))
        yield history
        return

    try:
        # Get answer using the new orchestrated query pipeline
        result = answer_question(
            question=message,
            pdf_slug=selected_pdf,
            model_name=embed_model,
            top_k=top_k,
            llm_provider=llm_provider,
            storage=storage,
        )

        # Format the response
        answer = result.get("answer", "No answer could be generated.")
        sources = result.get("sources", [])

        # Add source references
        if sources:
            source_refs = "\n\n📚 **Sources:**\n"
            for i, src in enumerate(sources[:5], 1): # Show top 5 sources
                # Handle different source types (chunk vs. graph node)
                if 'text' in src: # It's a chunk
                    page = src.get('page', 'N/A')
                    section = src.get('section_id', 'N/A')
                    score = src.get('similarity_score')
                    score_str = f"(score: {score:.2f})" if score else ""
                    source_refs += f"{i}. Section {section}, Page {page} {score_str}\n"
                else: # It's a graph node
                    node_type = src.get('type', 'Node')
                    label = src.get('label', src.get('_key', 'N/A'))
                    source_refs += f"{i}. {node_type}: {label}\n"
            answer += source_refs

        # Optionally, add the debug plan to the output
        if "debug" in result and result["debug"].get("plan"):
            plan_json = json.dumps(result["debug"]["plan"], indent=2)
            answer += f"\n\n🔍 **Query Plan:**\n```json\n{plan_json}\n```"

        history.append((message, answer))
        yield history

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        system_logger.log(f"Q&A Error: {error_details}", "ERROR")
        history.append((message, f"❌ **Error:** {str(e)}\n\nCheck system logs for full traceback."))
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


def build_cross_document_relationships(storage, similarity_threshold: float = 0.85, top_k: int = 10):
    """
    Build cross-document relationships for all PDFs in the database.

    Executes 3 phases:
    - Phase 1: Cross-doc refs, Entities, Versioning
    - Phase 2: Semantic similarity, Topic clustering
    - Phase 3: Citation network

    Args:
        storage: Storage backend
        similarity_threshold: Minimum similarity for semantic links
        top_k: Number of similar chunks to find
    """
    print("\n" + "=" * 80)
    print("BUILDING CROSS-DOCUMENT RELATIONSHIPS")
    print("=" * 80)

    # Get all PDFs
    all_pdfs = storage.list_pdfs()

    if len(all_pdfs) < 2:
        print(f"\n⊘ Skipping relationship building: Need at least 2 PDFs (found {len(all_pdfs)})")
        return

    print(f"\n📚 Found {len(all_pdfs)} PDFs in database")

    # Get Milvus client if available
    milvus_client = None
    if hasattr(storage, 'milvus_client') and storage.milvus_client:
        milvus_client = storage.milvus_client
        print("✓ Milvus client available for semantic search")
    else:
        print("⚠️  Milvus not available - will skip semantic similarity and topic clustering")

    # Initialize analyzer
    analyzer = CrossDocumentAnalyzer(storage, milvus_client)

    # === PHASE 1: MVP ===
    print("\n" + "-" * 80)
    print("Phase 1: Cross-doc refs, Entities, Versioning")
    print("-" * 80)

    # 1.1: Cross-document references
    print("\n📎 Step 1.1: Extracting cross-document references...")
    all_refs = []
    for pdf in all_pdfs:
        slug = pdf['slug']
        refs = analyzer.extract_cross_doc_refs(slug)
        resolved_refs = analyzer.resolve_cross_doc_refs(refs)
        all_refs.extend(resolved_refs)
        if refs:
            print(f"  {pdf['filename']}: {len(refs)} refs ({sum(1 for r in resolved_refs if r.target_pdf)} resolved)")

    if all_refs:
        storage.db_client.save_metadata('__global__', 'cross_doc_refs', [cross_doc_ref_to_dict(r) for r in all_refs])
        print(f"  ✓ Saved {len(all_refs)} cross-document references")

    # 1.2: Named entity extraction
    print("\n🏷️  Step 1.2: Extracting named entities...")
    all_entities = {}
    ner = TechnicalNER()

    for pdf in all_pdfs:
        slug = pdf['slug']
        chunks = storage.get_chunks(slug)
        entity_dict = extract_entities_from_chunks(chunks, include_products=True)
        total_entities = sum(len(entities) for entities in entity_dict.values())
        all_entities[slug] = entity_dict
        if total_entities > 0:
            print(f"  {pdf['filename']}: {total_entities} entities")

    for slug, entity_dict in all_entities.items():
        serializable = {}
        for chunk_id, entities in entity_dict.items():
            serializable[chunk_id] = [entity_to_dict(e) for e in entities]
        storage.db_client.save_metadata(slug, 'extracted_entities', serializable)
    print(f"  ✓ Saved entities for {len(all_entities)} PDFs")

    # 1.3: Document versioning
    print("\n📅 Step 1.3: Detecting version relationships...")
    version_rels = analyzer.detect_version_relationships(all_pdfs)
    if version_rels:
        storage.db_client.save_metadata('__global__', 'version_relations', [version_relation_to_dict(r) for r in version_rels])
        print(f"  ✓ Found {len(version_rels)} version relationships")
    else:
        print(f"  ⊘ No version relationships found")

    # === PHASE 2: Expansion ===
    if milvus_client:
        print("\n" + "-" * 80)
        print("Phase 2: Semantic Similarity & Topic Clustering")
        print("-" * 80)

        # 2.1: Semantic similarity
        print(f"\n🔍 Step 2.1: Finding semantic similarities (threshold={similarity_threshold}, top_k={top_k})...")
        all_semantic_links = []
        for pdf in all_pdfs:
            slug = pdf['slug']
            links = analyzer.find_semantic_similarities(slug, threshold=similarity_threshold, top_k=top_k)
            all_semantic_links.extend(links)
            if links:
                print(f"  {pdf['filename']}: {len(links)} links")

        if all_semantic_links:
            storage.db_client.save_metadata('__global__', 'semantic_links', [semantic_link_to_dict(l) for l in all_semantic_links])
            print(f"  ✓ Saved {len(all_semantic_links)} semantic links")

        # 2.2: Topic clustering
        print("\n📊 Step 2.2: Clustering documents by topics...")
        n_topics = min(10, max(2, len(all_pdfs) // 2))
        topics_result = analyzer.cluster_documents_by_topic(all_pdfs, n_topics=n_topics)

        if topics_result:
            storage.db_client.save_metadata('__global__', 'topics', topics_result)
            print(f"  ✓ Found {topics_result['n_topics']} topics (silhouette: {topics_result['silhouette_score']:.3f})")

    # === PHASE 3: Advanced ===
    print("\n" + "-" * 80)
    print("Phase 3: Citation Network")
    print("-" * 80)

    print("\n📚 Building citation network...")
    citations = analyzer.build_citation_network(all_pdfs)

    if citations:
        storage.db_client.save_metadata('__global__', 'citation_network', citations)
        print(f"  ✓ Saved {len(citations)} citation relationships")
    else:
        print(f"  ⊘ No citations found")

    print("\n" + "=" * 80)
    print("✓ Cross-document relationships built successfully!")
    print("=" * 80)


def classify_aas(llm_provider: str = "gemini"):
    """
    Classify all PDFs in database to AAS submodels (Phase 1).

    Args:
        llm_provider: LLM provider ("gemini" or "mistral")

    Returns:
        Status message with classification results
    """
    print("\n" + "=" * 80)
    print("🔄 AAS CLASSIFICATION (Phase 1): Starting...")
    print("=" * 80)

    try:
        # Check if LLM provider is configured
        if llm_provider == "gemini" and not os.getenv("GEMINI_API_KEY"):
            return "❌ **Error:** GEMINI_API_KEY not configured. Set it in your .env file."
        elif llm_provider == "mistral" and not os.getenv("MISTRAL_API_KEY"):
            return "❌ **Error:** MISTRAL_API_KEY not configured. Set it in your .env file."

        # Check if there are PDFs to classify
        pdf_list = storage.list_pdfs()
        if not pdf_list:
            return "⚠️ **No PDFs found in database.** Upload PDFs first."

        # Run classification
        classifications = classify_pdfs_to_aas(
            storage=storage,
            llm_provider=llm_provider
        )

        # Build result message
        if not classifications:
            return "⚠️ **No classifications generated.** Check terminal for errors."

        # Count submodels
        submodel_counts = {}
        for classification in classifications.values():
            for submodel in classification.get('submodels', []):
                submodel_counts[submodel] = submodel_counts.get(submodel, 0) + 1

        # Format result
        result_parts = [
            "✅ **AAS Classification Complete (Phase 1)!**\n",
            f"📊 **Summary:**",
            f"- PDFs classified: {len(classifications)}",
            f"- Submodels identified: {len(submodel_counts)}\n",
            "**Submodel Distribution:**"
        ]

        for submodel, count in sorted(submodel_counts.items(), key=lambda x: x[1], reverse=True):
            result_parts.append(f"- {submodel}: {count} PDF(s)")

        result_parts.append("\n**PDF → Submodel Mapping:**")

        for slug, classification in classifications.items():
            if 'error' in classification:
                result_parts.append(f"\n❌ **{classification['filename']}**")
                result_parts.append(f"   Error: {classification['error']}")
            else:
                result_parts.append(f"\n✓ **{classification['filename']}**")
                for submodel in classification.get('submodels', []):
                    confidence = classification.get('confidence_scores', {}).get(submodel, 0)
                    result_parts.append(f"   - {submodel} (confidence: {confidence:.2f})")

        result_parts.append("\n💾 **Results saved to database** as global metadata `aas_classifications`")
        result_parts.append("\n**Next:** Click 'Extract AAS Data' to extract structured information (Phase 2)")

        print("=" * 80)
        print("✅ AAS Classification complete!")
        print("=" * 80)

        return "\n".join(result_parts)

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"\n❌ Error during AAS classification: {e}")
        print(error_details)
        return f"❌ **Error during classification:**\n\n{str(e)}\n\nCheck terminal for full traceback."


def extract_aas(llm_provider: str = "gemini"):
    """
    Extract structured AAS data from classified PDFs (Phase 2).

    Args:
        llm_provider: LLM provider ("gemini" or "mistral")

    Returns:
        Status message with extraction results
    """
    print("\n" + "=" * 80)
    print("🔄 AAS DATA EXTRACTION (Phase 2): Starting...")
    print("=" * 80)

    try:
        # Check if LLM provider is configured
        if llm_provider == "gemini" and not os.getenv("GEMINI_API_KEY"):
            return "❌ **Error:** GEMINI_API_KEY not configured. Set it in your .env file."
        elif llm_provider == "mistral" and not os.getenv("MISTRAL_API_KEY"):
            return "❌ **Error:** MISTRAL_API_KEY not configured. Set it in your .env file."

        # Check if classifications exist
        classifications = storage.db_client.get_metadata('__global__', 'aas_classifications')
        if not classifications:
            return "⚠️ **No classifications found.** Run Phase 1 (Classify) first."

        # Run extraction
        extracted_data = extract_aas_data(
            storage=storage,
            llm_provider=llm_provider
        )

        # Build result message
        if not extracted_data:
            return "⚠️ **No data extracted.** Check terminal for errors."

        # Format result
        result_parts = [
            "✅ **AAS Data Extraction Complete (Phase 2)!**\n",
            f"📊 **Extracted Data Summary:**",
            f"- Submodels with data: {len(extracted_data)}\n",
            "**Extracted Submodels:**"
        ]

        for submodel, data in extracted_data.items():
            result_parts.append(f"\n**{submodel}:**")

            # Summary based on submodel type
            if submodel == "DigitalNameplate":
                mfg = data.get('ManufacturerName', 'N/A')
                product = data.get('ManufacturerProductDesignation', 'N/A')
                result_parts.append(f"   - Manufacturer: {mfg}")
                result_parts.append(f"   - Product: {product}")

            elif submodel == "TechnicalData":
                voltage = data.get('ElectricalProperties', {}).get('VoltageRange', 'N/A')
                result_parts.append(f"   - Voltage: {voltage}")

            elif submodel == "Documentation":
                docs = data.get('Documents', [])
                result_parts.append(f"   - Documents: {len(docs)} files")

            elif submodel == "HandoverDocumentation":
                certs = data.get('Certifications', [])
                result_parts.append(f"   - Certifications: {len(certs)}")

            elif submodel == "BillOfMaterials":
                components = data.get('Components', [])
                result_parts.append(f"   - Components: {len(components)}")

            else:
                result_parts.append(f"   - Data extracted successfully")

        result_parts.append("\n💾 **Results saved to database** as global metadata `aas_extracted_data`")
        result_parts.append("\n**Data ready for XML generation (Phase 3)**")

        print("=" * 80)
        print("✅ AAS Data Extraction complete!")
        print("=" * 80)

        return "\n".join(result_parts)

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"\n❌ Error during AAS extraction: {e}")
        print(error_details)
        return f"❌ **Error during extraction:**\n\n{str(e)}\n\nCheck terminal for full traceback."


def validate_aas(llm_provider: str = "gemini"):
    """
    Validate and complete AAS extracted data (Phase 3).

    Args:
        llm_provider: LLM provider ("gemini" or "mistral")

    Returns:
        Status message with validation results
    """
    print("\n" + "=" * 80)
    print("🔄 AAS VALIDATION (Phase 3): Starting...")
    print("=" * 80)

    try:
        # Check if LLM provider is configured
        if llm_provider == "gemini" and not os.getenv("GEMINI_API_KEY"):
            return "❌ **Error:** GEMINI_API_KEY not configured. Set it in your .env file."
        elif llm_provider == "mistral" and not os.getenv("MISTRAL_API_KEY"):
            return "❌ **Error:** MISTRAL_API_KEY not configured. Set it in your .env file."

        # Check if extracted data exists
        extracted_data = storage.db_client.get_metadata('__global__', 'aas_extracted_data')
        if not extracted_data:
            return "⚠️ **No extracted data found.** Run Phase 2 (Extract) first."

        # Run validation
        completed_data, validation_report = validate_aas_data(
            storage=storage,
            llm_provider=llm_provider
        )

        # Build result message
        if not validation_report:
            return "⚠️ **Validation failed.** Check terminal for errors."

        is_complete = validation_report.get('is_complete', False)
        missing = validation_report.get('missing_items', [])
        suggestions = validation_report.get('suggestions', [])
        completion_attempts = validation_report.get('completion_attempts', 0)

        # Format result
        result_parts = [
            f"{'✅' if is_complete else '⚠️ '} **AAS Validation Complete (Phase 3)!**\n",
            f"📊 **Validation Summary:**",
            f"- Status: {'✅ Complete' if is_complete else '⚠️  Incomplete'}",
            f"- Completion attempts: {completion_attempts}",
            f"- Original submodels: {len(validation_report.get('original_submodels', []))}",
            f"- Final submodels: {len(validation_report.get('completed_submodels', []))}\n"
        ]

        # Mandatory submodels check
        mandatory_check = validation_report.get('mandatory_submodels_present', False)
        result_parts.append(f"**Mandatory Submodels:** {'✅ All present' if mandatory_check else '⚠️  Some missing'}")

        # Mandatory fields check
        fields_check = validation_report.get('mandatory_fields_complete', {})
        if fields_check:
            result_parts.append("\n**Mandatory Fields:**")
            for submodel, complete in fields_check.items():
                status = '✅' if complete else '⚠️ '
                result_parts.append(f"   {status} {submodel}")

        # Missing items
        if missing:
            result_parts.append(f"\n**⚠️  Missing Items ({len(missing)}):**")
            for item in missing[:10]:  # Show first 10
                result_parts.append(f"   - {item}")
            if len(missing) > 10:
                result_parts.append(f"   ... and {len(missing) - 10} more")

        # Suggestions
        if suggestions:
            result_parts.append(f"\n**💡 Suggestions ({len(suggestions)}):**")
            for i, suggestion in enumerate(suggestions[:5], 1):
                result_parts.append(f"   {i}. {suggestion}")
            if len(suggestions) > 5:
                result_parts.append(f"   ... and {len(suggestions) - 5} more")

        result_parts.append("\n💾 **Results saved to database:**")
        result_parts.append("   - Completed data: `aas_validated_data`")
        result_parts.append("   - Validation report: `aas_validation_report`")

        if is_complete:
            result_parts.append("\n**✅ Data is ready for XML generation (Phase 4)**")
        else:
            result_parts.append("\n**⚠️  Consider manual review before XML generation**")

        print("=" * 80)
        print("✅ AAS Validation complete!")
        print("=" * 80)

        return "\n".join(result_parts)

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"\n❌ Error during AAS validation: {e}")
        print(error_details)
        return f"❌ **Error during validation:**\n\n{str(e)}\n\nCheck terminal for full traceback."


def generate_aasx_file(llm_provider: str = "gemini", progress=gr.Progress()):
    """
    Generate complete AASX file (4-phase pipeline with progress tracking).

    Args:
        llm_provider: LLM provider ("gemini" or "mistral")
        progress: Gradio progress tracker

    Returns:
        Tuple of (status_message, file_path_for_download)
    """
    system_logger.log("=" * 80, "INFO")
    system_logger.log("🏭 AAS AASX FILE GENERATION STARTED", "INFO")
    system_logger.log("=" * 80, "INFO")

    try:
        # Check if LLM provider is configured
        if llm_provider == "gemini" and not os.getenv("GEMINI_API_KEY"):
            error_msg = "❌ **Error:** GEMINI_API_KEY not configured. Set it in your .env file."
            system_logger.log(error_msg, "ERROR")
            return error_msg, None
        elif llm_provider == "mistral" and not os.getenv("MISTRAL_API_KEY"):
            error_msg = "❌ **Error:** MISTRAL_API_KEY not configured. Set it in your .env file."
            system_logger.log(error_msg, "ERROR")
            return error_msg, None

        # Check if there are PDFs
        pdf_list = storage.list_pdfs()
        if not pdf_list:
            error_msg = "⚠️ **No PDFs found in database.** Upload PDFs first in the Q&A tab."
            system_logger.log(error_msg, "WARNING")
            return error_msg, None

        system_logger.log(f"Found {len(pdf_list)} PDFs to process", "INFO")

        # Phase 1: Classification (25% progress)
        progress(0.0, desc="Phase 1: Classifying PDFs to AAS submodels...")
        system_logger.log("📋 Phase 1: Starting PDF classification...", "INFO")

        classifications = classify_pdfs_to_aas(
            storage=storage,
            llm_provider=llm_provider
        )

        if not classifications:
            error_msg = "⚠️ **Classification failed.** Check logs."
            system_logger.log(error_msg, "ERROR")
            return error_msg, None

        system_logger.log(f"✅ Phase 1 complete: {len(classifications)} PDFs classified", "INFO")
        progress(0.25, desc="Phase 1 complete ✓")

        # Phase 2: Extraction (50% progress)
        progress(0.25, desc="Phase 2: Extracting structured AAS data...")
        system_logger.log("📊 Phase 2: Starting data extraction...", "INFO")

        extracted_data = extract_aas_data(
            storage=storage,
            llm_provider=llm_provider
        )

        if not extracted_data:
            error_msg = "⚠️ **Extraction failed.** Check logs."
            system_logger.log(error_msg, "ERROR")
            return error_msg, None

        system_logger.log(f"✅ Phase 2 complete: {len(extracted_data)} submodels extracted", "INFO")
        progress(0.50, desc="Phase 2 complete ✓")

        # Phase 3: Validation (75% progress)
        progress(0.50, desc="Phase 3: Validating and completing data...")
        system_logger.log("✅ Phase 3: Starting validation...", "INFO")

        completed_data, validation_report = validate_aas_data(
            storage=storage,
            llm_provider=llm_provider
        )

        if not validation_report:
            error_msg = "⚠️ **Validation failed.** Check logs."
            system_logger.log(error_msg, "ERROR")
            return error_msg, None

        is_complete = validation_report.get('is_complete', False)
        system_logger.log(f"✅ Phase 3 complete: Data is {'complete' if is_complete else 'incomplete'}", "INFO")
        progress(0.75, desc="Phase 3 complete ✓")

        # Phase 4: XML Generation (90% progress)
        progress(0.75, desc="Phase 4: Generating AAS v5.0 XML...")
        system_logger.log("📄 Phase 4: Starting XML generation...", "INFO")

        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("data/output")
        output_dir.mkdir(parents=True, exist_ok=True)
        xml_filename = f"aas_output_{timestamp}.xml"
        xml_path = output_dir / xml_filename

        xml_output = generate_aas_xml(
            storage=storage,
            llm_provider=llm_provider,
            output_path=xml_path
        )

        if not xml_output:
            error_msg = "⚠️ **XML generation failed.** Check logs."
            system_logger.log(error_msg, "ERROR")
            return error_msg, None

        system_logger.log(f"✅ Phase 4 complete: XML generated ({len(xml_output):,} chars)", "INFO")
        system_logger.log(f"💾 Saved to: {xml_path}", "INFO")
        progress(0.90, desc="Phase 4 complete ✓")

        # TODO: Package into AASX (zip format) - for now just provide XML
        progress(0.95, desc="Finalizing...")
        system_logger.log("📦 Note: AASX packaging not yet implemented, providing XML file", "INFO")

        progress(1.0, desc="✅ Complete!")

        # Build success message
        result_parts = [
            "✅ **AAS File Generation Complete!**\n",
            f"**Pipeline Summary:**",
            f"- Phase 1: {len(classifications)} PDFs classified",
            f"- Phase 2: {len(extracted_data)} submodels extracted",
            f"- Phase 3: Data validation {'✅ complete' if is_complete else '⚠️ incomplete'}",
            f"- Phase 4: XML generated ({len(xml_output):,} characters)\n",
            f"**Output:**",
            f"- File: `{xml_path}`",
            f"- Format: AAS v5.0 XML",
            f"- Submodels: {', '.join(completed_data.keys())}\n",
            "**✅ Download your AAS file using the button below!**"
        ]

        system_logger.log("=" * 80, "INFO")
        system_logger.log("✅ AASX FILE GENERATION COMPLETE", "INFO")
        system_logger.log("=" * 80, "INFO")

        return "\n".join(result_parts), str(xml_path)

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        system_logger.log(f"❌ Error during AASX generation: {e}", "ERROR")
        system_logger.log(error_details, "ERROR")
        return f"❌ **Error during generation:**\n\n{str(e)}\n\nCheck logs tab for details.", None


def get_system_logs():
    """Get current system logs for display."""
    return system_logger.get_logs()

def clear_system_logs():
    """Clear system logs."""
    system_logger.clear()
    return "Logs cleared."

def auto_ingest_on_startup():
    """Auto-ingest all PDFs from data/input/ on app startup."""
    system_logger.log("=" * 80, "INFO")
    system_logger.log("🔄 AUTO-INGEST: Checking data/input/ for new PDFs...", "INFO")
    system_logger.log("=" * 80, "INFO")

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
            system_logger.log("⚠️  No PDFs found in data/input/", "WARNING")
        else:
            system_logger.log(f"✓ Auto-ingest complete:", "INFO")
            system_logger.log(f"  - Newly processed: {len(results['processed'])}", "INFO")
            system_logger.log(f"  - Skipped (cached): {len(results['skipped'])}", "INFO")
            system_logger.log(f"  - Failed: {len(results['failed'])}", "INFO")

        system_logger.log("=" * 80, "INFO")

        # Build cross-document relationships if we have 2+ PDFs
        pdf_list = storage.list_pdfs()
        if len(pdf_list) >= 2:
            build_cross_document_relationships(
                storage=storage,
                similarity_threshold=0.85,
                top_k=10
            )

    except Exception as e:
        system_logger.log(f"⚠️  Auto-ingest error: {e}", "ERROR")
        system_logger.log("=" * 80, "INFO")

    # Return updated dropdown state
    return load_available_pdfs()


# Build Gradio interface with tabs
with gr.Blocks(title="PDFKG - PDF Knowledge Graph System", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏭 PDFKG - PDF Knowledge Graph & AAS System")

    with gr.Tabs() as tabs:
        # =========================
        # TAB 1: PDF Ingestion
        # =========================
        with gr.Tab("📥 PDF Ingestion", id="tab_ingest"):
            gr.Markdown("## Upload technical PDF manuals and build the knowledge graph")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ Configuration")

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
                        label="Use Gemini for Visual Analysis",
                        value=True if os.getenv("GEMINI_API_KEY") else False,
                        info="Extract cross-references from diagrams (requires GEMINI_API_KEY)",
                        interactive=True
                    )

                    force_reprocess = gr.Checkbox(
                        label="Force Reprocess",
                        value=False,
                        info="Reprocess PDF even if cached",
                        interactive=True
                    )

                    process_btn = gr.Button("🚀 Process PDF", variant="primary", size="lg")

                with gr.Column(scale=2):
                    gr.Markdown("### 📊 Processing Status")
                    status_output = gr.Markdown(
                        value="Upload PDF(s) and click *Process PDF* to populate the knowledge graph.",
                        label="Status"
                    )

                    reset_btn = gr.Button(
                        "♻️ Reset Project Data",
                        variant="stop",
                        size="md"
                    )

                    gr.Markdown(
                        "⚠️ **Dangerous:** Clears `data/output/` and rebuilds the pdfkg database in ArangoDB and Milvus.",
                        elem_id="reset-warning"
                    )

        # =========================
        # TAB 2: Q&A Interface
        # =========================
        with gr.Tab("📚 PDF Q&A", id="tab_qa"):
            gr.Markdown("## Ask questions against processed manuals")

            with gr.Row():
                with gr.Column(scale=1):
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
                        label="LLM Provider",
                        info="AI model for natural language answers"
                    )

                    top_k = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1,
                        label="Number of Sources",
                        info="More sources = more context"
                    )

                with gr.Column(scale=2):
                    gr.Markdown("### 💬 Ask Questions")

                    chatbot = gr.Chatbot(
                        height=600,
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
                        **💡 Example Questions:**
                        - What are the operative temperature limits?
                        - How do you mount this device?
                        - What certifications does this product have?
                        - What is the input voltage range?
                        """,
                        visible=True
                    )

        # =============================
        # TAB 3: AASX File Generation
        # =============================
        with gr.Tab("🏭 Generate AASX", id="tab_aasx"):
            gr.Markdown(
                """
                ## Template-Driven AAS Generation

                Choose the submodels you need, review the extracted JSON, optionally edit it, then generate the final AAS XML file.
                """
            )

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ Configuration")

                    aasx_llm_provider = gr.Radio(
                        choices=[
                            ("Gemini", "gemini"),
                            ("Mistral", "mistral"),
                        ],
                        value="gemini" if os.getenv("GEMINI_API_KEY") else "mistral",
                        label="LLM Provider",
                        info="Model used for template extraction and XML generation"
                    )

                    submodel_selector = gr.CheckboxGroup(
                        choices=SUBMODEL_CHOICES,
                        label="Submodels",
                        info="Select the submodels to extract, edit, and include in the final AAS file."
                    )

                    extract_submodels_btn = gr.Button(
                        "🧠 Extract Selected Submodels",
                        variant="primary"
                    )

                    generate_aasx_btn = gr.Button(
                        "📦 Generate AASX from Current Data",
                        variant="primary"
                    )

                with gr.Column(scale=2):
                    submodel_status = gr.Markdown(
                        value="Select one or more submodels and click **Extract** to populate their templates.",
                        label="Extraction Status"
                    )

                    download_file = gr.File(
                        label="Download AAS File",
                        visible=False
                    )

                    submodel_editor_components = {}
                    for key in AVAILABLE_SUBMODELS:
                        template = get_template(key)
                        with gr.Accordion(f"{template.display_name}", open=False, visible=False) as accordion:
                            editor = gr.Textbox(
                                value=template.default_text,
                                lines=18,
                                label=f"{template.display_name} JSON",
                                interactive=True,
                                visible=True
                            )
                            evidence_display = gr.Markdown(
                                value="No evidence captured yet.",
                                visible=True,
                                label="Evidence"
                            )
                        submodel_editor_components[key] = {
                            "accordion": accordion,
                            "editor": editor,
                            "metadata": evidence_display,
                        }

        # =======================
        # TAB 4: System Logs
        # =======================
        with gr.Tab("📋 System Logs", id="tab_logs"):
            gr.Markdown(
                """
                ## System Logs

                View real-time system logs and activity.
                """
            )

            with gr.Row():
                with gr.Column():
                    refresh_logs_btn = gr.Button("🔄 Refresh Logs", variant="secondary")
                    clear_logs_btn = gr.Button("🗑️ Clear Logs", variant="secondary")

            logs_display = gr.Textbox(
                label="Logs",
                value=system_logger.get_logs(),
                lines=30,
                max_lines=50,
                interactive=False,
                show_copy_button=True
            )

            logs_refresh_timer = gr.Timer(value=5.0, render=False)

    submodel_state = gr.State({})

    # ================================
    # EVENT HANDLERS - Tab 1 (Ingestion)
    # ================================
    process_btn.click(
        fn=process_pdf,
        inputs=[pdf_input, embed_model, max_tokens, use_gemini_ingest, force_reprocess],
        outputs=[status_output, pdf_selector, pdf_selector, chatbot, chat_input_row]
    )

    reset_btn.click(
        fn=reset_project_data,
        outputs=[status_output, pdf_selector, pdf_selector, chatbot, chat_input_row]
    )

    # ============================
    # EVENT HANDLERS - Tab 2 (Q&A)
    # ============================
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

    chatbot.change(
        lambda: gr.update(visible=True),
        outputs=clear_btn
    )

    # ============================
    # EVENT HANDLERS - Tab 3 (AASX)
    # ============================
    extract_outputs = [submodel_status, submodel_state, download_file]
    extract_outputs.extend([submodel_editor_components[key]['editor'] for key in SUBMODEL_EDITOR_ORDER])
    extract_outputs.extend([submodel_editor_components[key]['metadata'] for key in SUBMODEL_EDITOR_ORDER])
    extract_outputs.extend([submodel_editor_components[key]['accordion'] for key in SUBMODEL_EDITOR_ORDER])

    extract_submodels_btn.click(
        fn=extract_submodels_handler,
        inputs=[submodel_selector, aasx_llm_provider, submodel_state],
        outputs=extract_outputs
    )

    for key in SUBMODEL_EDITOR_ORDER:
        editor_component = submodel_editor_components[key]['editor']
        editor_component.change(
            fn=lambda value, state, submodel_key=key: update_submodel_editor(value, state, submodel_key),
            inputs=[editor_component, submodel_state],
            outputs=[submodel_state]
        )

    generate_aasx_btn.click(
        fn=generate_selected_aasx,
        inputs=[aasx_llm_provider, submodel_selector, submodel_state],
        outputs=[submodel_status, download_file]
    )

    submodel_selector.change(
        fn=lambda _: gr.update(value=None, visible=False),
        inputs=[submodel_selector],
        outputs=download_file
    )

    # ============================
    # EVENT HANDLERS - Tab 4 (Logs)
    # ============================
    refresh_logs_btn.click(
        fn=get_system_logs,
        outputs=logs_display
    )

    clear_logs_btn.click(
        fn=clear_system_logs,
        outputs=logs_display
    )

    # Auto-refresh logs every 5 seconds
    logs_refresh_timer.tick(
        fn=get_system_logs,
        outputs=logs_display
    )

    demo.load(
        fn=get_system_logs,
        outputs=logs_display
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
        gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
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

    # Try ports in sequence until one is available
    ports_to_try = [8016, 8017, 8018, 8019, 7860]
    server_port = None

    for port in ports_to_try:
        try:
            # Launch with share=True for public link
            demo.launch(
                share=True,
                server_name="0.0.0.0",
                server_port=port,
                show_error=True
            )
            server_port = port
            break  # Success!
        except OSError as e:
            if "Cannot find empty port" in str(e):
                print(f"⚠️  Port {port} is busy, trying next port...")
                continue
            else:
                raise  # Re-raise if it's a different error

    if server_port is None:
        print(f"\n❌ ERROR: All ports {ports_to_try} are busy!")
        print("Kill existing processes with: pkill -f 'python.*app.py'")
        sys.exit(1)
