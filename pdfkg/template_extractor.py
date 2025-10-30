"""Template-driven AAS submodel extraction utilities with field-level evidence."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import logging

from pdfkg.submodel_templates import get_template, list_submodel_templates
from pdfkg import llm_stats

logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    GEMINI_AVAILABLE = False

try:
    from mistralai import Mistral
    MISTRAL_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    MISTRAL_AVAILABLE = False


SUBMODEL_QUERIES: Dict[str, str] = {
    "DigitalNameplate": "manufacturer name product designation serial number year of construction",
    "TechnicalData": "technical specifications voltage current power torque weight dimensions",
    "Documentation": "manual datasheet drawing document reference certificate",
    "HandoverDocumentation": "certificate compliance declaration warranty handover",
    "MaintenanceRecord": "maintenance schedule interval service procedure spare parts",
    "OperationalData": "operating mode startup shutdown commissioning safety",
    "BillOfMaterials": "bill of materials component list part number quantity",
    "CarbonFootprint": "carbon footprint emissions energy consumption sustainability",
}


@dataclass
class FieldExtraction:
    path: str
    value: Any
    confidence: float
    sources: List[str]
    notes: str = ""


@dataclass
class ExtractionResult:
    data: Dict[str, Any]
    metadata: Dict[str, Dict[str, Any]]


class TemplateAASExtractor:
    """Extract submodel data using JSON templates as the contract."""

    def __init__(self, storage, llm_provider: str = "gemini") -> None:
        self.storage = storage
        self.llm_provider = llm_provider.lower()

        if self.llm_provider == "gemini":
            if not GEMINI_AVAILABLE:
                raise ImportError("google-generativeai not installed")
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not configured")
            genai.configure(api_key=api_key)
            model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            self.llm_client = genai.GenerativeModel(model_name)
        elif self.llm_provider == "mistral":
            if not MISTRAL_AVAILABLE:
                raise ImportError("mistralai not installed")
            api_key = os.getenv("MISTRAL_API_KEY")
            if not api_key:
                raise ValueError("MISTRAL_API_KEY not configured")
            self.llm_client = Mistral(api_key=api_key)
            self.mistral_model = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")

        self.curated_blocks: Dict[str, List[str]] = {}
        self.block_fields: Dict[str, List[str]] = {}
        self.curated_dir = Path("data/input/curated")
        self.curated_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def extract(self, submodels: Iterable[str], progress_callback=None) -> Dict[str, ExtractionResult]:
        pdf_slugs = [pdf["slug"] for pdf in self.storage.list_pdfs()]
        if not pdf_slugs:
            raise RuntimeError("No PDFs available for extraction. Ingest PDFs first.")

        results: Dict[str, ExtractionResult] = {}
        submodel_list = list(submodels)
        total_submodels = len(submodel_list)

        for i, submodel in enumerate(submodel_list):
            if progress_callback:
                progress_callback(i / total_submodels, f"Extracting {submodel}...")

            template = get_template(submodel)
            data, metadata = self._extract_submodel(submodel, template.schema, pdf_slugs)
            results[submodel] = ExtractionResult(data=data, metadata=metadata)

        if progress_callback:
            progress_callback(1.0, "Extraction complete.")

        return results

    # ------------------------------------------------------------------
    def _extract_submodel(self, submodel: str, schema: Any, pdf_slugs: List[str]) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        data, metadata = self._fill_structure(submodel, schema, [], pdf_slugs)
        return data, metadata

    def _fill_structure(
        self,
        submodel: str,
        schema: Any,
        path: List[str],
        pdf_slugs: List[str],
    ) -> Tuple[Any, Dict[str, Dict[str, Any]]]:
        if isinstance(schema, dict):
            result: Dict[str, Any] = {}
            metadata: Dict[str, Dict[str, Any]] = {}
            for key, value in schema.items():
                filled, meta = self._fill_structure(submodel, value, path + [key], pdf_slugs)
                result[key] = filled
                metadata.update(meta)
            return result, metadata

        if isinstance(schema, list):
            list_schema = schema[0] if schema else None
            extraction = self._extract_list_field(submodel, path, list_schema, pdf_slugs)
            meta = {
                extraction.path: {
                    "confidence": extraction.confidence,
                    "sources": extraction.sources,
                    "notes": extraction.notes,
                }
            }
            return extraction.value, meta

        field_extraction = self._extract_field(submodel, path, pdf_slugs)
        meta = {
            field_extraction.path: {
                "confidence": field_extraction.confidence,
                "sources": field_extraction.sources,
                "notes": field_extraction.notes,
            }
        }
        return field_extraction.value, meta

    # ------------------------------------------------------------------
    def _extract_field(self, submodel: str, path: List[str], pdf_slugs: List[str]) -> FieldExtraction:
        field_path = ".".join(path)
        field_label = path[-1] if path else submodel
        self._register_field(submodel, field_path)
        evidence = self._collect_field_evidence(submodel, field_path, pdf_slugs)
        prompt = self._build_field_prompt(submodel, field_path, field_label, evidence)
        response = self._query_llm(
            prompt,
            phase="extraction",
            label=f"{submodel}:{field_path}",
        )
        extracted = self._parse_field_response(response)

        value = extracted.get("value")
        confidence = float(extracted.get("confidence", 0.0)) if _is_number(extracted.get("confidence")) else 0.0
        sources = extracted.get("sources") or []
        notes = extracted.get("notes", "")

        return FieldExtraction(path=field_path, value=value, confidence=confidence, sources=sources, notes=notes)

    def _extract_list_field(self, submodel: str, path: List[str], item_schema: Any, pdf_slugs: List[str]) -> FieldExtraction:
        field_path = ".".join(path) + "[]"
        self._register_field(submodel, ".".join(path))
        evidence = self._collect_field_evidence(submodel, ".".join(path), pdf_slugs, max_snippets=8)
        template_json = json.dumps(item_schema, indent=2) if item_schema is not None else "null"
        prompt = self._build_list_prompt(submodel, field_path, template_json, evidence)
        response = self._query_llm(
            prompt,
            phase="extraction",
            label=f"{submodel}:{field_path}",
        )
        extracted = self._parse_field_response(response)

        items = extracted.get("items")
        if not isinstance(items, list):
            items = []
        confidence = float(extracted.get("confidence", 0.0)) if _is_number(extracted.get("confidence")) else 0.0
        sources = extracted.get("sources") or []
        notes = extracted.get("notes", "")

        return FieldExtraction(path=field_path, value=items, confidence=confidence, sources=sources, notes=notes)

    # ------------------------------------------------------------------
    def _collect_field_evidence(
        self,
        submodel: str,
        field_path: str,
        pdf_slugs: List[str],
        max_snippets: int = 5,
    ) -> List[str]:
        block_id = self._block_identifier(submodel, field_path)
        curated = self._get_curated_block(block_id, pdf_slugs)

        if curated:
            snippets = curated[:max_snippets]
        else:
            snippets = self._gather_raw_snippets(field_path, pdf_slugs, max_snippets)

        if not snippets:
            snippets = ["No direct evidence found in documents."]

        return snippets[:max_snippets]

    def _register_field(self, submodel: str, field_path: str) -> None:
        block_id = self._block_identifier(submodel, field_path)
        self.block_fields.setdefault(block_id, [])
        clean_path = field_path.rstrip(".")
        if clean_path and clean_path not in self.block_fields[block_id]:
            self.block_fields[block_id].append(clean_path)

    @staticmethod
    def _block_identifier(submodel: str, field_path: str) -> str:
        path = field_path.replace("[]", "")
        root = path.split(".", 1)[0] if path else submodel
        return f"{submodel}:{root}"

    def _gather_raw_snippets(
        self,
        field_path: str,
        pdf_slugs: List[str],
        max_snippets: int = 5,
        token_override: Optional[List[str]] = None,
    ) -> List[str]:
        tokens = token_override or [
            token for token in field_path.replace('.', ' ').replace('_', ' ').lower().split() if len(token) > 2
        ]
        snippets: List[str] = []
        seen: set[str] = set()
        max_candidates = max(max_snippets * 4, max_snippets)

        for slug in pdf_slugs:
            pdf_info = self.storage.get_pdf_metadata(slug) or {}
            filename = pdf_info.get("filename", slug)
            chunks = self.storage.get_chunks(slug)
            for chunk in chunks:
                text = chunk.get("text", "")
                if not text:
                    continue
                lower = text.lower()
                if tokens and not any(token in lower for token in tokens):
                    continue
                snippet = self._format_snippet(filename, chunk, text)
                if snippet in seen:
                    continue
                snippets.append(snippet)
                seen.add(snippet)
                if len(snippets) >= max_candidates:
                    return snippets

        if not snippets:
            context = self._build_context(pdf_slugs, field_path.split('.')[0] if field_path else '')
            for block in context.split("\n\n"):
                block = block.strip()
                if block:
                    snippets.append(block)
                if len(snippets) >= max_candidates:
                    break

        return snippets

    def _get_curated_block(self, block_id: str, pdf_slugs: List[str]) -> List[str]:
        if block_id in self.curated_blocks:
            return self.curated_blocks[block_id]

        tokens: List[str] = []
        for field in self.block_fields.get(block_id, []):
            tokens.extend(
                token for token in field.replace('.', ' ').replace('_', ' ').lower().split() if len(token) > 2
            )
        tokens = list(dict.fromkeys(tokens))

        block_name = block_id.split(":", 1)[1] if ":" in block_id else block_id
        raw_snippets = self._gather_raw_snippets(
            field_path=block_name,
            pdf_slugs=pdf_slugs,
            max_snippets=15,
            token_override=tokens or None,
        )

        if not raw_snippets:
            self.curated_blocks[block_id] = []
            return []

        submodel = block_id.split(":", 1)[0] if ":" in block_id else "submodel"
        joined_raw = "\n".join(raw_snippets[:30])
        prompt = (
            f"You are curating evidence for the '{block_name}' section of the {submodel} submodel.\n"
            "From the raw text below, keep only concise paragraphs that contain concrete facts, measurements, "
            "lists of relevant components, maintenance steps, or other data that could fill structured fields. "
            "Remove boilerplate, tables, duplicate entries, and generic statements.\n"
            "Return the filtered content as plain text paragraphs separated by blank lines. "
            "If nothing relevant remains, return exactly 'NO_DATA'.\n\n"
            "RAW TEXT:\n"
            "```text\n"
            f"{joined_raw}\n"
            "```"
        )

        try:
            curated_text = self._query_llm(
                prompt,
                phase="curation",
                label=block_id,
            )
        except Exception as exc:
            logger.warning("Curation failed for block %s: %s", block_id, exc)
            curated_text = ""

        curated_snippets: List[str] = []
        if curated_text:
            cleaned = curated_text.strip()
            if cleaned.upper().startswith("NO_DATA"):
                curated_snippets = []
            else:
                parts = [part.strip() for part in cleaned.split("\n\n") if part.strip()]
                if not parts:
                    parts = [line.strip() for line in cleaned.splitlines() if line.strip()]
                curated_snippets = parts

        if not curated_snippets:
            curated_snippets = raw_snippets[:10]
        else:
            output_path = self.curated_dir / f"{block_id.replace(':', '_')}.txt"
            try:
                output_path.write_text("\n\n".join(curated_snippets), encoding="utf-8")
            except Exception as exc:
                logger.warning("Failed to write curated evidence for %s: %s", block_id, exc)

        self.curated_blocks[block_id] = curated_snippets
        return curated_snippets

    def _format_snippet(self, filename: str, chunk: Dict[str, Any], text: str) -> str:
        page = chunk.get("page")
        clean = " ".join(text.strip().split())[:220]
        prefix = f"[{filename}"
        if page is not None:
            prefix += f" | page {page}"
        prefix += "] "
        return prefix + clean

    def _build_field_prompt(self, submodel: str, field_path: str, field_label: str, evidence: List[str]) -> str:
        evidence_block = "\n".join(f"{idx+1}. {snippet}" for idx, snippet in enumerate(evidence))
        return (
            f"You are extracting data for the '{field_label}' field (path: {field_path}) of the {submodel} submodel.\n"
            "Use only the evidence provided. If the information is missing, return null with confidence 0.0.\n"
            "Respond in JSON with the following structure:\n"
            '{"value": ..., "confidence": 0.0-1.0, "sources": ["..."], "notes": "optional"}'
            "\n\nEvidence:\n"
            f"{evidence_block}\n"
        )

    def _build_list_prompt(self, submodel: str, field_path: str, template_json: str, evidence: List[str]) -> str:
        evidence_block = "\n".join(f"{idx+1}. {snippet}" for idx, snippet in enumerate(evidence))
        return (
            f"You are extracting a list for path '{field_path}' within the {submodel} submodel.\n"
            "Use only the evidence provided. Fill the template structure for each list entry.\n"
            "Respond in JSON with the structure:\n"
            '{"items": [ ... ], "confidence": 0.0-1.0, "sources": ["..."], "notes": "optional"}'
            "\n\nItem template (JSON):\n"
            f"```json\n{template_json}\n```\n\n"
            "Evidence:\n"
            f"{evidence_block}\n"
        )

    def _build_context(self, pdf_slugs: List[str], submodel: str) -> str:
        query_hint = SUBMODEL_QUERIES.get(submodel, submodel)
        context_blocks: List[str] = []

        for slug in pdf_slugs:
            pdf_info = self.storage.get_pdf_metadata(slug) or {}
            filename = pdf_info.get("filename", slug)
            chunks = self.storage.get_chunks(slug)[:5]
            text_snippets = "\n".join(chunk.get("text", "") for chunk in chunks if chunk.get("text"))
            if not text_snippets:
                continue
            block = f"[PDF: {filename}]\n{text_snippets}"
            context_blocks.append(block)

        if not context_blocks:
            return f"No textual context found. Focus on template: {query_hint}."

        return "\n\n".join(context_blocks[:8])  # limit context size

    def _query_llm(self, prompt: str, *, phase: str, label: str) -> str:
        max_retries = 4
        retry_delay = 30  # seconds

        for attempt in range(max_retries + 1):
            start_time = time.time()
            try:
                if self.llm_provider == "gemini":
                    response = self.llm_client.generate_content(
                        prompt,
                        request_options={'timeout': 240}  # 4 minutes timeout
                    )
                    usage = getattr(response, "usage_metadata", None)
                    tokens_in, tokens_out, total_tokens = llm_stats.extract_token_usage(usage)
                    elapsed_ms = int((time.time() - start_time) * 1000)
                    llm_stats.record_call(
                        self.llm_provider,
                        phase=phase,
                        label=label,
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                        total_tokens=total_tokens,
                        metadata={
                            "model": getattr(self.llm_client, "model_name", os.getenv("GEMINI_MODEL", "gemini-2.5-flash")),
                            "elapsed_ms": elapsed_ms,
                            "prompt_chars": len(prompt),
                        },
                    )
                    return response.text
                elif self.llm_provider == "mistral":
                    response = self.llm_client.chat.complete(
                        model=self.mistral_model,
                        messages=[{"role": "user", "content": prompt}],
                        timeout=240  # 4 minutes timeout
                    )
                    content = response.choices[0].message.content
                    # Mistral sometimes returns list, ensure it's always a string
                    if isinstance(content, list):
                        content = "\n".join(str(item) for item in content)
                    usage = getattr(response, "usage", None)
                    tokens_in, tokens_out, total_tokens = llm_stats.extract_token_usage(usage)
                    elapsed_ms = int((time.time() - start_time) * 1000)
                    llm_stats.record_call(
                        self.llm_provider,
                        phase=phase,
                        label=label,
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                        total_tokens=total_tokens,
                        metadata={
                            "model": self.mistral_model,
                            "elapsed_ms": elapsed_ms,
                            "prompt_chars": len(prompt),
                        },
                    )
                    return str(content)
                else:
                    raise RuntimeError("Unsupported LLM provider")

            except Exception as e:
                error_str = str(e).lower()
                is_timeout = any(keyword in error_str for keyword in ['timeout', '504', 'deadline', 'timed out'])

                if is_timeout and attempt < max_retries:
                    print(f"⚠️  Timeout on attempt {attempt + 1}/{max_retries + 1}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    continue
                else:
                    # Re-raise if not a timeout or max retries reached
                    raise

    def _parse_field_response(self, response_text: str) -> Dict[str, Any]:
        # Ensure response_text is a string (defensive programming)
        if not isinstance(response_text, str):
            if isinstance(response_text, list):
                response_text = "\n".join(str(item) for item in response_text)
            else:
                response_text = str(response_text)

        cleaned = response_text.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in cleaned:
            cleaned = cleaned.split("```", 1)[1].split("```", 1)[0]
        cleaned = cleaned.strip()

        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            return {"value": None, "confidence": 0.0, "sources": [], "notes": "LLM response parse error."}

        if not isinstance(payload, dict):
            return {"value": None, "confidence": 0.0, "sources": [], "notes": "LLM response not a JSON object."}

        payload.setdefault("sources", [])
        payload.setdefault("notes", "")
        return payload


def extract_submodels(storage, submodels: Iterable[str], llm_provider: str = "gemini", progress_callback=None) -> Dict[str, Dict[str, Any]]:
    extractor = TemplateAASExtractor(storage=storage, llm_provider=llm_provider)
    raw_results = extractor.extract(submodels, progress_callback=progress_callback)
    prepared: Dict[str, Dict[str, Any]] = {}
    for key, result in raw_results.items():
        prepared[key] = {"data": result.data, "metadata": result.metadata}
    return prepared


def available_submodels() -> List[str]:
    return list_submodel_templates()


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False
