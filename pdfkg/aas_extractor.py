"""
AAS Data Extractor - Phase 2

Extract structured data for each AAS submodel using:
- Named Entity Recognition (NER)
- Semantic search (Milvus)
- LLM queries for structured extraction

Submodels:
1. DigitalNameplate - Basic identification
2. TechnicalData - Technical specifications
3. Documentation - File references
4. HandoverDocumentation - Certificates, warranties
5. MaintenanceRecord - Maintenance information
6. OperationalData - Operational parameters
7. BillOfMaterials - Component lists
8. CarbonFootprint - Environmental data
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Any
from collections import defaultdict

from pdfkg import llm_stats
from pdfkg.llm.config import resolve_llm_provider
from pdfkg.llm.mistral_client import chat as mistral_chat, get_model_name as get_mistral_model_name
from pdfkg.submodel_templates import template_schema
from pdfkg.template_utils import sanitize_submodel_data


def _format_unique_examples(items: List[str], limit: int) -> str:
    """Return a comma-separated preview of unique values in original order."""
    if not items:
        return "None"
    unique_items = [item for item in dict.fromkeys(items) if item]
    if not unique_items:
        return "None"
    return ", ".join(unique_items[:limit])

# LLM imports
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from mistralai import Mistral
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False


class AASDataExtractor:
    """
    Extract structured data for AAS submodels from classified PDFs.
    """

    def __init__(self, storage, llm_provider: Optional[str] = None):
        """
        Initialize AAS Data Extractor.

        Args:
            storage: Storage backend (with Milvus and ArangoDB)
            llm_provider: LLM provider ("gemini" or "mistral")
        """
        self.storage = storage
        self.llm_provider = resolve_llm_provider(llm_provider)
        self.max_workers = max(1, int(os.getenv("AAS_EXTRACTION_CONCURRENCY", "3")))

        # Initialize LLM client
        if self.llm_provider == "gemini":
            if not GEMINI_AVAILABLE:
                raise ImportError("google-generativeai not installed")
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment")
            genai.configure(api_key=api_key)
            model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            self.llm_client = genai.GenerativeModel(model_name)
            self.gemini_model = model_name
            print(f"✅ Initialized Gemini model: {model_name}")

        elif self.llm_provider == "mistral":
            if not MISTRAL_AVAILABLE:
                raise ImportError("mistralai not installed")
            api_key = os.getenv("MISTRAL_API_KEY")
            if not api_key:
                raise ValueError("MISTRAL_API_KEY not found in environment")
            self.mistral_model = get_mistral_model_name()
            print(f"✅ Initialized Mistral model: {self.mistral_model}")

        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")

    def extract_all_submodel_data(self) -> Dict[str, Any]:
        """
        Extract data for all submodels from all classified PDFs.

        Returns:
            Dict with extracted data organized by submodel
        """
        print("\n" + "=" * 80)
        print("AAS DATA EXTRACTION - Phase 2")
        print("=" * 80)

        # Load classifications from Phase 1
        classifications = self.storage.db_client.get_metadata('__global__', 'aas_classifications')
        if not classifications:
            print("\n❌ No classifications found. Run Phase 1 first (--classify-aas)")
            return {}

        print(f"\n📚 Found {len(classifications)} classified PDFs")

        # Organize PDFs by submodel
        pdfs_by_submodel = self._organize_pdfs_by_submodel(classifications)

        # Extract data for each submodel
        submodel_data = {}
        extraction_functions = {
            "DigitalNameplate": self._extract_digital_nameplate,
            "TechnicalData": self._extract_technical_data,
            "Documentation": self._extract_documentation,
            "HandoverDocumentation": self._extract_handover_documentation,
            "MaintenanceRecord": self._extract_maintenance_record,
            "OperationalData": self._extract_operational_data,
            "BillOfMaterials": self._extract_bill_of_materials,
            "CarbonFootprint": self._extract_carbon_footprint,
        }

        print(f"⚙️  Concurrency: {self.max_workers} worker(s)")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_map = {}

            for submodel_key, extractor in extraction_functions.items():
                if submodel_key not in pdfs_by_submodel:
                    continue

                pdf_list = pdfs_by_submodel[submodel_key]
                future = executor.submit(
                    self._run_submodel_extraction,
                    submodel_key,
                    extractor,
                    pdf_list,
                )
                future_map[future] = submodel_key

            for future in as_completed(future_map):
                submodel_key = future_map[future]

                try:
                    result = future.result()
                    submodel_data[submodel_key] = result
                except Exception as exc:
                    print(f"  ❌ Extraction failed for {submodel_key}: {exc}")

        # Save to storage
        self.storage.db_client.save_metadata('__global__', 'aas_extracted_data', submodel_data)
        print(f"\n✅ Saved extracted data for {len(submodel_data)} submodels")

        print("\n" + "=" * 80)
        print("✓ Data extraction complete!")
        print("=" * 80)

        return submodel_data

    def _run_submodel_extraction(self, submodel_key: str, extractor_fn, pdf_list: List[str]) -> Any:
        """Run a submodel extraction function with logging."""
        print("\n" + "-" * 80)
        print(f"Extracting: {submodel_key}")
        print("-" * 80)
        return extractor_fn(pdf_list)

    def _organize_pdfs_by_submodel(self, classifications: Dict) -> Dict[str, List[str]]:
        """Organize PDF slugs by their assigned submodels."""
        pdfs_by_submodel = defaultdict(list)

        for pdf_slug, classification in classifications.items():
            if 'error' in classification:
                continue

            for submodel in classification.get('submodels', []):
                pdfs_by_submodel[submodel].append(pdf_slug)

        # Print organization
        print("\n📊 PDFs organized by submodel:")
        for submodel, slugs in sorted(pdfs_by_submodel.items()):
            print(f"   {submodel}: {len(slugs)} PDF(s)")

        return dict(pdfs_by_submodel)

    def _extract_digital_nameplate(self, pdf_slugs: List[str]) -> Dict:
        """
        Extract DigitalNameplate data: manufacturer, product, serial, year.
        """
        print(f"  Processing {len(pdf_slugs)} PDF(s)...")

        # Collect NER entities from all PDFs
        all_entities = {}
        for slug in pdf_slugs:
            entities = self.storage.db_client.get_metadata(slug, 'extracted_entities')
            if entities:
                all_entities[slug] = entities

        # Aggregate entities
        manufacturers = []
        products = []
        serials = []
        years = []
        part_numbers = []

        for slug, entity_dict in all_entities.items():
            for chunk_id, entities in entity_dict.items():
                for entity in entities:
                    entity_type = entity.get('type', '')
                    entity_text = entity.get('text', '')

                    if entity_type == 'manufacturer':
                        manufacturers.append(entity_text)
                    elif entity_type in ('model_number', 'product'):
                        products.append(entity_text)
                    elif entity_type == 'serial_number':
                        serials.append(entity_text)
                    elif entity_type == 'year':
                        years.append(entity_text)
                    elif entity_type == 'part_number':
                        part_numbers.append(entity_text)

        # Semantic search for missing data
        search_results = self._semantic_search_all_pdfs(
            pdf_slugs,
            "manufacturer name product designation model serial number year of construction",
            top_k=5
        )

        # Build context for LLM
        context_parts = []
        for result in search_results[:10]:
            context_parts.append(f"[{result['pdf_filename']}] {result['text']}")

        context = "\n\n".join(context_parts)
        template_json = json.dumps(template_schema("DigitalNameplate"), indent=2)

        # LLM extraction
        prompt = f"""Extract identification information for this industrial product.

Context from documents:
{context}

Entities found:
- Manufacturers: {_format_unique_examples(manufacturers, 5)}
- Products: {_format_unique_examples(products, 5)}
- Serial numbers: {_format_unique_examples(serials, 3)}
- Years: {_format_unique_examples(years, 3)}
- Part numbers: {_format_unique_examples(part_numbers, 3)}

Template JSON (use exactly these keys, no additions):
```json
{template_json}
```

Instructions:
- Fill each field with the best-supported value from the evidence above.
- Use null when information is not confirmed.
- Do not add new keys or rename existing ones.
"""

        llm_response = self._query_llm(prompt, label="DigitalNameplate")
        extracted_data = self._parse_json_response(llm_response) or {}
        extracted_data = sanitize_submodel_data("DigitalNameplate", extracted_data)

        # Fallbacks using aggregated entities if LLM returned nulls
        if not extracted_data.get("ManufacturerName"):
            fallback = next((m for m in manufacturers if m), None)
            if fallback:
                extracted_data["ManufacturerName"] = str(fallback).strip()
        if not extracted_data.get("ManufacturerProductDesignation"):
            fallback = next((p for p in products if p), None)
            if fallback:
                extracted_data["ManufacturerProductDesignation"] = str(fallback).strip()
        if not extracted_data.get("SerialNumber"):
            fallback = next((s for s in serials if s), None)
            if fallback:
                extracted_data["SerialNumber"] = str(fallback).strip()
        if not extracted_data.get("YearOfConstruction"):
            fallback = next((y for y in years if y), None)
            if fallback:
                extracted_data["YearOfConstruction"] = str(fallback).strip()
        if not extracted_data.get("PartNumber"):
            fallback = next((pn for pn in part_numbers if pn), None)
            if fallback:
                extracted_data["PartNumber"] = str(fallback).strip()

        print(f"  ✓ Extracted: Manufacturer={extracted_data.get('ManufacturerName', 'N/A')}")
        print(f"             Product={extracted_data.get('ManufacturerProductDesignation', 'N/A')}")

        return extracted_data

    def _extract_technical_data(self, pdf_slugs: List[str]) -> Dict:
        """
        Extract TechnicalData: specifications, electrical, mechanical properties.
        """
        print(f"  Processing {len(pdf_slugs)} PDF(s)...")

        # Semantic search for technical specifications
        search_results = self._semantic_search_all_pdfs(
            pdf_slugs,
            "technical specifications voltage current power dimensions weight IP rating temperature range pressure",
            top_k=15
        )

        # Build context
        context_parts = []
        for result in search_results[:15]:
            context_parts.append(f"[{result['pdf_filename']}] {result['text']}")

        context = "\n\n".join(context_parts)

        template_json = json.dumps(template_schema("TechnicalData"), indent=2)

        # LLM extraction
        prompt = f"""Extract technical specifications for this product.

Context:
{context}

Template JSON (use exactly these keys, no additions):
```json
{template_json}
```

Instructions:
- Populate each field using measurements or identifiers from the context.
- Include units (e.g., '2.5 kg', '100-240 VAC') when available.
- Use null when information is not confirmed.
- Do not add new keys or rename existing ones.
"""

        llm_response = self._query_llm(prompt, label="TechnicalData")
        extracted_data = self._parse_json_response(llm_response) or {}
        extracted_data = sanitize_submodel_data("TechnicalData", extracted_data)

        print(f"  ✓ Extracted technical specifications")

        return extracted_data

    def _extract_documentation(self, pdf_slugs: List[str]) -> Dict:
        """
        Extract Documentation: map PDFs to document references.
        """
        print(f"  Processing {len(pdf_slugs)} PDF(s)...")

        documents = []

        for slug in pdf_slugs:
            pdf_info = self.storage.get_pdf_metadata(slug)

            # Classify document type
            filename = pdf_info['filename'].lower()
            if 'manual' in filename:
                doc_type = "Manual"
            elif 'datasheet' in filename or 'data-sheet' in filename:
                doc_type = "Datasheet"
            elif 'safety' in filename:
                doc_type = "SafetyManual"
            elif 'installation' in filename:
                doc_type = "InstallationGuide"
            else:
                doc_type = "TechnicalDocument"

            # Detect language (simple heuristic)
            if '-EN' in pdf_info['filename'] or '_EN' in pdf_info['filename']:
                language = "en"
            elif '-DE' in pdf_info['filename'] or '_DE' in pdf_info['filename']:
                language = "de"
            elif '-ES' in pdf_info['filename'] or '_ES' in pdf_info['filename']:
                language = "es"
            else:
                language = "en"  # Default

            doc_entry = {
                "Title": pdf_info['filename'],
                "Type": doc_type,
                "Language": language,
                "Version": None,
                "Location": f"/documents/{pdf_info['filename']}",
                "Notes": None
            }

            documents.append(doc_entry)

        print(f"  ✓ Mapped {len(documents)} documents")

        return sanitize_submodel_data("Documentation", {"Documents": documents})

    def _extract_handover_documentation(self, pdf_slugs: List[str]) -> Dict:
        """
        Extract HandoverDocumentation: certificates, warranties, compliance.
        """
        print(f"  Processing {len(pdf_slugs)} PDF(s)...")

        # Collect certification entities
        all_certs = set()
        for slug in pdf_slugs:
            entities = self.storage.db_client.get_metadata(slug, 'extracted_entities')
            if entities:
                for chunk_id, entity_list in entities.items():
                    for entity in entity_list:
                        if entity.get('type') == 'certification':
                            all_certs.add(entity.get('text'))

        # Semantic search for certificates
        search_results = self._semantic_search_all_pdfs(
            pdf_slugs,
            "certificate certification CE UL ATEX compliance declaration conformity warranty",
            top_k=10
        )

        # Build context
        context_parts = []
        for result in search_results[:10]:
            context_parts.append(f"[{result['pdf_filename']}] {result['text']}")

        context = "\n\n".join(context_parts)

        template_json = json.dumps(template_schema("HandoverDocumentation"), indent=2)

        # LLM extraction
        prompt = f"""Extract certificate and declaration information.

Context:
{context}

Known certifications: {', '.join(sorted(all_certs)) if all_certs else 'None'}

Template JSON (use exactly these keys, no additions):
```json
{template_json}
```

Instructions:
- Populate certificate and declaration entries using explicit evidence from the context.
- Use null when information is not confirmed.
- Leave arrays empty if no data is available.
- Do not add new keys or rename existing ones.
"""

        llm_response = self._query_llm(prompt, label="HandoverDocumentation")
        extracted_data = self._parse_json_response(llm_response) or {}
        extracted_data = sanitize_submodel_data("HandoverDocumentation", extracted_data)

        print(f"  ✓ Extracted {len(extracted_data.get('Certificates', []))} certificates")

        return extracted_data

    def _extract_maintenance_record(self, pdf_slugs: List[str]) -> Dict:
        """
        Extract MaintenanceRecord: maintenance schedules, procedures.
        """
        print(f"  Processing {len(pdf_slugs)} PDF(s)...")

        # Semantic search
        search_results = self._semantic_search_all_pdfs(
            pdf_slugs,
            "maintenance service inspection cleaning replacement spare parts preventive maintenance schedule interval",
            top_k=10
        )

        # Build context
        context_parts = []
        for result in search_results[:10]:
            context_parts.append(f"[{result['pdf_filename']}] {result['text']}")

        context = "\n\n".join(context_parts)

        template_json = json.dumps(template_schema("MaintenanceRecord"), indent=2)

        # LLM extraction
        prompt = f"""Extract maintenance-related information.

Context:
{context}

Template JSON (use exactly these keys, no additions):
```json
{template_json}
```

Instructions:
- Summarize maintenance intervals using concise text.
- Populate procedure and spare part lists with concrete tasks and references.
- Use null when information is not confirmed.
- Do not add new keys or rename existing ones.
"""

        llm_response = self._query_llm(prompt, label="MaintenanceRecord")
        extracted_data = self._parse_json_response(llm_response) or {}
        extracted_data = sanitize_submodel_data("MaintenanceRecord", extracted_data)

        print(f"  ✓ Extracted maintenance information")

        return extracted_data

    def _extract_operational_data(self, pdf_slugs: List[str]) -> Dict:
        """
        Extract OperationalData: operating parameters, settings, conditions.
        """
        print(f"  Processing {len(pdf_slugs)} PDF(s)...")

        # Semantic search
        search_results = self._semantic_search_all_pdfs(
            pdf_slugs,
            "operating conditions parameters settings startup shutdown commissioning operating mode",
            top_k=10
        )

        # Build context
        context_parts = []
        for result in search_results[:10]:
            context_parts.append(f"[{result['pdf_filename']}] {result['text']}")

        context = "\n\n".join(context_parts)

        template_json = json.dumps(template_schema("OperationalData"), indent=2)

        # LLM extraction
        prompt = f"""Summarize operational data for this equipment.

Context:
{context}

Template JSON (use exactly these keys, no additions):
```json
{template_json}
```

Instructions:
- Provide concise textual descriptions for each field based on the context.
- Use null when information is not confirmed.
- Do not add new keys or rename existing ones.
"""

        llm_response = self._query_llm(prompt, label="OperationalData")
        extracted_data = self._parse_json_response(llm_response) or {}
        extracted_data = sanitize_submodel_data("OperationalData", extracted_data)

        print(f"  ✓ Extracted operational data")

        return extracted_data

    def _extract_bill_of_materials(self, pdf_slugs: List[str]) -> Dict:
        """
        Extract BillOfMaterials: components, part numbers, accessories.
        """
        print(f"  Processing {len(pdf_slugs)} PDF(s)...")

        # Semantic search
        search_results = self._semantic_search_all_pdfs(
            pdf_slugs,
            "component part number article number accessories options bill of materials BOM parts list",
            top_k=10
        )

        # Build context
        context_parts = []
        for result in search_results[:10]:
            context_parts.append(f"[{result['pdf_filename']}] {result['text']}")

        context = "\n\n".join(context_parts)

        template_json = json.dumps(template_schema("BillOfMaterials"), indent=2)

        # LLM extraction
        prompt = f"""Extract bill of materials information.

Context:
{context}

Template JSON (use exactly these keys, no additions):
```json
{template_json}
```

Instructions:
- Populate each component entry with the corresponding details found in the context.
- Use null when information is not confirmed.
- Do not add new keys or rename existing ones.
"""

        llm_response = self._query_llm(prompt, label="BillOfMaterials")
        extracted_data = self._parse_json_response(llm_response) or {}
        extracted_data = sanitize_submodel_data("BillOfMaterials", extracted_data)

        print(f"  ✓ Extracted {len(extracted_data.get('Components', []))} components")

        return extracted_data

    def _extract_carbon_footprint(self, pdf_slugs: List[str]) -> Dict:
        """
        Extract CarbonFootprint: environmental data, lifecycle, energy consumption.
        """
        print(f"  Processing {len(pdf_slugs)} PDF(s)...")

        # Semantic search
        search_results = self._semantic_search_all_pdfs(
            pdf_slugs,
            "environmental carbon CO2 energy consumption efficiency lifecycle sustainability recycling disposal eco",
            top_k=10
        )

        # Build context
        context_parts = []
        for result in search_results[:10]:
            context_parts.append(f"[{result['pdf_filename']}] {result['text']}")

        context = "\n\n".join(context_parts)

        template_json = json.dumps(template_schema("CarbonFootprint"), indent=2)

        # LLM extraction
        prompt = f"""Extract environmental and carbon footprint information.

Context:
{context}

Template JSON (use exactly these keys, no additions):
```json
{template_json}
```

Instructions:
- Populate each field with concise, evidence-backed text (include units where relevant).
- Use null when information is not confirmed.
- Do not add new keys or rename existing ones.
"""

        llm_response = self._query_llm(prompt, label="CarbonFootprint")
        extracted_data = self._parse_json_response(llm_response) or {}
        extracted_data = sanitize_submodel_data("CarbonFootprint", extracted_data)

        print(f"  ✓ Extracted environmental data")

        return extracted_data

    def _semantic_search_all_pdfs(self, pdf_slugs: List[str], query: str, top_k: int = 5) -> List[Dict]:
        """
        Perform semantic search across specified PDFs.
        """
        if not hasattr(self.storage, 'milvus_client') or not self.storage.milvus_client:
            return []

        # Encode query
        from pdfkg.embeds import get_sentence_transformer
        model = get_sentence_transformer("sentence-transformers/all-MiniLM-L6-v2")
        query_embedding = model.encode([query])[0]

        # Global search
        search_results = self.storage.milvus_client.search_global(
            query_embedding=query_embedding,
            top_k=top_k * len(pdf_slugs)  # Get more results to filter
        )

        # Filter to target PDFs and fetch chunks
        results = []
        for search_result in search_results:
            if search_result['pdf_slug'] in pdf_slugs:
                # Fetch chunk text
                chunks = self.storage.get_chunks(search_result['pdf_slug'])
                chunk_index = search_result['chunk_index']

                if chunk_index < len(chunks):
                    chunk = chunks[chunk_index]
                    pdf_info = self.storage.get_pdf_metadata(search_result['pdf_slug'])

                    results.append({
                        'pdf_slug': search_result['pdf_slug'],
                        'pdf_filename': pdf_info['filename'],
                        'text': chunk['text'],
                        'score': search_result['distance']
                    })

                if len(results) >= top_k:
                    break

        return results

    def _query_llm(self, prompt: str, *, label: str) -> str:
        """Query the LLM and return response text."""
        if self.llm_provider == "gemini":
            start = time.time()
            response = self.llm_client.generate_content(prompt)
            usage = getattr(response, "usage_metadata", None)
            tokens_in, tokens_out, total_tokens = llm_stats.extract_token_usage(usage)
            llm_stats.record_call(
                self.llm_provider,
                phase='extraction',
                label=label,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                total_tokens=total_tokens,
                metadata={
                    "model": getattr(self.llm_client, "model_name", getattr(self, "gemini_model", os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))),
                    "elapsed_ms": int((time.time() - start) * 1000),
                    "prompt_chars": len(prompt),
                },
            )
            return response.text

        elif self.llm_provider == "mistral":
            start = time.time()
            response = mistral_chat(
                messages=[{"role": "user", "content": prompt}],
                model=self.mistral_model,
            )
            usage = getattr(response, "usage", None)
            tokens_in, tokens_out, total_tokens = llm_stats.extract_token_usage(usage)
            llm_stats.record_call(
                self.llm_provider,
                phase='extraction',
                label=label,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                total_tokens=total_tokens,
                metadata={
                    "model": self.mistral_model,
                    "elapsed_ms": int((time.time() - start) * 1000),
                    "prompt_chars": len(prompt),
                },
            )
            content = response.choices[0].message.content
            if isinstance(content, list):
                content = "\n".join(str(item) for item in content)
            return str(content)

    def _parse_json_response(self, response_text: str) -> Dict:
        """Parse LLM JSON response."""
        try:
            # Remove markdown code blocks if present
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]

            result = json.loads(response_text.strip())
            return result

        except json.JSONDecodeError as e:
            print(f"  ⚠️  Failed to parse LLM JSON response: {e}")
            print(f"  Raw response: {response_text[:200]}")
            return {}


def extract_aas_data(storage, llm_provider: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract AAS submodel data from all classified PDFs.

    Args:
        storage: Storage backend
        llm_provider: LLM provider ("gemini" or "mistral")

    Returns:
        Dict with extracted data organized by submodel
    """
    extractor = AASDataExtractor(storage, llm_provider=llm_provider)
    return extractor.extract_all_submodel_data()
