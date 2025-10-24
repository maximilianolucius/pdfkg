"""
AAS (Asset Administration Shell) Classifier

Classifies PDFs to AAS submodels using LLM analysis.

AAS v5.0 Submodels:
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
from typing import Dict, List, Optional

from pdfkg import llm_stats
from pdfkg.submodel_templates import get_template, list_submodel_templates

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


# AAS Submodel definitions


def _flatten_template_fields(schema, prefix='') -> List[str]:
    fields: List[str] = []
    if isinstance(schema, dict):
        for key, value in schema.items():
            new_prefix = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                fields.extend(_flatten_template_fields(value, new_prefix))
            elif isinstance(value, list):
                if value and isinstance(value[0], dict):
                    fields.extend(_flatten_template_fields(value[0], new_prefix))
                else:
                    fields.append(new_prefix)
            else:
                fields.append(new_prefix)
    elif isinstance(schema, list):
        if schema and isinstance(schema[0], dict):
            fields.extend(_flatten_template_fields(schema[0], prefix))
        else:
            fields.append(prefix)
    return fields


class AASClassifier:
    """
    Classifier for mapping PDFs to AAS submodels using LLM analysis.
    """

    def __init__(self, storage, llm_provider: str = "gemini"):
        """
        Initialize AAS Classifier.

        Args:
            storage: Storage backend (ArangoStorage or MilvusArangoStorage)
            llm_provider: LLM provider ("gemini" or "mistral")
        """
        self.storage = storage
        self.llm_provider = llm_provider.lower()

        # Initialize LLM client
        if self.llm_provider == "gemini":
            if not GEMINI_AVAILABLE:
                raise ImportError("google-generativeai not installed. Install with: pip install google-generativeai")
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment")
            genai.configure(api_key=api_key)
            model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            self.llm_client = genai.GenerativeModel(model_name)
            print(f"✅ Initialized Gemini model: {model_name}")

        elif self.llm_provider == "mistral":
            if not MISTRAL_AVAILABLE:
                raise ImportError("mistralai not installed. Install with: pip install mistralai")
            api_key = os.getenv("MISTRAL_API_KEY")
            if not api_key:
                raise ValueError("MISTRAL_API_KEY not found in environment")
            self.llm_client = Mistral(api_key=api_key)
            self.mistral_model = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
            print(f"✅ Initialized Mistral model: {self.mistral_model}")

        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}. Use 'gemini' or 'mistral'")

        self.submodel_keys = list_submodel_templates()

    def classify_pdf(self, pdf_slug: str) -> Dict:
        """
        Classify a single PDF to AAS submodels.

        Args:
            pdf_slug: PDF identifier

        Returns:
            Dict with classification results:
            {
                "pdf_slug": str,
                "filename": str,
                "submodels": [str],
                "confidence_scores": {submodel: float},
                "reasoning": str
            }
        """
        print(f"\n📄 Classifying: {pdf_slug}")

        # Get PDF metadata
        pdf_info = self.storage.get_pdf_metadata(pdf_slug)

        # Get TOC
        toc = self.storage.get_toc(pdf_slug)

        # Get entities (if available)
        entities_data = self.storage.db_client.get_metadata(pdf_slug, 'extracted_entities')

        # Get first few chunks for context
        chunks = self.storage.get_chunks(pdf_slug)
        sample_chunks = chunks[:5] if len(chunks) > 5 else chunks

        # Build classification prompt
        prompt = self._build_classification_prompt(pdf_info, toc, entities_data, sample_chunks)

        # Query LLM
        llm_response = self._query_llm(prompt)

        # Parse response
        classification = self._parse_llm_response(llm_response, pdf_slug, pdf_info['filename'])

        print(f"  ✓ Classified to: {', '.join(classification['submodels'])}")

        return classification

    def classify_all_pdfs(self) -> Dict[str, Dict]:
        """
        Classify all PDFs in the database to AAS submodels.

        Returns:
            Dict mapping pdf_slug to classification results
        """
        print("\n" + "=" * 80)
        print("AAS CLASSIFICATION: Mapping PDFs to AAS Submodels")
        print("=" * 80)

        all_pdfs = self.storage.list_pdfs()

        if not all_pdfs:
            print("\n⚠️  No PDFs found in database")
            return {}

        print(f"\n📚 Found {len(all_pdfs)} PDFs to classify")
        print(f"🤖 Using LLM: {self.llm_provider}")

        classifications = {}

        for i, pdf in enumerate(all_pdfs, 1):
            slug = pdf['slug']
            print(f"\n[{i}/{len(all_pdfs)}] Processing: {pdf['filename']}")

            try:
                classification = self.classify_pdf(slug)
                classifications[slug] = classification

            except Exception as e:
                print(f"  ❌ Error classifying {slug}: {e}")
                classifications[slug] = {
                    "pdf_slug": slug,
                    "filename": pdf['filename'],
                    "submodels": [],
                    "confidence_scores": {},
                    "reasoning": f"Error: {str(e)}",
                    "error": str(e)
                }

        # Save classifications to storage
        self.storage.db_client.save_metadata('__global__', 'aas_classifications', classifications)
        print(f"\n✅ Saved classifications for {len(classifications)} PDFs")

        # Print summary
        self._print_classification_summary(classifications)

        return classifications

    def _build_classification_prompt(
        self,
        pdf_info: Dict,
        toc: List[Dict],
        entities_data: Optional[Dict],
        sample_chunks: List[Dict]
    ) -> str:
        """Build LLM prompt for PDF classification."""

        # Extract entity summary
        entity_summary = ""
        if entities_data:
            entity_counts = {}
            for chunk_id, entities in entities_data.items():
                for entity in entities:
                    entity_type = entity.get('type', 'unknown')
                    entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1

            if entity_counts:
                entity_summary = "Named entities found:\n"
                for entity_type, count in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True):
                    entity_summary += f"  - {entity_type}: {count}\n"

        # Build TOC summary
        toc_summary = ""
        if toc:
            toc_summary = "Table of Contents:\n"
            for item in toc[:15]:  # First 15 items
                level = "  " * item.get('level', 0)
                toc_summary += f"{level}- {item.get('title', 'Untitled')}\n"

        # Build sample text
        sample_text = ""
        if sample_chunks:
            sample_text = "Sample content from document:\n"
            for chunk in sample_chunks[:3]:
                text_preview = chunk['text'][:200].replace('\n', ' ')
                sample_text += f"  - {text_preview}...\n"

                submodel_desc = "AAS Submodels:\n"
        for i, key in enumerate(self.submodel_keys, 1):
            template = get_template(key)
            fields = _flatten_template_fields(template.schema)
            preview = ', '.join(fields[:8])
            if len(fields) > 8:
                preview += ', ...'
            submodel_desc += f"{i}. {template.display_name} (key: {key})\n"
            submodel_desc += f"   Template fields: {preview}\n\n"

        allowed_keys = ', '.join(self.submodel_keys)

        prompt = f"""You are an expert in Asset Administration Shell (AAS) v5.0 classification.

Analyze this PDF document and decide which submodel templates apply.

=== DOCUMENT INFORMATION ===
Filename: {pdf_info.get('filename', 'Unknown')}
Pages: {pdf_info.get('num_pages', 0)}
Sections: {pdf_info.get('num_sections', 0)}

{toc_summary}

{entity_summary}

{sample_text}

=== AVAILABLE SUBMODELS ===
{submodel_desc}

=== TASK ===
Using the information above, determine which submodel templates (by key) have sufficient information in this document.
Return JSON using this format and only the keys listed above:
{{
  "submodels": ["DigitalNameplate", "TechnicalData"],
  "confidence_scores": {{
    "DigitalNameplate": 0.95,
    "TechnicalData": 0.78
  }},
  "reasoning": "Brief explanation referencing the document context"
}}

Valid submodel keys: {allowed_keys}.
Only include submodels with confidence >= 0.5.
"""

        return prompt

    def _query_llm(self, prompt: str) -> str:
        """Query the LLM and return response text."""

        if self.llm_provider == "gemini":
            response = self.llm_client.generate_content(prompt)
            llm_stats.record_call(self.llm_provider, 'classification', 'global')
            return response.text

        elif self.llm_provider == "mistral":
            response = self.llm_client.chat.complete(
                model=self.mistral_model,
                messages=[{"role": "user", "content": prompt}]
            )
            llm_stats.record_call(self.llm_provider, 'classification', 'global')
            return response.choices[0].message.content

    def _parse_llm_response(self, response_text: str, pdf_slug: str, filename: str) -> Dict:
        """Parse LLM JSON response."""

        # Try to extract JSON from response
        try:
            # Remove markdown code blocks if present
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]

            result = json.loads(response_text.strip())

            # Validate structure
            if "submodels" not in result or "confidence_scores" not in result:
                raise ValueError("Missing required fields in LLM response")

            filtered_submodels = [key for key in result.get('submodels', []) if key in self.submodel_keys]
            result['submodels'] = filtered_submodels
            result['confidence_scores'] = {k: v for k, v in result.get('confidence_scores', {}).items() if k in self.submodel_keys}

            # Add metadata
            result["pdf_slug"] = pdf_slug
            result["filename"] = filename

            return result

        except json.JSONDecodeError as e:
            print(f"  ⚠️  Failed to parse LLM response as JSON: {e}")
            print(f"  Raw response: {response_text[:200]}")

            # Return fallback
            return {
                "pdf_slug": pdf_slug,
                "filename": filename,
                "submodels": ["Documentation"],  # Default fallback
                "confidence_scores": {"Documentation": 0.5},
                "reasoning": "Failed to parse LLM response, defaulted to Documentation",
                "raw_response": response_text
            }

    def _print_classification_summary(self, classifications: Dict[str, Dict]) -> None:
        """Print summary of classifications."""

        print("\n" + "=" * 80)
        print("CLASSIFICATION SUMMARY")
        print("=" * 80)

        # Count submodels
        submodel_counts = {}
        for classification in classifications.values():
            for submodel in classification.get('submodels', []):
                submodel_counts[submodel] = submodel_counts.get(submodel, 0) + 1

        print("\n📊 Submodel Distribution:")
        for submodel, count in sorted(submodel_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   {submodel}: {count} PDF(s)")

        print("\n📄 PDF → Submodel Mapping:")
        for slug, classification in classifications.items():
            if 'error' in classification:
                print(f"\n   ❌ {classification['filename']}")
                print(f"      Error: {classification['error']}")
            else:
                print(f"\n   ✓ {classification['filename']}")
                for submodel in classification.get('submodels', []):
                    confidence = classification.get('confidence_scores', {}).get(submodel, 0)
                    print(f"      - {submodel} (confidence: {confidence:.2f})")

        print("\n" + "=" * 80)


def classify_pdfs_to_aas(storage, llm_provider: str = "gemini") -> Dict[str, Dict]:
    """
    Classify all PDFs in storage to AAS submodels.

    Args:
        storage: Storage backend
        llm_provider: LLM provider ("gemini" or "mistral")

    Returns:
        Dict mapping pdf_slug to classification results
    """
    classifier = AASClassifier(storage, llm_provider=llm_provider)
    return classifier.classify_all_pdfs()


def classify_single_pdf_submodels(storage, pdf_slug: str, llm_provider: str = "gemini") -> Optional[Dict]:
    """
    Classify a single PDF to AAS submodels without instantiating the full class.
    This is a lightweight version for use during ingestion.

    Args:
        storage: Storage backend
        pdf_slug: The slug of the PDF to classify
        llm_provider: LLM provider ("gemini" or "mistral")

    Returns:
        A dictionary with classification results, or None on failure.
    """
    try:
        classifier = AASClassifier(storage, llm_provider=llm_provider)
        return classifier.classify_pdf(pdf_slug)
    except Exception as e:
        print(f"❌ Error during single PDF classification for {pdf_slug}: {e}")
        return None
