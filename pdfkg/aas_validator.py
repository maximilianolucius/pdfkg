"""
AAS Validator - Phase 3

Validate extracted AAS data and complete missing information.

Validation checks:
1. Are all mandatory submodels present?
2. Are mandatory properties filled?
3. Are semantic IDs correct?
4. Are references valid?

If incomplete:
- Perform additional semantic searches
- Re-extract missing data with targeted queries
- Provide suggestions for manual completion
"""

import json
import os
from typing import Dict, List, Optional, Any, Tuple

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


# AAS Mandatory submodels (minimal required)
REQUIRED_SUBMODELS = [
    "DigitalNameplate",
    "TechnicalData",
    "Documentation"
]

# Mandatory fields per submodel
MANDATORY_FIELDS = {
    "DigitalNameplate": [
        "ManufacturerName",
        "ManufacturerProductDesignation"
    ],
    "TechnicalData": [
        "GeneralTechnicalData"
    ],
    "Documentation": [
        "Documents"
    ]
}


class AASValidator:
    """
    Validate and complete AAS extracted data.
    """

    def __init__(self, storage, llm_provider: str = "gemini"):
        """
        Initialize AAS Validator.

        Args:
            storage: Storage backend
            llm_provider: LLM provider ("gemini" or "mistral")
        """
        self.storage = storage
        self.llm_provider = llm_provider.lower()

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
            print(f"✅ Initialized Gemini model: {model_name}")

        elif self.llm_provider == "mistral":
            if not MISTRAL_AVAILABLE:
                raise ImportError("mistralai not installed")
            api_key = os.getenv("MISTRAL_API_KEY")
            if not api_key:
                raise ValueError("MISTRAL_API_KEY not found in environment")
            self.llm_client = Mistral(api_key=api_key)
            self.mistral_model = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
            print(f"✅ Initialized Mistral model: {self.mistral_model}")

        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")

    def validate_and_complete(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Validate extracted AAS data and attempt to complete missing information.

        Returns:
            Tuple of (completed_data, validation_report)
        """
        print("\n" + "=" * 80)
        print("AAS VALIDATION & COMPLETION - Phase 3")
        print("=" * 80)

        # Load extracted data from Phase 2
        extracted_data = self.storage.db_client.get_metadata('__global__', 'aas_extracted_data')
        if not extracted_data:
            print("\n❌ No extracted data found. Run Phase 2 first (--extract-aas)")
            return {}, {}

        print(f"\n📚 Found data for {len(extracted_data)} submodels")

        # Validate using LLM
        validation_result = self._validate_with_llm(extracted_data)

        # Check completeness
        is_complete = validation_result.get('complete', False)
        missing = validation_result.get('missing', [])
        suggestions = validation_result.get('suggestions', [])

        print(f"\n{'✅' if is_complete else '⚠️ '} Completeness: {'Complete' if is_complete else 'Incomplete'}")

        if missing:
            print(f"\n⚠️  Missing data detected: {len(missing)} item(s)")
            for item in missing:
                print(f"   - {item}")

        # Attempt to complete missing data
        completed_data = extracted_data.copy()
        completion_attempts = []

        if not is_complete and missing:
            print("\n🔄 Attempting to complete missing data...")

            for missing_item in missing:
                attempt_result = self._attempt_completion(missing_item, extracted_data)
                if attempt_result:
                    completion_attempts.append(attempt_result)

                    # Merge completed data
                    submodel = attempt_result.get('submodel')
                    data = attempt_result.get('data')
                    if submodel and data:
                        if submodel not in completed_data:
                            completed_data[submodel] = {}
                        completed_data[submodel].update(data)

        # Re-validate completed data
        if completion_attempts:
            print("\n🔍 Re-validating completed data...")
            final_validation = self._validate_with_llm(completed_data)
            is_complete = final_validation.get('complete', False)
            missing = final_validation.get('missing', [])
            suggestions = final_validation.get('suggestions', [])

        # Build validation report
        validation_report = {
            "is_complete": is_complete,
            "missing_items": missing,
            "suggestions": suggestions,
            "completion_attempts": len(completion_attempts),
            "original_submodels": list(extracted_data.keys()),
            "completed_submodels": list(completed_data.keys()),
            "mandatory_submodels_present": self._check_mandatory_submodels(completed_data),
            "mandatory_fields_complete": self._check_mandatory_fields(completed_data)
        }

        # Save completed data
        self.storage.db_client.save_metadata('__global__', 'aas_validated_data', completed_data)
        self.storage.db_client.save_metadata('__global__', 'aas_validation_report', validation_report)

        print(f"\n✅ Validation complete!")
        print(f"   - Original submodels: {len(extracted_data)}")
        print(f"   - Completed submodels: {len(completed_data)}")
        print(f"   - Completion attempts: {len(completion_attempts)}")
        print(f"   - Status: {'✅ Complete' if is_complete else '⚠️  Incomplete'}")

        if not is_complete and suggestions:
            print(f"\n💡 Suggestions for improvement:")
            for i, suggestion in enumerate(suggestions[:5], 1):
                print(f"   {i}. {suggestion}")

        print("\n" + "=" * 80)
        print("✓ Validation and completion finished!")
        print("=" * 80)

        return completed_data, validation_report

    def _validate_with_llm(self, data: Dict[str, Any]) -> Dict:
        """
        Use LLM to validate AAS data completeness and correctness.
        """
        # Build validation prompt
        data_summary = json.dumps(data, indent=2)

        prompt = f"""Review this extracted AAS (Asset Administration Shell) data for completeness and correctness.

Extracted Data:
```json
{data_summary}
```

Required Submodels (mandatory):
{', '.join(REQUIRED_SUBMODELS)}

Mandatory Fields:
{json.dumps(MANDATORY_FIELDS, indent=2)}

Check:
1. Are all mandatory submodels present? ({', '.join(REQUIRED_SUBMODELS)})
2. Are mandatory fields filled with valid data (not null, not empty)?
3. Are the data formats appropriate (e.g., voltage includes unit, dates are valid)?
4. Are there any obvious errors or inconsistencies?

Return ONLY valid JSON in this exact format:
{{
  "complete": true/false,
  "missing": [
    "List of missing mandatory submodels or fields",
    "Example: 'DigitalNameplate.ManufacturerName is null'",
    "Example: 'TechnicalData submodel is missing'"
  ],
  "suggestions": [
    "Suggestions for improvement or data that should be added",
    "Example: 'Add SerialNumber to DigitalNameplate'",
    "Example: 'TechnicalData should include electrical specifications'"
  ],
  "errors": [
    "Any data format errors or inconsistencies found"
  ]
}}

Be strict but practical. Mark as incomplete if mandatory data is missing.
"""

        llm_response = self._query_llm(prompt)
        validation = self._parse_json_response(llm_response)

        # Ensure structure
        if not validation:
            validation = {
                "complete": False,
                "missing": ["Failed to validate - LLM response parsing error"],
                "suggestions": [],
                "errors": []
            }

        return validation

    def _check_mandatory_submodels(self, data: Dict) -> bool:
        """Check if all mandatory submodels are present."""
        present_submodels = set(data.keys())
        required_submodels = set(REQUIRED_SUBMODELS)
        return required_submodels.issubset(present_submodels)

    def _check_mandatory_fields(self, data: Dict) -> Dict[str, bool]:
        """Check if mandatory fields are present and non-null for each submodel."""
        field_status = {}

        for submodel, required_fields in MANDATORY_FIELDS.items():
            if submodel not in data:
                field_status[submodel] = False
                continue

            submodel_data = data[submodel]
            all_present = True

            for field in required_fields:
                value = submodel_data.get(field)
                if value is None or value == "" or value == {}:
                    all_present = False
                    break

            field_status[submodel] = all_present

        return field_status

    def _attempt_completion(self, missing_item: str, current_data: Dict) -> Optional[Dict]:
        """
        Attempt to complete a missing data item using semantic search and LLM extraction.

        Args:
            missing_item: Description of missing item (e.g., "DigitalNameplate.SerialNumber")
            current_data: Current extracted data

        Returns:
            Dict with completion attempt results or None
        """
        print(f"\n  🔍 Attempting to complete: {missing_item}")

        # Parse missing item to identify submodel and field
        submodel, field = self._parse_missing_item(missing_item)

        if not submodel:
            print(f"    ⚠️  Could not parse missing item, skipping")
            return None

        # Get PDFs associated with this submodel
        classifications = self.storage.db_client.get_metadata('__global__', 'aas_classifications')
        if not classifications:
            return None

        pdf_slugs = []
        for slug, classification in classifications.items():
            if submodel in classification.get('submodels', []):
                pdf_slugs.append(slug)

        if not pdf_slugs:
            print(f"    ⚠️  No PDFs classified for {submodel}")
            return None

        # Perform targeted semantic search
        search_query = self._build_search_query_for_missing(submodel, field)
        search_results = self._semantic_search(pdf_slugs, search_query, top_k=5)

        if not search_results:
            print(f"    ⚠️  No relevant results found")
            return None

        # Build context
        context_parts = []
        for result in search_results:
            context_parts.append(f"[{result['pdf_filename']}] {result['text']}")

        context = "\n\n".join(context_parts)

        # Extract missing data with LLM
        extraction_prompt = self._build_extraction_prompt(submodel, field, context)
        llm_response = self._query_llm(extraction_prompt)
        extracted = self._parse_json_response(llm_response)

        if extracted:
            print(f"    ✅ Completed: {field}")
            return {
                "submodel": submodel,
                "field": field,
                "data": extracted,
                "source": "semantic_search + llm"
            }
        else:
            print(f"    ⚠️  Extraction failed")
            return None

    def _parse_missing_item(self, missing_item: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse missing item string to extract submodel and field.

        Examples:
        - "DigitalNameplate.ManufacturerName" -> ("DigitalNameplate", "ManufacturerName")
        - "TechnicalData submodel is missing" -> ("TechnicalData", None)
        """
        # Try dot notation first
        if '.' in missing_item:
            parts = missing_item.split('.')
            return parts[0].strip(), parts[1].strip()

        # Try to extract submodel name
        for submodel in REQUIRED_SUBMODELS:
            if submodel in missing_item:
                return submodel, None

        return None, None

    def _build_search_query_for_missing(self, submodel: str, field: Optional[str]) -> str:
        """Build semantic search query for missing data."""
        queries = {
            "DigitalNameplate": "manufacturer name product designation model serial number year construction nameplate identification",
            "TechnicalData": "technical specifications voltage current power dimensions weight temperature pressure",
            "Documentation": "manual datasheet documentation reference",
            "HandoverDocumentation": "certificate warranty compliance declaration conformity",
            "MaintenanceRecord": "maintenance service spare parts inspection",
            "OperationalData": "operating conditions parameters settings",
            "BillOfMaterials": "components parts bill materials accessories",
            "CarbonFootprint": "environmental energy consumption carbon emissions"
        }

        base_query = queries.get(submodel, submodel)

        if field:
            # Add field-specific keywords
            base_query += f" {field}"

        return base_query

    def _build_extraction_prompt(self, submodel: str, field: Optional[str], context: str) -> str:
        """Build LLM prompt for extracting missing data."""
        prompts = {
            "DigitalNameplate": """Extract identification information from this context.

Context:
{context}

Return ONLY valid JSON:
{{
  "ManufacturerName": "Company name if found, otherwise null",
  "ManufacturerProductDesignation": "Product model if found, otherwise null",
  "SerialNumber": "Serial number if found, otherwise null",
  "YearOfConstruction": "Year as integer if found, otherwise null"
}}
""",
            "TechnicalData": """Extract technical specifications from this context.

Context:
{context}

Return ONLY valid JSON:
{{
  "GeneralTechnicalData": {{
    "Weight": "Weight with unit, otherwise null",
    "Dimensions": "Dimensions with units, otherwise null"
  }},
  "ElectricalProperties": {{
    "VoltageRange": "Voltage range, otherwise null",
    "Current": "Current rating, otherwise null"
  }}
}}
"""
        }

        template = prompts.get(submodel, """Extract relevant data for {submodel}.

Context:
{context}

Return ONLY valid JSON with any relevant data found.
""")

        return template.format(context=context, submodel=submodel)

    def _semantic_search(self, pdf_slugs: List[str], query: str, top_k: int = 5) -> List[Dict]:
        """Perform semantic search across specified PDFs."""
        if not hasattr(self.storage, 'milvus_client') or not self.storage.milvus_client:
            return []

        # Encode query
        from pdfkg.embeds import get_sentence_transformer
        model = get_sentence_transformer("sentence-transformers/all-MiniLM-L6-v2")
        query_embedding = model.encode([query])[0]

        # Global search
        search_results = self.storage.milvus_client.search_global(
            query_embedding=query_embedding,
            top_k=top_k * len(pdf_slugs)
        )

        # Filter to target PDFs
        results = []
        for search_result in search_results:
            if search_result['pdf_slug'] in pdf_slugs:
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

    def _query_llm(self, prompt: str) -> str:
        """Query the LLM and return response text."""
        if self.llm_provider == "gemini":
            response = self.llm_client.generate_content(prompt)
            return response.text

        elif self.llm_provider == "mistral":
            response = self.llm_client.chat.complete(
                model=self.mistral_model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content

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
            print(f"  ⚠️  Failed to parse LLM JSON: {e}")
            return {}


def validate_aas_data(storage, llm_provider: str = "gemini") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Validate and complete AAS extracted data.

    Args:
        storage: Storage backend
        llm_provider: LLM provider ("gemini" or "mistral")

    Returns:
        Tuple of (completed_data, validation_report)
    """
    validator = AASValidator(storage, llm_provider=llm_provider)
    return validator.validate_and_complete()
