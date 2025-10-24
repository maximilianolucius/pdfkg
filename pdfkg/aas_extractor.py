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
from typing import Dict, List, Optional, Any
from collections import defaultdict


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

    def __init__(self, storage, llm_provider: str = "gemini"):
        """
        Initialize AAS Data Extractor.

        Args:
            storage: Storage backend (with Milvus and ArangoDB)
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

        # 2.1: DigitalNameplate
        if "DigitalNameplate" in pdfs_by_submodel:
            print("\n" + "-" * 80)
            print("Extracting: DigitalNameplate")
            print("-" * 80)
            submodel_data["DigitalNameplate"] = self._extract_digital_nameplate(
                pdfs_by_submodel["DigitalNameplate"]
            )

        # 2.2: TechnicalData
        if "TechnicalData" in pdfs_by_submodel:
            print("\n" + "-" * 80)
            print("Extracting: TechnicalData")
            print("-" * 80)
            submodel_data["TechnicalData"] = self._extract_technical_data(
                pdfs_by_submodel["TechnicalData"]
            )

        # 2.3: Documentation
        if "Documentation" in pdfs_by_submodel:
            print("\n" + "-" * 80)
            print("Extracting: Documentation")
            print("-" * 80)
            submodel_data["Documentation"] = self._extract_documentation(
                pdfs_by_submodel["Documentation"]
            )

        # 2.4: HandoverDocumentation
        if "HandoverDocumentation" in pdfs_by_submodel:
            print("\n" + "-" * 80)
            print("Extracting: HandoverDocumentation")
            print("-" * 80)
            submodel_data["HandoverDocumentation"] = self._extract_handover_documentation(
                pdfs_by_submodel["HandoverDocumentation"]
            )

        # 2.5: MaintenanceRecord
        if "MaintenanceRecord" in pdfs_by_submodel:
            print("\n" + "-" * 80)
            print("Extracting: MaintenanceRecord")
            print("-" * 80)
            submodel_data["MaintenanceRecord"] = self._extract_maintenance_record(
                pdfs_by_submodel["MaintenanceRecord"]
            )

        # 2.6: OperationalData
        if "OperationalData" in pdfs_by_submodel:
            print("\n" + "-" * 80)
            print("Extracting: OperationalData")
            print("-" * 80)
            submodel_data["OperationalData"] = self._extract_operational_data(
                pdfs_by_submodel["OperationalData"]
            )

        # 2.7: BillOfMaterials
        if "BillOfMaterials" in pdfs_by_submodel:
            print("\n" + "-" * 80)
            print("Extracting: BillOfMaterials")
            print("-" * 80)
            submodel_data["BillOfMaterials"] = self._extract_bill_of_materials(
                pdfs_by_submodel["BillOfMaterials"]
            )

        # 2.8: CarbonFootprint
        if "CarbonFootprint" in pdfs_by_submodel:
            print("\n" + "-" * 80)
            print("Extracting: CarbonFootprint")
            print("-" * 80)
            submodel_data["CarbonFootprint"] = self._extract_carbon_footprint(
                pdfs_by_submodel["CarbonFootprint"]
            )

        # Save to storage
        self.storage.db_client.save_metadata('__global__', 'aas_extracted_data', submodel_data)
        print(f"\n✅ Saved extracted data for {len(submodel_data)} submodels")

        print("\n" + "=" * 80)
        print("✓ Data extraction complete!")
        print("=" * 80)

        return submodel_data

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

        for slug, entity_dict in all_entities.items():
            for chunk_id, entities in entity_dict.items():
                for entity in entities:
                    entity_type = entity.get('type', '')
                    entity_text = entity.get('text', '')

                    if entity_type == 'manufacturer':
                        manufacturers.append(entity_text)
                    elif entity_type == 'model_number' or entity_type == 'product':
                        products.append(entity_text)
                    elif entity_type == 'serial_number':
                        serials.append(entity_text)
                    elif entity_type == 'year':
                        years.append(entity_text)

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

        # LLM extraction
        prompt = f"""Extract identification information for this industrial product.

Context from documents:
{context}

Entities found:
- Manufacturers: {_format_unique_examples(manufacturers, 5)}
- Products: {_format_unique_examples(products, 5)}
- Serial numbers: {_format_unique_examples(serials, 3)}
- Years: {_format_unique_examples(years, 3)}

Extract and return ONLY valid JSON in this exact format:
{{
  "ManufacturerName": "Company name",
  "ManufacturerProductDesignation": "Product model/designation",
  "SerialNumber": "Serial number if found, otherwise null",
  "YearOfConstruction": "Year as integer if found, otherwise null"
}}

Only include confirmed information. Use null for missing fields.
"""

        llm_response = self._query_llm(prompt)
        extracted_data = self._parse_json_response(llm_response)

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

        # LLM extraction
        prompt = f"""Extract technical specifications from these documents.

Context:
{context}

Return ONLY valid JSON in this exact format:
{{
  "GeneralTechnicalData": {{
    "ProductArticleNumber": "Article number if found, otherwise null",
    "Weight": "Weight with unit (e.g., '2.5 kg'), otherwise null",
    "Dimensions": "Dimensions with units (e.g., '100x50x30 mm'), otherwise null"
  }},
  "ElectricalProperties": {{
    "VoltageRange": "Voltage range (e.g., '100-240 VAC'), otherwise null",
    "Current": "Current rating (e.g., '10 A'), otherwise null",
    "Power": "Power rating (e.g., '2.4 kW'), otherwise null",
    "Frequency": "Frequency (e.g., '50/60 Hz'), otherwise null"
  }},
  "MechanicalProperties": {{
    "IPRating": "IP rating (e.g., 'IP65'), otherwise null",
    "TemperatureRange": "Operating temperature range (e.g., '-25 to +60°C'), otherwise null",
    "PressureRange": "Pressure range if applicable, otherwise null"
  }}
}}

Only include confirmed information. Use null for missing fields.
"""

        llm_response = self._query_llm(prompt)
        extracted_data = self._parse_json_response(llm_response)

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
                "DocumentType": doc_type,
                "FilePath": f"/documents/{pdf_info['filename']}",
                "ContentType": "application/pdf",
                "Language": language,
                "Pages": pdf_info['num_pages']
            }

            documents.append(doc_entry)

        print(f"  ✓ Mapped {len(documents)} documents")

        return {"Documents": documents}

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

        # LLM extraction
        prompt = f"""Extract certificate and compliance information.

Context:
{context}

Certifications found: {', '.join(all_certs) if all_certs else 'None'}

Return ONLY valid JSON in this exact format:
{{
  "Certifications": [
    {{
      "CertificationType": "Type (e.g., CE, UL, ATEX)",
      "Issuer": "Issuing organization if mentioned, otherwise null",
      "CertificateNumber": "Certificate number if found, otherwise null",
      "IssueDate": "Issue date if found, otherwise null"
    }}
  ],
  "Warranties": [
    {{
      "WarrantyPeriod": "Period (e.g., '2 years'), otherwise null",
      "WarrantyConditions": "Brief conditions if mentioned, otherwise null"
    }}
  ]
}}

Only include confirmed information. Return empty arrays if no data found.
"""

        llm_response = self._query_llm(prompt)
        extracted_data = self._parse_json_response(llm_response)

        print(f"  ✓ Extracted {len(extracted_data.get('Certifications', []))} certifications")

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

        # LLM extraction
        prompt = f"""Extract maintenance information.

Context:
{context}

Return ONLY valid JSON in this exact format:
{{
  "MaintenanceSchedule": {{
    "Interval": "Maintenance interval (e.g., 'every 6 months'), otherwise null",
    "Description": "Brief description of maintenance tasks, otherwise null"
  }},
  "SpareParts": [
    {{
      "PartName": "Part name",
      "PartNumber": "Part number if available, otherwise null"
    }}
  ]
}}

Only include confirmed information. Return empty arrays if no data found.
"""

        llm_response = self._query_llm(prompt)
        extracted_data = self._parse_json_response(llm_response)

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

        # LLM extraction
        prompt = f"""Extract operational data and parameters.

Context:
{context}

Return ONLY valid JSON in this exact format:
{{
  "OperatingConditions": {{
    "AmbientTemperature": "Ambient temperature range, otherwise null",
    "Humidity": "Humidity range if specified, otherwise null",
    "Altitude": "Maximum altitude if specified, otherwise null"
  }},
  "OperatingModes": [
    {{
      "ModeName": "Mode name",
      "Description": "Brief description"
    }}
  ]
}}

Only include confirmed information. Return empty arrays if no data found.
"""

        llm_response = self._query_llm(prompt)
        extracted_data = self._parse_json_response(llm_response)

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

        # LLM extraction
        prompt = f"""Extract bill of materials and component information.

Context:
{context}

Return ONLY valid JSON in this exact format:
{{
  "Components": [
    {{
      "ComponentName": "Component name",
      "PartNumber": "Part or article number",
      "Quantity": "Quantity if specified, otherwise null",
      "IsOptional": false
    }}
  ],
  "Accessories": [
    {{
      "AccessoryName": "Accessory name",
      "PartNumber": "Part number if available, otherwise null"
    }}
  ]
}}

Only include confirmed information. Return empty arrays if no data found.
"""

        llm_response = self._query_llm(prompt)
        extracted_data = self._parse_json_response(llm_response)

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

        # LLM extraction
        prompt = f"""Extract environmental and carbon footprint data.

Context:
{context}

Return ONLY valid JSON in this exact format:
{{
  "EnergyConsumption": {{
    "PowerRating": "Power rating (e.g., '2.4 kW'), otherwise null",
    "EnergyEfficiencyClass": "Efficiency class if specified, otherwise null"
  }},
  "Lifecycle": {{
    "ExpectedLifetime": "Expected lifetime if specified, otherwise null",
    "DisposalInstructions": "Brief disposal instructions if mentioned, otherwise null"
  }},
  "CarbonData": {{
    "CO2Emissions": "CO2 emissions data if available, otherwise null",
    "RecyclableContent": "Recyclable content percentage if specified, otherwise null"
  }}
}}

Only include confirmed information.
"""

        llm_response = self._query_llm(prompt)
        extracted_data = self._parse_json_response(llm_response)

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
            print(f"  ⚠️  Failed to parse LLM JSON response: {e}")
            print(f"  Raw response: {response_text[:200]}")
            return {}


def extract_aas_data(storage, llm_provider: str = "gemini") -> Dict[str, Any]:
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
