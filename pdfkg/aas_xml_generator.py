"""
AAS XML Generator - Phase 4

Generate AAS v5.0 compliant XML from validated data.

Structure:
- AAS Environment wrapper
- Asset Administration Shell definition
- Submodels with semantic IDs
- Concept descriptions
"""

import json
import os
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from pdfkg import llm_stats

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


# AAS v5.0 Namespace
AAS_NS = "https://admin-shell.io/aas/5/0"

# Semantic IDs for submodels (IDTA standards)
SEMANTIC_IDS = {
    "DigitalNameplate": "https://admin-shell.io/ZVEI/TechnicalData/Nameplate/1/1",
    "TechnicalData": "https://admin-shell.io/ZVEI/TechnicalData/Submodel/1/1",
    "Documentation": "https://admin-shell.io/ZVEI/TechnicalData/Documentation/1/1",
    "HandoverDocumentation": "https://admin-shell.io/ZVEI/TechnicalData/HandoverDocumentation/1/1",
    "MaintenanceRecord": "https://admin-shell.io/idta/Maintenance/1/0",
    "OperationalData": "https://admin-shell.io/idta/OperationalData/1/0",
    "BillOfMaterials": "https://admin-shell.io/idta/BillOfMaterial/1/0",
    "CarbonFootprint": "https://admin-shell.io/idta/CarbonFootprint/1/0"
}


class AASXMLGenerator:
    """
    Generate AAS v5.0 XML from validated data.
    """

    def __init__(self, storage, llm_provider: str = "gemini"):
        """
        Initialize AAS XML Generator.

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

    def generate_xml(self, output_path: Optional[Path] = None, data: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate complete AAS XML file.

        Args:
            output_path: Optional path to save XML file

        Returns:
            XML string
        """
        print("\n" + "=" * 80)
        print("AAS XML GENERATION - Phase 4")
        print("=" * 80)

        if data is None:
            validated_data = self.storage.db_client.get_metadata('__global__', 'aas_validated_data')
            if not validated_data:
                validated_data = self.storage.db_client.get_metadata('__global__', 'aas_extracted_data')
            data = validated_data

        if not data:
            print("\n❌ No data available for XML generation. Provide data or run previous phases.")
            return ""

        print(f"\n📚 Generating XML for {len(data)} submodels")

        xml_string = self._build_xml(data)

        # Save to file
        if not output_path:
            output_path = Path("data/out/aas_output.xml")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(xml_string)

        # Save to storage
        self.storage.db_client.save_metadata('__global__', 'aas_generated_xml', {
            'xml': xml_string,
            'generated_at': datetime.now().isoformat(),
            'output_path': str(output_path)
        })

        print(f"\n✅ XML generated successfully!")
        print(f"   Output: {output_path}")
        print(f"   Size: {len(xml_string)} characters")

        print("\n" + "=" * 80)
        print("✓ AAS XML generation complete!")
        print("=" * 80)

        return xml_string

    def _build_xml(self, data: Dict[str, Any]) -> str:
        """Build complete AAS XML structure."""

        # Get basic info from DigitalNameplate
        nameplate = data.get('DigitalNameplate', {})
        manufacturer = nameplate.get('ManufacturerName', 'Unknown')
        product = nameplate.get('ManufacturerProductDesignation', 'Unknown')

        # Generate unique IDs
        aas_id = f"https://example.com/aas/{product.replace(' ', '_')}"
        asset_id = f"https://example.com/asset/{product.replace(' ', '_')}"

        # Start XML with proper namespaces
        xml_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<environment xmlns="https://admin-shell.io/aas/5/0">',
            '',
            '  <!-- Asset Administration Shells -->',
            '  <assetAdministrationShells>',
            '    <assetAdministrationShell>',
            f'      <id>{aas_id}</id>',
            '      <idShort>AAS_' + product.replace(' ', '_')[:50] + '</idShort>',
            '      <assetInformation>',
            f'        <assetKind>Instance</assetKind>',
            f'        <globalAssetId>{asset_id}</globalAssetId>',
            '      </assetInformation>',
            '      <submodels>',
        ]

        # Add submodel references
        for submodel_name in data.keys():
            submodel_id = f"https://example.com/submodel/{submodel_name}"
            xml_lines.extend([
                '        <reference>',
                '          <type>ExternalReference</type>',
                '          <keys>',
                '            <key>',
                '              <type>Submodel</type>',
                f'              <value>{submodel_id}</value>',
                '            </key>',
                '          </keys>',
                '        </reference>',
            ])

        xml_lines.extend([
            '      </submodels>',
            '    </assetAdministrationShell>',
            '  </assetAdministrationShells>',
            '',
            '  <!-- Submodels -->',
            '  <submodels>',
        ])

        # Generate each submodel
        for submodel_name, submodel_data in data.items():
            print(f"\n  🔨 Generating submodel: {submodel_name}")
            submodel_xml = self._generate_submodel_xml(submodel_name, submodel_data)
            xml_lines.append(submodel_xml)

        xml_lines.extend([
            '  </submodels>',
            '',
            '  <!-- Concept Descriptions -->',
            '  <conceptDescriptions>',
            '    <!-- Semantic definitions would go here -->',
            '  </conceptDescriptions>',
            '',
            '</environment>'
        ])

        return '\n'.join(xml_lines)

    def _generate_submodel_xml(self, submodel_name: str, submodel_data: Dict) -> str:
        """
        Generate XML for a single submodel using LLM.

        Args:
            submodel_name: Name of the submodel
            submodel_data: Data for the submodel

        Returns:
            XML string for the submodel
        """
        submodel_id = f"https://example.com/submodel/{submodel_name}"
        semantic_id = SEMANTIC_IDS.get(submodel_name, "https://example.com/semantic/" + submodel_name)

        # Build prompt for LLM
        data_json = json.dumps(submodel_data, indent=2)

        prompt = f"""Generate AAS v5.0 XML submodel for {submodel_name}.

Data to include:
```json
{data_json}
```

Submodel ID: {submodel_id}
Semantic ID: {semantic_id}

Generate ONLY the <submodel> element with proper AAS v5.0 structure:
- Use <id>{submodel_id}</id>
- Use <idShort>{submodel_name}</idShort>
- Use <kind>Instance</kind>
- Include semanticId with the provided semantic ID
- Create submodelElements with proper property types (string, integer, double)
- Use meaningful idShort names for properties
- Include valueType attributes (xs:string, xs:integer, xs:double, xs:boolean)

Example structure:
```xml
<submodel>
  <id>{{submodel_id}}</id>
  <idShort>{{submodel_name}}</idShort>
  <kind>Instance</kind>
  <semanticId>
    <type>ExternalReference</type>
    <keys>
      <key>
        <type>GlobalReference</type>
        <value>{{semantic_id}}</value>
      </key>
    </keys>
  </semanticId>
  <submodelElements>
    <property>
      <idShort>PropertyName</idShort>
      <valueType>xs:string</valueType>
      <value>Value from data</value>
    </property>
    <!-- More properties based on the data -->
  </submodelElements>
</submodel>
```

Generate the complete <submodel> element with ALL data from the JSON.
Only return the XML, no explanations.
"""

        llm_response = self._query_llm(prompt)
        llm_stats.record_call(self.llm_provider, 'xml_generation', submodel_name)

        # Clean and validate XML
        xml_content = self._extract_xml_from_response(llm_response)

        if not xml_content:
            # Fallback: generate basic structure
            xml_content = self._generate_fallback_submodel(submodel_name, submodel_id, semantic_id, submodel_data)

        return xml_content

    def _extract_xml_from_response(self, response: str) -> str:
        """Extract XML content from LLM response."""
        # Remove markdown code blocks
        if "```xml" in response:
            response = response.split("```xml")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]

        # Clean whitespace
        response = response.strip()

        # Verify it starts with <submodel>
        if not response.startswith('<submodel'):
            return ""

        return response

    def _generate_fallback_submodel(self, name: str, submodel_id: str, semantic_id: str, data: Dict) -> str:
        """Generate basic submodel XML as fallback."""
        lines = [
            '    <submodel>',
            f'      <id>{submodel_id}</id>',
            f'      <idShort>{name}</idShort>',
            '      <kind>Instance</kind>',
            '      <semanticId>',
            '        <type>ExternalReference</type>',
            '        <keys>',
            '          <key>',
            '            <type>GlobalReference</type>',
            f'            <value>{semantic_id}</value>',
            '          </key>',
            '        </keys>',
            '      </semanticId>',
            '      <submodelElements>',
        ]

        # Add properties from data
        for key, value in data.items():
            if isinstance(value, dict):
                # Nested structure - flatten
                for nested_key, nested_value in value.items():
                    if nested_value is not None and not isinstance(nested_value, (list, dict)):
                        value_type = self._infer_value_type(nested_value)
                        lines.extend([
                            '        <property>',
                            f'          <idShort>{nested_key}</idShort>',
                            f'          <valueType>{value_type}</valueType>',
                            f'          <value>{self._escape_xml(str(nested_value))}</value>',
                            '        </property>',
                        ])
            elif isinstance(value, list):
                # Skip lists in fallback
                continue
            elif value is not None:
                value_type = self._infer_value_type(value)
                lines.extend([
                    '        <property>',
                    f'          <idShort>{key}</idShort>',
                    f'          <valueType>{value_type}</valueType>',
                    f'          <value>{self._escape_xml(str(value))}</value>',
                    '        </property>',
                ])

        lines.extend([
            '      </submodelElements>',
            '    </submodel>',
        ])

        return '\n'.join(lines)

    def _infer_value_type(self, value: Any) -> str:
        """Infer XSD value type from Python value."""
        if isinstance(value, bool):
            return "xs:boolean"
        elif isinstance(value, int):
            return "xs:integer"
        elif isinstance(value, float):
            return "xs:double"
        else:
            return "xs:string"

    def _escape_xml(self, text: str) -> str:
        """Escape XML special characters."""
        return (text
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
                .replace("'", "&apos;"))

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


def generate_aas_xml(storage, llm_provider: str = "gemini", output_path: Optional[Path] = None, data: Optional[Dict[str, Any]] = None) -> str:
    """
    Generate AAS v5.0 XML from validated data.

    Args:
        storage: Storage backend
        llm_provider: LLM provider ("gemini" or "mistral")
        output_path: Optional path to save XML file

    Returns:
        XML string
    """
    generator = AASXMLGenerator(storage, llm_provider=llm_provider)
    return generator.generate_xml(output_path=output_path, data=data)
