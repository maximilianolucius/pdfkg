"""
Named Entity Recognition (NER) for technical documents.

Extracts entities like products, models, standards, certifications, and technical specifications.
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Entity:
    """Represents an extracted entity."""
    text: str
    type: str
    start: int
    end: int
    confidence: float = 1.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# Technical entity patterns for common types in manuals
TECHNICAL_PATTERNS = {
    'model_number': [
        r'\b[A-Z]{2,4}[-_]?\d{3,6}[A-Z]?\b',  # ABC-1234A
        r'\bModel\s+([A-Z0-9\-]+)\b',
    ],
    'part_number': [
        r'\b(?:P/N|Part\s*#|PN)[:\s]*([A-Z0-9\-]+)\b',
        r'\bPart\s+Number[:\s]+([A-Z0-9\-]+)\b',
    ],
    'serial_number': [
        r'\b(?:S/N|Serial\s*#|SN)[:\s]*([A-Z0-9\-]+)\b',
    ],
    'ip_rating': [
        r'\bIP\d{2}\b',
    ],
    'standard': [
        r'\b(?:IEEE|ISO|IEC|EN|ANSI|DIN)\s*\d+(?:[.\-]\d+)*(?:[:\-][A-Z0-9]+)?\b',
        r'\bASTM\s+[A-Z]\d+\b',
    ],
    'voltage': [
        r'\b\d+(?:\.\d+)?\s*(?:V|VAC|VDC|kV|mV)\b',
        r'\b\d+(?:\.\d+)?\s*volts?\b',
    ],
    'current': [
        r'\b\d+(?:\.\d+)?\s*(?:A|mA|µA|Amps?)\b',
    ],
    'power': [
        r'\b\d+(?:\.\d+)?\s*(?:W|kW|MW|watts?)\b',
    ],
    'temperature': [
        r'\b-?\d+(?:\.\d+)?\s*°?\s*[CF](?:\b|(?=\s))',
        r'\b-?\d+(?:\.\d+)?\s*(?:degrees?\s+)?(?:Celsius|Fahrenheit)\b',
    ],
    'frequency': [
        r'\b\d+(?:\.\d+)?\s*(?:Hz|kHz|MHz|GHz)\b',
    ],
    'pressure': [
        r'\b\d+(?:\.\d+)?\s*(?:Pa|kPa|MPa|bar|psi|PSI)\b',
    ],
    'certification': [
        r'\b(?:CE|FCC|UL|RoHS|REACH|CSA|ETL|ATEX)\b',
    ],
    'firmware_version': [
        r'\b(?:v|version|fw|firmware)\s*\d+\.\d+(?:\.\d+)?(?:\.\d+)?\b',
        r'\bVersion\s+([0-9.]+)\b',
    ],
    'software_version': [
        r'\b(?:sw|software)\s*\d+\.\d+(?:\.\d+)?\b',
    ],
    'protocol': [
        r'\b(?:HTTP|HTTPS|FTP|SFTP|SSH|TCP|UDP|MQTT|Modbus|BACnet|SNMP)\b',
    ],
    'connector_type': [
        r'\b(?:RJ45|RJ11|USB|HDMI|DisplayPort|VGA|DVI)\b',
        r'\b(?:M12|M8)\s*connector\b',
    ],
    'cable_type': [
        r'\b(?:Cat5e|Cat6|Cat6A|Cat7)\b',
        r'\b(?:twisted pair|coaxial|fiber optic)\b',
    ],
}


class TechnicalNER:
    """Named Entity Recognition for technical documents."""

    def __init__(self):
        """Initialize NER with compiled patterns."""
        self.patterns = {}
        for entity_type, patterns in TECHNICAL_PATTERNS.items():
            self.patterns[entity_type] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]

    def extract_entities(self, text: str, chunk_id: Optional[str] = None) -> List[Entity]:
        """
        Extract entities from text using regex patterns.

        Args:
            text: Input text
            chunk_id: Optional chunk identifier for context

        Returns:
            List of Entity objects
        """
        entities = []

        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    # Get matched text (group 1 if exists, otherwise group 0)
                    entity_text = match.group(1) if match.lastindex else match.group(0)

                    entities.append(Entity(
                        text=entity_text.strip(),
                        type=entity_type,
                        start=match.start(),
                        end=match.end(),
                        confidence=1.0,  # Regex-based, high confidence
                        metadata={
                            'chunk_id': chunk_id,
                            'pattern_matched': pattern.pattern,
                            'full_match': match.group(0)
                        }
                    ))

        # Remove duplicates (same text at same position)
        entities = self._deduplicate_entities(entities)

        return entities

    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities."""
        seen = set()
        unique = []

        for entity in entities:
            key = (entity.text.lower(), entity.type, entity.start, entity.end)
            if key not in seen:
                seen.add(key)
                unique.append(entity)

        return unique

    def normalize_entity(self, entity: Entity) -> str:
        """
        Normalize entity text for canonical representation.

        Args:
            entity: Entity to normalize

        Returns:
            Normalized entity text
        """
        text = entity.text.strip()

        # Type-specific normalization
        if entity.type == 'model_number':
            # Remove extra spaces, uppercase
            text = re.sub(r'\s+', '', text).upper()
        elif entity.type in ['standard', 'certification']:
            # Uppercase
            text = text.upper()
        elif entity.type == 'ip_rating':
            # Uppercase
            text = text.upper()
        elif entity.type in ['voltage', 'current', 'power', 'temperature', 'frequency', 'pressure']:
            # Normalize units
            text = text.replace(' ', '')

        return text


class EntityResolver:
    """Resolve and link entities across documents."""

    def __init__(self, storage):
        """
        Initialize resolver.

        Args:
            storage: Storage backend with entity collections
        """
        self.storage = storage
        self.entity_cache = {}

    def resolve_entity(self, entity: Entity, context: str = "") -> str:
        """
        Resolve entity to canonical ID, creating new entity if needed.

        Args:
            entity: Entity to resolve
            context: Surrounding text for disambiguation

        Returns:
            Entity ID (canonical key)
        """
        normalized = self._normalize(entity.text)

        # Check cache
        cache_key = (normalized, entity.type)
        if cache_key in self.entity_cache:
            return self.entity_cache[cache_key]

        # Query existing entities
        existing = self._find_existing_entity(normalized, entity.type)

        if existing:
            entity_id = existing['_key']
            # Update mention count
            self._update_entity(entity_id, entity)
        else:
            # Create new entity
            entity_id = self._create_entity(entity)

        # Cache result
        self.entity_cache[cache_key] = entity_id

        return entity_id

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        # Remove extra whitespace, lowercase
        return re.sub(r'\s+', ' ', text.strip().lower())

    def _find_existing_entity(self, normalized_text: str, entity_type: str) -> Optional[Dict]:
        """Find existing entity in database."""
        # This will be implemented with ArangoDB queries
        # For now, return None (will create new entities)
        return None

    def _create_entity(self, entity: Entity) -> str:
        """Create new entity in database."""
        # Will be implemented with ArangoDB
        # For now, generate a simple ID
        import hashlib
        entity_id = hashlib.md5(f"{entity.type}_{entity.text}".encode()).hexdigest()[:12]
        return f"entity_{entity_id}"

    def _update_entity(self, entity_id: str, entity: Entity):
        """Update existing entity with new mention."""
        # Will be implemented with ArangoDB
        pass


def extract_product_names(text: str) -> List[Entity]:
    """
    Extract product names using heuristics.

    Args:
        text: Input text

    Returns:
        List of product name entities
    """
    entities = []

    # Pattern: Capitalized words followed by model/product indicators
    product_patterns = [
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:Series|Model|System|Device|Unit|Product)',
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+\d{3,4}[A-Z]?\b',
    ]

    for pattern in product_patterns:
        for match in re.finditer(pattern, text):
            product_name = match.group(1)
            entities.append(Entity(
                text=product_name,
                type='product',
                start=match.start(1),
                end=match.end(1),
                confidence=0.7,  # Lower confidence for heuristic extraction
            ))

    return entities


def extract_entities_from_chunks(chunks: List[Dict], include_products: bool = True) -> Dict[str, List[Entity]]:
    """
    Extract entities from a list of chunks.

    Args:
        chunks: List of chunk dictionaries with 'text' and 'chunk_id'
        include_products: Whether to extract product names (heuristic)

    Returns:
        Dictionary mapping chunk_id to list of entities
    """
    ner = TechnicalNER()
    results = {}

    for chunk in chunks:
        chunk_id = chunk.get('chunk_id') or chunk.get('id')
        text = chunk.get('text', '')

        # Extract technical entities
        entities = ner.extract_entities(text, chunk_id=chunk_id)

        # Extract products (optional)
        if include_products:
            products = extract_product_names(text)
            entities.extend(products)

        results[chunk_id] = entities

    return results


def entity_to_dict(entity: Entity) -> Dict[str, Any]:
    """Convert Entity to dictionary for storage."""
    return {
        'text': entity.text,
        'type': entity.type,
        'start': entity.start,
        'end': entity.end,
        'confidence': entity.confidence,
        'metadata': entity.metadata or {}
    }


def dict_to_entity(data: Dict[str, Any]) -> Entity:
    """Convert dictionary to Entity object."""
    return Entity(
        text=data['text'],
        type=data['type'],
        start=data['start'],
        end=data['end'],
        confidence=data.get('confidence', 1.0),
        metadata=data.get('metadata', {})
    )
