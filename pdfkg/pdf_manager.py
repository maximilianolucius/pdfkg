"""
PDF management utilities for multi-PDF platform.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Any, Optional


class PDFManager:
    """Manage multiple PDFs and their processed artifacts."""

    def __init__(self, base_dir: Path = Path("data")):
        self.base_dir = base_dir
        self.input_dir = base_dir / "input"
        self.output_dir = base_dir / "out"
        self.registry_file = self.output_dir / "processed_pdfs.json"

        # Create directories
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_pdf_slug(self, filename: str) -> str:
        """Convert filename to slug for directory names."""
        # Remove .pdf extension and sanitize
        slug = Path(filename).stem
        # Replace spaces and special chars with underscores
        slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in slug)
        return slug.lower()

    def register_pdf(
        self,
        filename: str,
        num_pages: int,
        num_chunks: int,
        num_sections: int,
        *,
        slug: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict:
        """Register a processed PDF in the registry."""
        registry = self.load_registry()

        pdf_slug = slug or self.get_pdf_slug(filename)
        pdf_info = {
            "filename": filename,
            "slug": pdf_slug,
            "processed_date": datetime.now().isoformat(),
            "num_pages": num_pages,
            "num_chunks": num_chunks,
            "num_sections": num_sections,
            "output_dir": str(self.output_dir / pdf_slug),
            "metadata": metadata or {},
        }

        registry[pdf_slug] = pdf_info
        self.save_registry(registry)
        return pdf_info

    def load_registry(self) -> dict:
        """Load the PDF registry."""
        if self.registry_file.exists():
            with open(self.registry_file, "r") as f:
                return json.load(f)
        return {}

    def save_registry(self, registry: dict) -> None:
        """Save the PDF registry."""
        with open(self.registry_file, "w") as f:
            json.dump(registry, f, indent=2)

    def get_pdf_info(self, slug: str) -> Optional[dict]:
        """Get info for a specific PDF."""
        registry = self.load_registry()
        return registry.get(slug)

    def list_pdfs(self) -> list[dict]:
        """List all processed PDFs."""
        registry = self.load_registry()
        return list(registry.values())

    def get_pdf_output_dir(self, slug: str) -> Path:
        """Get output directory for a PDF."""
        return self.output_dir / slug

    def get_pdf_input_path(self, filename: str) -> Path:
        """Get input path for a PDF."""
        return self.input_dir / filename

    def pdf_exists(self, slug: str) -> bool:
        """Check if a PDF has been processed."""
        registry = self.load_registry()
        return slug in registry

    def delete_pdf(self, slug: str) -> bool:
        """Delete a PDF and its artifacts."""
        registry = self.load_registry()
        if slug not in registry:
            return False

        # Remove from registry
        pdf_info = registry.pop(slug)
        self.save_registry(registry)

        # Optionally delete files (commented out for safety)
        # output_dir = Path(pdf_info["output_dir"])
        # if output_dir.exists():
        #     shutil.rmtree(output_dir)

        return True
