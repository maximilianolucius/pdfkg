# PROJECT SUMMARY

## 1. Project Overview

**PDFKG** (PDF Knowledge Graph) is an intelligent document processing system that transforms technical PDF manuals into queryable knowledge graphs with AAS (Asset Administration Shell) generation capabilities. The system extracts structured information from PDFs, builds semantic relationships, and enables natural language Q&A over technical documentation.

**Core Capabilities:**
- **Knowledge Graph Construction**: Parses PDFs to build graph databases with sections, cross-references, figures, and tables
- **Semantic Search**: Vector embeddings + FAISS for similarity search across documents
- **Multi-PDF Support**: Process and query multiple documents with cross-document relationship analysis
- **AAS Generation**: Template-driven extraction of AAS v5.0 XML files from technical documentation
- **Natural Language Q&A**: LLM-powered question answering with source attribution

**Technology Stack:**
- **Storage**: ArangoDB (graph/document database) + Milvus (vector database)
- **UI**: Gradio web interface with multi-tab layout
- **PDF Processing**: PyMuPDF + pdfplumber for text extraction
- **Embeddings**: Sentence-transformers models (MiniLM, MPNet, BGE)
- **LLMs**: Google Gemini + Mistral for NLP tasks
- **Graph**: NetworkX for relationship modeling

---

## 1bis. User Interface (Tabs)

### Tab 1: PDF Ingestion
**Purpose**: Upload and process technical PDF manuals

**Features**:
- Multi-file PDF upload (drag & drop or click)
- Configurable embedding model selection (4 options: MiniLM-L6, MPNet, BGE-small/base)
- Adjustable chunk size (100-1000 tokens, default: 500)
- Optional Gemini visual analysis for diagram cross-references
- Force reprocess option to bypass cache
- Real-time progress tracking
- Processing statistics (pages, chunks, sections, figures, tables, cross-refs)
- "Reset Project Data" button to wipe database and rebuild

**Workflow**:
1. Upload PDF(s) → 2. Configure settings → 3. Click "Process PDF" → 4. View status & statistics

### Tab 2: PDF Q&A
**Purpose**: Interactive question answering over processed documents

**Features**:
- PDF dropdown selector with refresh button
- LLM provider selection (None/Gemini/Mistral)
- Adjustable retrieval parameters (top-k sources: 1-10)
- Chatbot interface with history
- Source citations and related content (figures, tables, sections)
- Copy-to-clipboard support

**Example Questions**:
- "What are the operative temperature limits?"
- "How do you mount this device?"
- "What certifications does this product have?"

**Response Types**:
- **Without LLM**: Returns relevant text chunks + page/section references
- **With LLM**: Generates natural language answers citing source material

### Tab 3: Generate AASX
**Purpose**: Template-driven AAS (Asset Administration Shell) file generation

**Features**:
- LLM provider selection (Gemini/Mistral)
- Submodel selector (checkboxes for multiple submodels):
  - Digital Nameplate
  - Technical Data
  - Documentation
  - Handover Documentation
  - Bill of Materials
  - (and more...)
- **Extract Selected Submodels**: Auto-populates JSON templates from PDF content
- **JSON Editor**: Review/edit extracted data per submodel with confidence scores
- **Evidence Display**: Shows extraction confidence and source chunks per field
- **Generate AASX**: Creates AAS v5.0 XML file from current data
- Download button for generated XML file

**Workflow**:
1. Select submodels → 2. Extract templates → 3. Review/edit JSON → 4. Generate AASX → 5. Download

### Tab 4: System Logs
**Purpose**: Real-time system activity monitoring

**Features**:
- Auto-refreshing logs (5-second interval)
- Manual refresh button
- Clear logs button
- Timestamped entries with log levels (INFO/WARNING/ERROR)
- Copy-to-clipboard support

**Log Types**:
- Database connection status
- PDF processing events
- Extraction/validation progress
- Error diagnostics
- LLM API call statistics

---

## 2. How It Works Under the Hood

### PDF Ingestion Pipeline (app.py:178-355)

1. **PDF Upload & Validation**
   - Files stored in `data/input/`
   - Generate slug from filename (sanitized)

2. **Text Extraction** (via `pdf_manager.ingest_pdf()`)
   - PyMuPDF extracts text from all pages
   - pdfplumber handles complex table structures
   - Preserve page numbers (1-indexed)

3. **Document Topology** (via `topology.py`)
   - Parse Table of Contents (ToC)
   - Build section hierarchy with numeric IDs (e.g., "11.4.2")
   - Map sections to page ranges

4. **Intelligent Chunking** (via `chunking.py`)
   - Sentence-based chunking with token limits
   - Configurable max tokens (100-1000)
   - Preserve section context and paragraph boundaries

5. **Embedding Generation** (via `embeds.py`)
   - Use sentence-transformers models
   - Generate dense vectors (384 or 768 dims)
   - L2-normalize for cosine similarity

6. **Cross-reference Extraction** (via `xrefs.py`)
   - Regex patterns detect references:
     - Sections: "see section 11.4.2"
     - Pages: "refer to page 42"
     - Figures: "see Fig. 3", "Figure 12A"
     - Tables: "see Table 2"
   - Optional Gemini vision API for diagram references

7. **Figure/Table Indexing** (via `figtables.py`)
   - Extract captions using regex
   - Link to page locations
   - Index for graph traversal

8. **Knowledge Graph Construction** (via `graph.py`)
   - **Nodes**: Document, Page, Section, Paragraph, Figure, Table
   - **Edges**: CONTAINS (hierarchy), LOCATED_ON (spatial), REFERS_TO (cross-refs)
   - Export formats: Cypher, GraphML, JSON

9. **Storage** (via `storage.py`)
   - **ArangoDB**: Stores documents, chunks, sections, metadata, graph edges
   - **Milvus**: Vector index for semantic search (FAISS alternative for production)
   - **File Fallback**: Legacy file-based storage (parquet, JSON, FAISS)

### Question Answering Pipeline (app.py:688-767)

1. **Query Embedding**
   - Embed user question with same model as chunks
   - Generate query vector

2. **Semantic Search**
   - Query Milvus/FAISS for top-k similar chunks (cosine similarity)
   - Retrieve chunk content, page, section metadata

3. **Graph Traversal**
   - Follow REFERS_TO edges to find related content
   - Aggregate related figures, tables, sections

4. **Answer Generation**
   - **No LLM**: Return raw chunks with source references
   - **With LLM** (Gemini/Mistral):
     - Construct context from retrieved chunks
     - Send prompt: "Answer question using only this context..."
     - Generate natural language response
     - Add source citations

5. **Response Formatting**
   - Display answer in chat
   - Show top 3 sources with page/section/score
   - List related figures/tables/sections

### AAS Generation Pipeline (app.py:1200-1345)

**4-Phase Pipeline** (runs automatically on "Generate AASX" button):

**Phase 1: Classification** (via `aas_classifier.py`)
- Retrieve all PDF chunks from ArangoDB
- Send sample chunks to LLM with submodel definitions
- LLM classifies which AAS submodels apply (e.g., DigitalNameplate, TechnicalData)
- Save classifications to database

**Phase 2: Extraction** (via `aas_extractor.py`)
- Load submodel templates (JSON schemas)
- For each classified submodel:
  - Semantic search for relevant chunks
  - Send template + chunks to LLM
  - LLM extracts structured data matching schema
- Save extracted data + confidence scores + source chunks

**Phase 3: Validation** (via `aas_validator.py`)
- Check mandatory submodels present (DigitalNameplate required)
- Validate mandatory fields complete
- LLM attempts to fill missing fields from available data
- Generate validation report (is_complete, missing_items, suggestions)
- Save completed data

**Phase 4: XML Generation** (via `aas_xml_generator.py`)
- Load validated data
- Generate AAS v5.0 compliant XML structure
- Include all submodels with proper nesting
- Save to `data/output/aas_output_TIMESTAMP.xml`

### Cross-Document Relationships (app.py:789-923)

**When**: Auto-run after multi-PDF ingestion or reset

**Phase 1: MVP** (Low cost, high value)
- **Cross-doc refs**: Detect references like "see Installation Manual section 4"
- **Entity extraction**: Find model numbers, standards, certifications, voltages
- **Versioning**: Detect version relationships (v1.0 vs v2.0)

**Phase 2: Expansion** (Medium cost, global overview)
- **Semantic similarity**: Find similar chunks across documents (cosine threshold)
- **Topic clustering**: Group documents by themes (K-means on embeddings)

**Phase 3: Advanced** (Higher cost, deeper insights)
- **Citation network**: Build graph of document influence patterns

### Auto-Ingestion (app.py:1357-1408)

**On Startup** (`demo.load()` event):
1. Check `data/input/` directory for PDFs
2. Process any new PDFs not in database (cached slugs skipped)
3. Build cross-document relationships if 2+ PDFs present
4. Update dropdown with available PDFs
5. Log summary (newly processed, cached, failed)

---

## 3. Additional Information

### Data Flow Architecture

```
PDF Upload → data/input/
     ↓
PDF Parsing (PyMuPDF)
     ↓
Chunking + Embedding (sentence-transformers)
     ↓
Storage Layer:
  ├─ ArangoDB (graph, documents, metadata)
  └─ Milvus (vector index)
     ↓
Query Interface:
  ├─ Q&A: Semantic search → Graph traversal → LLM generation
  └─ AAS: Classification → Extraction → Validation → XML
```

### Database Schema (ArangoDB)

**Collections**:
- `pdfs`: PDF metadata (filename, slug, num_pages, processed_date)
- `chunks`: Text chunks with embeddings metadata (chunk_id, text, section_id, page, token_count)
- `sections`: Section hierarchy (section_id, title, level, parent_id, page_range)
- `figures`: Figure captions (figure_id, caption, page)
- `tables`: Table captions (table_id, caption, page)
- `metadata`: Global metadata (cross_doc_refs, entities, topics, aas_classifications, etc.)

**Edge Collections**:
- `contains`: Hierarchical edges (Document→Section, Section→Chunk)
- `refers_to`: Cross-reference edges (Chunk→Section, Chunk→Figure)
- `located_on`: Spatial edges (Chunk→Page, Figure→Page)

### Milvus Schema

**Collection**: `pdfkg_chunks`
- **Fields**: `chunk_id` (str), `pdf_slug` (str), `embedding` (vector[dim])
- **Index**: IVF_FLAT + IP (inner product = cosine for normalized vectors)
- **Dimension**: Depends on embedding model (384 or 768)

### Performance Characteristics

**PDF Processing**:
- Small (10 pages): 15-30 seconds
- Medium (50 pages): 30-60 seconds
- Large (100+ pages): 1-3 minutes

**Q&A Latency**:
- Without LLM: <1 second (semantic search only)
- With LLM: 2-5 seconds (includes API call)

**Memory Usage**:
- Base application: ~500 MB
- Per small PDF: +100-200 MB
- Per medium PDF: +200-500 MB
- Per large PDF: +500 MB - 1 GB

### Configuration (.env)

Required variables:
```bash
# Databases
ARANGO_HOST=localhost
ARANGO_PORT=8529
MILVUS_HOST=localhost
MILVUS_PORT=19530

# LLM APIs (optional)
GEMINI_API_KEY=your_key
GEMINI_MODEL=gemini-1.5-flash
MISTRAL_API_KEY=your_key

# Storage backend
STORAGE_BACKEND=arango  # or 'file'
DEFAULT_EMBED_DIM=384   # or 768
```

### Deployment

**Local**:
```bash
docker-compose up -d  # Start ArangoDB + Milvus
python app.py         # Launch web interface (port 8016)
```

**Production**:
- Deploy to Hugging Face Spaces (Gradio native)
- Use persistent volumes for `data/` directory
- Set `share=False` for security
- Add authentication: `demo.launch(auth=("user", "pass"))`

### Limitations

- PDFs must have extractable text (no scanned images)
- ToC detection depends on PDF internal structure
- Cross-reference regex may miss complex patterns
- Concurrent users share same backend (no session isolation)
- Gradio public links expire after 72 hours
- LLM API calls incur costs (track with `llm_stats.py`)

### Key Files

- `app.py`: Gradio web interface (entry point)
- `cli.py`: Command-line interface
- `pdfkg/pdf_manager.py`: Unified ingestion pipeline
- `pdfkg/query.py`: Q&A logic
- `pdfkg/storage.py`: Storage abstraction layer
- `pdfkg/aas_*.py`: AAS generation modules
- `pdfkg/cross_document.py`: Multi-PDF relationship analysis
- `pdfkg/db/arango_client.py`: ArangoDB wrapper
- `pdfkg/db/milvus_client.py`: Milvus wrapper

### Future Enhancements

- AASX packaging (currently generates XML only)
- Session management for multi-user support
- Incremental PDF updates (re-ingest only changed sections)
- Advanced NER for domain-specific entities
- Timeline analysis for document evolution tracking
- GraphQL API for programmatic access
