# AQL Queries for PDF Knowledge Graphs

Useful queries to explore your PDF knowledge graphs in ArangoDB.

## Basic Queries

### List all PDFs
```javascript
FOR pdf IN pdfs
  SORT pdf.processed_date DESC
  RETURN {
    slug: pdf.slug,
    filename: pdf.filename,
    pages: pdf.num_pages,
    chunks: pdf.num_chunks,
    figures: pdf.num_figures,
    tables: pdf.num_tables,
    processed: pdf.processed_date
  }
```

### Count nodes and edges by type
```javascript
RETURN {
  nodes: (
    FOR node IN nodes
      COLLECT type = node.type WITH COUNT INTO count
      RETURN {type, count}
  ),
  edges: (
    FOR edge IN edges
      COLLECT type = edge.type WITH COUNT INTO count
      RETURN {type, count}
  )
}
```

---

## Graph Traversal

### Find all references from a section
```javascript
FOR v, e, p IN 1..2 OUTBOUND
  "nodes/section_11_4"  // Change to your section
  edges
  FILTER e.type == "REFERS_TO"
  RETURN {
    from_section: "11.4",
    to: v.node_id,
    to_type: v.type,
    to_label: v.label,
    distance: LENGTH(p.edges)
  }
```

### Find most referenced figures
```javascript
FOR node IN nodes
  FILTER node.type == "Figure"
  LET references = (
    FOR edge IN edges
      FILTER edge._to == CONCAT("nodes/", node._key)
      FILTER edge.type == "REFERS_TO"
      RETURN edge
  )
  SORT LENGTH(references) DESC
  LIMIT 10
  RETURN {
    figure: node.node_id,
    label: node.label,
    page: node.page,
    num_references: LENGTH(references)
  }
```

### Find circular references
```javascript
FOR node IN nodes
  FILTER node.type IN ["Section", "Paragraph"]
  FOR v, e, p IN 2..4 OUTBOUND
    node._id
    edges
    FILTER v._id == node._id  // Back to start
    RETURN {
      cycle: [FOR vertex IN p.vertices RETURN vertex.node_id],
      length: LENGTH(p.vertices)
    }
```

---

## Content Analysis

### Search chunks by full-text
```javascript
FOR chunk IN FULLTEXT(chunks, 'text', 'voltage temperature')
  FILTER chunk.pdf_slug == "manual_tecnico"
  LIMIT 10
  RETURN {
    chunk_id: chunk.chunk_id,
    section: chunk.section_id,
    page: chunk.page,
    text: SUBSTRING(chunk.text, 0, 200)
  }
```

### Find sections without figures
```javascript
FOR section IN nodes
  FILTER section.type == "Section"
  LET has_figures = (
    FOR v, e IN 1..2 OUTBOUND section._id edges
      FILTER v.type == "Figure"
      RETURN 1
  )
  FILTER LENGTH(has_figures) == 0
  RETURN {
    section: section.node_id,
    title: section.label,
    page: section.page
  }
```

### Find pages with most cross-references
```javascript
FOR page IN nodes
  FILTER page.type == "Page"
  LET refs = (
    FOR v, e IN 1..1 OUTBOUND page._id edges
      FILTER v.type == "Paragraph"
      FOR v2, e2 IN 1..1 OUTBOUND v._id edges
        FILTER e2.type == "REFERS_TO"
        RETURN 1
  )
  SORT LENGTH(refs) DESC
  LIMIT 10
  RETURN {
    page: page.node_id,
    label: page.label,
    cross_references: LENGTH(refs)
  }
```

---

## Statistics

### Graph density by PDF
```javascript
FOR pdf IN pdfs
  LET pdf_nodes = (
    FOR node IN nodes
      FILTER node.pdf_slug == pdf.slug
      RETURN 1
  )
  LET pdf_edges = (
    FOR edge IN edges
      FILTER edge.pdf_slug == pdf.slug
      RETURN 1
  )
  RETURN {
    pdf: pdf.filename,
    nodes: LENGTH(pdf_nodes),
    edges: LENGTH(pdf_edges),
    density: LENGTH(pdf_edges) / (LENGTH(pdf_nodes) * (LENGTH(pdf_nodes) - 1))
  }
```

### Section depth distribution
```javascript
FOR section IN nodes
  FILTER section.type == "Section"
  COLLECT depth = LENGTH(SPLIT(section.node_id, ".")) WITH COUNT INTO count
  RETURN {
    depth: depth,
    num_sections: count
  }
```

### Average references per paragraph
```javascript
FOR pdf IN pdfs
  LET paragraphs = (
    FOR node IN nodes
      FILTER node.pdf_slug == pdf.slug
      FILTER node.type == "Paragraph"
      RETURN node
  )
  LET total_refs = (
    FOR node IN paragraphs
      FOR v, e IN 1..1 OUTBOUND node._id edges
        FILTER e.type == "REFERS_TO"
        RETURN 1
  )
  RETURN {
    pdf: pdf.filename,
    total_paragraphs: LENGTH(paragraphs),
    total_references: LENGTH(total_refs),
    avg_refs_per_paragraph: LENGTH(total_refs) / LENGTH(paragraphs)
  }
```

---

## Advanced Queries

### Find isolated nodes (no connections)
```javascript
FOR node IN nodes
  LET incoming = (
    FOR edge IN edges
      FILTER edge._to == node._id
      RETURN 1
  )
  LET outgoing = (
    FOR edge IN edges
      FILTER edge._from == node._id
      RETURN 1
  )
  FILTER LENGTH(incoming) == 0 AND LENGTH(outgoing) == 0
  RETURN {
    node: node.node_id,
    type: node.type,
    label: node.label
  }
```

### Shortest path between two nodes
```javascript
FOR path IN OUTBOUND SHORTEST_PATH
  "nodes/section_1" TO "nodes/figure_5"
  edges
  RETURN {
    path: [FOR v IN path.vertices RETURN v.node_id],
    length: LENGTH(path.edges)
  }
```

### Find hub nodes (most connections)
```javascript
FOR node IN nodes
  LET connections = (
    FOR v, e IN 1..1 ANY node._id edges
      RETURN 1
  )
  SORT LENGTH(connections) DESC
  LIMIT 20
  RETURN {
    node: node.node_id,
    type: node.type,
    label: node.label,
    connections: LENGTH(connections)
  }
```

### Export subgraph for visualization
```javascript
// All nodes and edges for a specific PDF
LET pdf_slug = "manual_tecnico"

LET graph_nodes = (
  FOR node IN nodes
    FILTER node.pdf_slug == pdf_slug
    RETURN {
      id: node.node_id,
      label: node.label || node.node_id,
      type: node.type,
      page: node.page
    }
)

LET graph_edges = (
  FOR edge IN edges
    FILTER edge.pdf_slug == pdf_slug
    RETURN {
      from: edge.from_id,
      to: edge.to_id,
      type: edge.type
    }
)

RETURN {
  nodes: graph_nodes,
  edges: graph_edges
}
```

---

## Maintenance Queries

### Delete all data for a PDF
```javascript
LET pdf_slug = "manual_to_delete"

// Delete chunks
FOR chunk IN chunks
  FILTER chunk.pdf_slug == pdf_slug
  REMOVE chunk IN chunks

// Delete nodes
FOR node IN nodes
  FILTER node.pdf_slug == pdf_slug
  REMOVE node IN nodes

// Delete edges
FOR edge IN edges
  FILTER edge.pdf_slug == pdf_slug
  REMOVE edge IN edges

// Delete PDF metadata
FOR pdf IN pdfs
  FILTER pdf.slug == pdf_slug
  REMOVE pdf IN pdfs

RETURN "Deleted"
```

### Check for orphaned edges
```javascript
FOR edge IN edges
  LET from_exists = DOCUMENT(edge._from) != null
  LET to_exists = DOCUMENT(edge._to) != null
  FILTER !from_exists OR !to_exists
  RETURN {
    edge: edge._id,
    from_exists: from_exists,
    to_exists: to_exists,
    from: edge.from_id,
    to: edge.to_id
  }
```

### Database statistics
```javascript
RETURN {
  total_pdfs: LENGTH(pdfs),
  total_chunks: LENGTH(chunks),
  total_nodes: LENGTH(nodes),
  total_edges: LENGTH(edges),
  db_size_mb: (
    COLLECTION_COUNT(pdfs) +
    COLLECTION_COUNT(chunks) +
    COLLECTION_COUNT(nodes) +
    COLLECTION_COUNT(edges)
  ) / 1000000
}
```
