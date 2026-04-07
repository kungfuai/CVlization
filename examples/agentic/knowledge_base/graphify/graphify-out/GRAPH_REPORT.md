# Graph Report - /workspace/corpus  (2026-04-07)

## Corpus Check
- Corpus is ~2,159 words - fits in a single context window. You may not need a graph.

## Summary
- 94 nodes · 148 edges · 11 communities detected
- Extraction: 84% EXTRACTED · 16% INFERRED · 0% AMBIGUOUS · INFERRED: 24 edges (avg confidence: 0.57)
- Token cost: 0 input · 0 output

## God Nodes (most connected - your core abstractions)
1. `ValidationError` - 11 edges
2. `load_index()` - 10 edges
3. `find_cross_references()` - 9 edges
4. `validate_document()` - 8 edges
5. `_ensure_storage()` - 7 edges
6. `extract_keywords()` - 7 edges
7. `parse_file()` - 7 edges
8. `parse_and_save()` - 7 edges
9. `save_index()` - 6 edges
10. `save_parsed()` - 6 edges

## Surprising Connections (you probably didn't know these)
- `Pipeline Extensibility Guide` --references--> `parse_file()`  [EXTRACTED]
  corpus/architecture.md → /workspace/corpus/parser.py
- `Keyword Extraction Analysis` --references--> `extract_keywords()`  [EXTRACTED]
  corpus/notes.md → /workspace/corpus/processor.py
- `TF-IDF Alternative` --conceptually_related_to--> `extract_keywords()`  [EXTRACTED]
  corpus/notes.md → /workspace/corpus/processor.py
- `Embedding-Based Similarity Alternative` --conceptually_related_to--> `extract_keywords()`  [EXTRACTED]
  corpus/notes.md → /workspace/corpus/processor.py
- `Pipeline Extensibility Guide` --references--> `enrich_document()`  [EXTRACTED]
  corpus/architecture.md → /workspace/corpus/processor.py

## Hyperedges (group relationships)
- **Document Ingestion Pipeline** — parser_parse_file, validator_validate_document, processor_enrich_document, storage_save_parsed, storage_save_processed [EXTRACTED 1.00]
- **Keyword-Based Search and Cross-Reference System** — processor_extract_keywords, processor_find_cross_references, api_handle_search, storage_load_index [INFERRED 0.85]
- **API Orchestration Layer** — api_handle_upload, api_handle_get, api_handle_delete, api_handle_list, api_handle_search, api_handle_enrich [EXTRACTED 1.00]

## Communities

### Community 0 - "Processing & Cross-References"
Cohesion: 0.13
Nodes (20): Pipeline Extensibility Guide, Naive Cross-Reference Rationale, Cross-Reference Storage vs Compute Question, Cross-Reference Threshold Analysis, Embedding-Based Similarity Alternative, Keyword Extraction Analysis, TF-IDF Alternative, enrich_document() (+12 more)

### Community 1 - "Storage Layer"
Cohesion: 0.21
Nodes (16): delete_record(), _ensure_storage(), list_records(), load_index(), load_record(), Storage module - persists documents to disk and maintains the search index. All, Load the full document index from disk., Persist the index to disk. (+8 more)

### Community 2 - "API Endpoints"
Cohesion: 0.17
Nodes (15): handle_delete(), handle_enrich(), handle_get(), handle_list(), handle_search(), handle_upload(), API module - exposes the document pipeline over HTTP. Thin layer over parser, va, Accept a list of file paths, run the full pipeline on each,     and return a sum (+7 more)

### Community 3 - "Parsing Pipeline"
Cohesion: 0.18
Nodes (14): Transaction Safety Concern, batch_parse(), parse_and_save(), parse_file(), parse_json(), parse_markdown(), parse_plaintext(), Parser module - reads raw input documents and converts them into a structured fo (+6 more)

### Community 4 - "Validation Logic"
Cohesion: 0.24
Nodes (11): check_format(), check_required_fields(), normalize_fields(), Validator module - checks that parsed documents meet schema requirements before, Run all validation checks on a parsed document. Raises ValidationError on failur, Raise if any required field is missing., Raise if the format is not in the allowed list., Clean up text fields using the processor. (+3 more)

### Community 5 - "Storage Design Rationale"
Cohesion: 0.67
Nodes (4): Flat Storage Rationale, Repository Pattern Question, Storage Trade-Offs Analysis, Storage Module

### Community 6 - "Validation Placement"
Cohesion: 0.67
Nodes (3): Validation Placement Open Question, Parser Module, Validator Module

### Community 7 - "API Layer Analysis"
Cohesion: 1.0
Nodes (2): API Module, API Layer Analysis

### Community 8 - "Pipeline Architecture"
Cohesion: 1.0
Nodes (2): Data Flow Architecture, Linear Pipeline Rationale

### Community 9 - "ValidationError"
Cohesion: 1.0
Nodes (1): ValidationError

### Community 10 - "Processor Module"
Cohesion: 1.0
Nodes (1): Processor Module

## Knowledge Gaps
- **40 isolated node(s):** `Storage module - persists documents to disk and maintains the search index. All`, `Load the full document index from disk.`, `Persist the index to disk.`, `Write a parsed document to storage. Returns the assigned record ID.`, `Write an enriched document to storage, updating the index with keywords.` (+35 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `API Layer Analysis`** (2 nodes): `API Module`, `API Layer Analysis`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Pipeline Architecture`** (2 nodes): `Data Flow Architecture`, `Linear Pipeline Rationale`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `ValidationError`** (1 nodes): `ValidationError`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Processor Module`** (1 nodes): `Processor Module`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `find_cross_references()` connect `Processing & Cross-References` to `Storage Layer`, `API Endpoints`, `Storage Design Rationale`?**
  _High betweenness centrality (0.157) - this node is a cross-community bridge._
- **Why does `load_index()` connect `Storage Layer` to `Processing & Cross-References`, `API Endpoints`?**
  _High betweenness centrality (0.086) - this node is a cross-community bridge._
- **Are the 9 inferred relationships involving `ValidationError` (e.g. with `check_required_fields()` and `check_format()`) actually correct?**
  _`ValidationError` has 9 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `find_cross_references()` (e.g. with `Storage Module` and `handle_search()`) actually correct?**
  _`find_cross_references()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `_ensure_storage()` (e.g. with `load_index()` and `save_index()`) actually correct?**
  _`_ensure_storage()` has 6 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Storage module - persists documents to disk and maintains the search index. All`, `Load the full document index from disk.`, `Persist the index to disk.` to the rest of the system?**
  _40 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Processing & Cross-References` be split into smaller, more focused modules?**
  _Cohesion score 0.13 - nodes in this community are weakly interconnected._