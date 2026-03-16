# AI-Invoice-Agentic-RAG

## Solution Design v1.0

**Capstone:** Agentic AI-Powered Multilingual Invoice Validation System
**LLM:** Azure OpenAI for capstone submission; Gemini for local development
**Orchestration:** LangGraph
**Status:** Draft v1.1 (revised)

---

## 1. Project Objectives (from Spec)

The system must autonomously:

1. Monitor `data/incoming/` (simulated email inbox) for new invoice files
2. Extract structured fields from PDF, DOCX, and PNG — format-agnostic
3. Translate non-English invoices to English with a confidence score
4. Validate line items against a mock ERP system via FastAPI
5. Generate HTML audit reports per invoice
6. Support RAG-based Q&A over all processed invoices
7. Support human-in-the-loop review and corrections via Streamlit

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Streamlit UI                                │
│   [Live Audit Feed]   [Human Review / HITL]   [RAG Q&A]            │
└────────────────────────────┬────────────────────────────────────────┘
                             │ REST / WebSocket
┌────────────────────────────▼────────────────────────────────────────┐
│                     FastAPI Application                              │
│   /erp/*  (mock ERP)    /invoice/*  (pipeline)   /rag/*  (Q&A)     │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│              FastMCP Server  (tool discovery layer)                 │
│  invoice-watcher | data-harvester | lang-bridge | completeness-     │
│  checker | business-validator | insight-reporter | vector-indexer   │
│  semantic-retriever | chunk-ranker | response-synthesizer           │
└────────┬───────────────────────────────────────────┬────────────────┘
         │                                           │
┌────────▼───────────────┐             ┌────────────▼───────────────┐
│   PIPELINE GRAPH        │             │      RAG GRAPH             │
│   (LangGraph)           │             │      (LangGraph)           │
│                         │             │                            │
│  monitor                │             │  retrieve                  │
│    → extract            │             │    → augment               │
│      → translate        │             │      → generate            │
│        → data_validate  │             │        → reflect           │
│          → biz_validate │             │          → respond         │
│            → report     │             │                            │
│              → index    │             │                            │
└────────────────────────┘             └────────────────────────────┘
         │                                           │
┌────────▼───────────────────────────────────────────▼──────────────┐
│                     LangFuse  (observability)                      │
│         All agent nodes + LLM calls traced automatically           │
└────────────────────────────────────────────────────────────────────┘
```

---

## 3. Technology Stack

| Layer              | Technology                          | Justification                                              |
|--------------------|-------------------------------------|------------------------------------------------------------|
| Language           | Python 3.11+                        | Spec mandate; asyncio, match statements                    |
| LLM                | Dual-provider workflow              | Azure for submission compliance; Gemini for local iteration and prototyping |
| Orchestration      | LangGraph + langgraph-supervisor    | Stateful graph, interrupt(), parallel nodes                |
| LLM integration    | langchain-openai + langchain-google-genai | One config layer can instantiate either provider      |
| Backend / ERP mock | FastAPI + Uvicorn                   | Async, Pydantic-native, auto Swagger docs                  |
| PDF extraction     | pdfplumber                          | Best-in-class table extraction (benchmarked 2025)          |
| DOCX extraction    | python-docx                         | Direct paragraph/table object model access                 |
| OCR                | Pytesseract                         | Spec-mandated; the selected LLM provider structures OCR output |
| Schema validation  | Pydantic v2                         | 17x faster than v1; model_validator for cross-field checks |
| Vector DB          | ChromaDB                            | Zero-setup, persistent, LangChain-native                   |
| Embeddings         | Provider-matched embeddings         | Use Azure embeddings when Azure is active; Gemini embeddings when Gemini is active |
| Reranking          | Active chat model (LLM-as-ranker)   | Reuses whichever provider is active; no extra reranker infra |
| RAG evaluation     | Active chat model (LLM-as-judge)    | 3 prompts replace TruLens; provider chosen at runtime     |
| Tool protocol      | FastMCP + langchain-mcp-adapters    | Spec-mandated MCP; dynamic tool discovery                  |
| Observability      | LangFuse                            | Spec-mandated; one callback covers all nodes               |
| UI                 | Streamlit                           | Spec-mandated; HITL interrupt() integration                |
| Config             | PyYAML                              | rules.yaml loaded at runtime — no hardcoded thresholds     |
| File monitoring    | os.listdir() scan                   | Simple, no async event handling; triggered by Streamlit button or manual call |
| Templating         | Jinja2                              | HTML report generation                                     |

**Provider selection policy**

- Default local-dev mode is `MODEL_PROVIDER=auto`
- If full Azure credentials are present, use Azure OpenAI
- Else, if `GEMINI_API_KEY` is present, use Gemini for local experimentation only
- Else, fail fast at startup with a configuration error
- For the scored capstone demo and final submission, force `MODEL_PROVIDER=azure`

```python
def init_model_provider() -> tuple[str, BaseChatModel, Embeddings]:
    if os.getenv("MODEL_PROVIDER", "auto") in {"auto", "azure"}:
        azure_ready = all([
            os.getenv("AZURE_OPENAI_API_KEY"),
            os.getenv("AZURE_OPENAI_ENDPOINT"),
            os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
            os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        ])
        if azure_ready:
            return ("azure", build_azure_chat_model(), build_azure_embeddings())

    if os.getenv("MODEL_PROVIDER", "auto") in {"auto", "gemini"} and os.getenv("GEMINI_API_KEY"):
        return ("gemini", build_gemini_chat_model(), build_gemini_embeddings())

    raise RuntimeError("No supported model provider credentials found")
```

---

## 4. Project Structure

```
ai_invoice_auditor/
│
├── main.py                        # FastAPI app + MCP server startup
├── config.py                      # Loads rules.yaml, env vars
├── observability.py               # LangFuse CallbackHandler singleton
├── guardrails.py                  # RAI: input sanitise, PII mask, confidence check
│
├── models/
│   ├── invoice.py                 # InvoiceHeader, LineItem, TranslationResult
│   ├── validation.py              # ValidationResult, Discrepancy
│   ├── erp.py                     # Vendor, PurchaseOrder, SKU
│   ├── rag.py                     # RAGQuery, RAGResponse, TriadScores
│   └── state.py                   # InvoiceState, RAGState (LangGraph TypedDicts)
│
├── agents/
│   ├── orchestrator.py            # LangGraph pipeline supervisor graph
│   ├── monitor_agent.py           # Invoice-Monitor Agent
│   ├── extractor_agent.py         # Extractor Agent
│   ├── translator_agent.py        # Translation Agent
│   ├── data_validator_agent.py    # Invoice Data Validation Agent
│   ├── business_validator_agent.py# Business Validation Agent
│   ├── reporter_agent.py          # Reporting Agent
│   └── rag/
│       ├── rag_orchestrator.py    # LangGraph RAG supervisor graph
│       ├── indexing_agent.py      # Indexing Agent
│       ├── retrieval_agent.py     # Retrieval Agent
│       ├── augmentation_agent.py  # Augmentation Agent (reranker)
│       ├── generation_agent.py    # Generation Agent
│       └── reflection_agent.py    # Reflection Agent (RAG Triad)
│
├── tools/
│   ├── base_tool.py               # BaseTool ABC — all tools inherit
│   ├── invoice_watcher.py         # Invoice-Watcher Tool
│   ├── data_harvester.py          # Data Harvester Tool
│   ├── lang_bridge.py             # Lang-Bridge Tool
│   ├── completeness_checker.py    # Data Completeness Checker Tool
│   ├── business_validator.py      # Business Validation Tool
│   ├── insight_reporter.py        # Insight Reporter Tool
│   ├── vector_indexer.py          # Vector-Indexer Tool
│   ├── semantic_retriever.py      # Semantic-Retriever Tool
│   ├── chunk_ranker.py            # Chunk-Ranker Tool
│   └── response_synthesizer.py   # Response-Synthesizer Tool
│
├── api/
│   ├── erp_router.py              # GET /erp/vendor, /erp/po, /erp/sku
│   └── invoice_router.py          # POST /invoice/process, GET /invoice/{id}
│
├── mcp_server.py                  # FastMCP — all 10 tools as @mcp.tool() endpoints
│
├── streamlit_app.py               # Streamlit UI — 3 tabs
│
├── templates/
│   └── report.html.j2             # Jinja2 HTML report template
│
├── config/
│   └── rules.yaml                 # Already exists — validation thresholds
│
├── data/                          # Already exists — invoices + ERP mock data
│
├── outputs/
│   └── reports/                   # Generated HTML reports
│
├── logs/
│   └── invoice_auditor.log        # Audit log
│
├── chroma_db/                     # ChromaDB persistence directory
│
├── pyproject.toml
└── .env                           # Provider credentials + LangFuse configuration
```

---

## 5. Data Models (Pydantic v2)

### 5.1 Extraction Models (Permissive by Design)

```python
# models/invoice.py
from pydantic import BaseModel, field_validator, model_validator, Field
from datetime import date
from typing import Literal, Optional

class ExtractedLineItem(BaseModel):
    item_code: Optional[str] = None
    description: Optional[str] = None
    qty: Optional[float] = None
    unit_price: Optional[float] = None
    total: Optional[float] = None

class ExtractedInvoiceHeader(BaseModel):
    invoice_no: Optional[str] = None
    invoice_date: Optional[date] = None
    po_number: Optional[str] = None         # extracted from invoice or meta["subject"]
    vendor_id: Optional[str] = None         # resolved from name → vendors.json
    currency: Optional[str] = None
    total_amount: Optional[float] = None
    line_items: list[ExtractedLineItem] = Field(default_factory=list)

class TranslationResult(BaseModel):
    original_language: str
    translated_text: str
    translation_confidence: float = Field(ge=0.0, le=1.0)

class MissingFieldReport(BaseModel):
    field: str
    location: Literal["header", "line_item"]
    action: str                             # from rules.yaml: flag / reject
```

These extraction models are intentionally permissive. Missing fields should survive extraction as `None` so the Data Validation Agent can enforce `rules.yaml` and generate a reportable audit outcome rather than failing fast with a `ValidationError`.

### 5.2 Validation Models

```python
# models/validation.py
from pydantic import BaseModel
from typing import Literal

class Discrepancy(BaseModel):
    item_code: str
    field: str                              # unit_price | qty | currency
    invoice_value: float | str
    erp_value: float | str
    deviation_pct: Optional[float]
    breaches_tolerance: bool

class ValidationResult(BaseModel):
    invoice_no: str
    status: Literal["auto_approve", "flag", "manual_review", "reject"]
    missing_fields: list[MissingFieldReport] = []
    discrepancies: list[Discrepancy] = []
    translation_confidence: float = 1.0
    recommendation: str
    reviewer_decision: Optional[str] = None   # set by HITL
```

### 5.3 LangGraph State

```python
# models/state.py
from typing import TypedDict, Annotated, Optional
from operator import add

class InvoiceState(TypedDict):
    # Inputs
    file_path: str
    meta: dict

    # Pipeline outputs (populated per node)
    raw_text: Optional[str]
    extracted: Optional[dict]              # InvoiceHeader as dict
    translated: Optional[dict]            # TranslationResult as dict
    validation_result: Optional[dict]     # ValidationResult as dict
    report_path: Optional[str]

    # Accumulated log messages
    messages: Annotated[list, add]

    # Routing
    status: str
    next: str

class RAGState(TypedDict):
    query: str
    retrieved_chunks: list[str]
    reranked_chunks: list[str]
    answer: str
    triad_scores: dict                     # context_rel, groundedness, answer_rel
    passed_reflection: bool
    messages: Annotated[list, add]
```

---

## 6. Pipeline Agents

### 6.1 Invoice-Monitor Agent

**Tool:** Invoice-Watcher Tool
**Trigger:** `os.listdir()` scan — called manually, on a Streamlit button press, or on a simple polling loop. No watchdog; avoids async event-handling race conditions with the LangGraph pipeline.

```python
def scan_incoming(directory: str) -> list[dict]:
    seen = set()
    results = []
    for fname in os.listdir(directory):
        if fname.endswith(".meta.json"):
            stem = fname.replace(".meta.json", "")
            # Find actual attachment — ignore declared extension in meta
            attachment = next(
                (f for f in os.listdir(directory)
                 if f.startswith(stem) and not f.endswith(".meta.json")),
                None
            )
            if attachment and stem not in seen:
                seen.add(stem)
                results.append({"file_path": ..., "meta": ...})
    return results
```

**Known edge cases from data analysis:**
- `INV_DE_004.meta.json` declares `.docx` but file is `.pdf` → scan uses actual file on disk, ignores meta's declared extension
- `Actual_invoice.pdf` has no meta → treat as an out-of-band local file, skip and log, quarantine to `data/unmatched/`; exclude from capstone demo runs

### 6.2 Extractor Agent

**Tool:** Data Harvester Tool
**Logic per format:**

```
PDF  → pdfplumber.extract_text() + extract_tables()
DOCX → python-docx paragraphs + table rows joined as text
PNG  → pytesseract.image_to_string(Image.open(path), lang="eng")
     → raw OCR text passed to the active provider for structuring
```

**Structured extraction call** (provider selected at startup; raw text → `ExtractedInvoiceHeader`):
```python
extractor_llm = chat_llm.with_structured_output(ExtractedInvoiceHeader)
invoice = extractor_llm.invoke(raw_text)
```

**Post-extraction hallucination check** — the LLM may fill in missing fields on malformed invoices (e.g. `INV_EN_006_malformed.pdf`) rather than returning null. Validate extracted output against the source text before trusting it:

```python
def verify_extraction(extracted: dict, raw_text: str) -> dict:
    """Cross-check extracted fields exist in source text. Flag invented values."""
    flagged = []
    if extracted.get("invoice_no"):
        if extracted["invoice_no"] not in raw_text:
            flagged.append("invoice_no")      # LLM hallucinated it
            extracted["invoice_no"] = None    # treat as missing
    if extracted.get("po_number"):
        if extracted["po_number"] not in raw_text:
            flagged.append("po_number")
            extracted["po_number"] = None
    if extracted.get("currency"):
        symbol_present = any(s in raw_text for s in ["$","€","₹","£","USD","EUR","INR","GBP"])
        if not symbol_present:
            flagged.append("currency")
            extracted["currency"] = None
    if flagged:
        logger.warning(f"Hallucinated fields cleared: {flagged}")
    return extracted
```

This ensures `INV_EN_006_malformed` correctly reaches the Data Validation Agent with `invoice_no=None` and `currency=None`, triggering the `flag` action from `rules.yaml`.

**Vendor ID resolution** (since no invoice contains VEND-XXX explicitly):
1. Extract vendor name from invoice
2. Fuzzy match against `vendors.json[].vendor_name`
3. Fall back to email sender domain from `meta.json`

**PO number resolution** (required before ERP lookup):
1. Extract `po_number` from the invoice body
2. If missing, regex-extract it from `meta["subject"]`
3. If still missing, keep it as `None` and let the Data Validation Agent apply `rules.yaml`

**Currency normalisation** (applied before Pydantic validation):
```python
SYMBOL_MAP = {"$": "USD", "€": "EUR", "₹": "INR", "£": "GBP"}
```

### 6.3 Translation Agent

**Tool:** Lang-Bridge Tool
**Condition:** `meta["language"] != "en"` OR detected language ≠ English
**Translation model returns:**
```json
{"translated_text": "...", "translation_confidence": 0.94}
```

**Guardrail:** If `translation_confidence < 0.70` → route to `manual_review`; continue to report generation but do not auto-approve
**German field label mapping** (required by DE004):

| German | English |
|--------|---------|
| Rechnungsnummer | invoice_no |
| Rechnungsdatum | invoice_date |
| Bestellnummer | po_number |
| MwSt | tax |
| Gesamtbetrag | total_amount |
| Zwischensumme | subtotal |

**Spanish field label mapping** (required by ES003):

| Spanish | English |
|---------|---------|
| Factura No | invoice_no |
| Fecha | invoice_date |
| IVA | tax |
| Total | total_amount |

### 6.4 Invoice Data Validation Agent

**Tool:** Data Completeness Checker Tool
**Config source:** `config/rules.yaml` loaded at runtime via PyYAML
**Checks:**
- All `required_fields.header` present and non-null after extraction + metadata enrichment
- `po_number` is included in `required_fields.header` in `rules.yaml`
- All `required_fields.line_item` present per line
- `data_types` match (invoice_date is parseable as date, amounts are float)
- Currency is in `accepted_currencies`
- Line total arithmetic is correct (`qty × unit_price ≈ total`)

**rules.yaml expectation:**
```yaml
required_fields:
  header:
    - invoice_no
    - invoice_date
    - po_number
    - vendor_id
    - currency
    - total_amount
```

**Action mapping from rules.yaml:**
```python
if missing_field:       status = "flag"          # missing_field_action
if invalid_currency:    status = "reject"         # invalid_currency_action
```

**Expected outcomes per test invoice:**
- `INV_EN_002`: flag — missing item_code
- `INV_EN_005`: flag — missing currency + item_codes
- `INV_EN_006`: flag — missing invoice_no + currency

### 6.5 Business Validation Agent

**Tool:** Business Validation Tool
**ERP calls (via FastAPI, run concurrently):**
```python
po_number = invoice.po_number or extract_po_from_subject(state["meta"]["subject"])
if not po_number:
    return {
        **state,
        "messages": ["Skipping ERP validation because Data Validation already flagged missing po_number"]
    }

vendor, po = await asyncio.gather(
    get_vendor(invoice.vendor_id),
    get_po(po_number)
)
```

**Comparison logic per line item:**
```python
# Thresholds loaded from rules.yaml — not hardcoded
PRICE_TOL = config["tolerances"]["price_difference_percent"] / 100   # 0.05
QTY_TOL   = config["tolerances"]["quantity_difference_percent"] / 100 # 0.00
TAX_TOL   = config["tolerances"]["tax_difference_percent"] / 100      # 0.02

price_dev = abs(inv_price - erp_price) / erp_price
if price_dev > PRICE_TOL:
    discrepancies.append(Discrepancy(..., breaches_tolerance=True))
```

**Auto-approve condition:**
```python
no_discrepancies = len(discrepancies) == 0
high_confidence  = translation_confidence >= AUTO_APPROVE_THRESHOLD  # 0.95
if no_discrepancies and high_confidence:
    status = "auto_approve"
elif translation_confidence < 0.70:
    status = "manual_review"
elif discrepancies:
    status = "manual_review"       # total_mismatch_action
```

**Expected outcomes per test invoice:**
- `INV_EN_001`: manual_review — SKU-002 price $3.50 vs ERP $3.00 (+16.7%)
- `INV_ES_003`: auto_approve — all values match, confidence ≥ 0.95
- `INV_DE_004`: auto_approve — all values match

### 6.6 Reporting Agent

**Tool:** Insight Reporter Tool
**Output:** `outputs/reports/{invoice_no}_report.html`
**Template sections (Jinja2):**
1. Invoice summary (vendor, date, PO, currency, total)
2. Translation details (source language, confidence score, confidence indicator)
3. Data completeness — table of missing fields with action taken
4. Business validation — line-item diff table (invoice vs ERP, deviation %)
5. Final recommendation badge (auto_approve / flag / manual_review / reject)
6. Audit trail — timestamps per pipeline stage

---

## 7. RAG Pipeline (5 Agents)

### Flow

```
user_query
    │
    ▼
[Retrieval Agent]  →  semantic search in ChromaDB  →  top-10 chunks
    │
    ▼
[Augmentation Agent]  →  cross-encoder reranking  →  top-3 chunks
    │
    ▼
[Generation Agent]  →  active chat model + context  →  answer
    │
    ▼
[Reflection Agent]  →  TruLens RAG Triad scoring
    │
    ├── passed (all scores ≥ 0.7)  →  return answer to user
    └── failed  →  re-prompt Generation Agent (max 2 retries)
```

### 7.1 Indexing Agent — Vector-Indexer Tool

**Graph boundary note:** The Indexing Agent is the **last node in the pipeline graph** (not the first node in the RAG graph). It needs `translated_text`, `invoice_no`, and `vendor` from `InvoiceState` — data that lives in the pipeline graph. The RAG graph only activates on a user Q&A query and has no knowledge of `InvoiceState`.

```
Pipeline graph:  monitor → extract → translate → data_validate
                 → biz_validate → report → [index]  ← last node, uses InvoiceState

RAG graph:       retrieve → augment → generate → reflect    ← activated by user query only
```

**Triggered** after each invoice is processed by the Reporting Agent
**Process:**
```python
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
chunks   = splitter.split_text(translated_text)
vectorstore.add_texts(
    texts=chunks,
    metadatas=[{"invoice_no": invoice_no, "vendor": vendor_name, "po": po_number}] * len(chunks)
)
```

### 7.2 Retrieval Agent — Semantic-Retriever Tool

```python
results = vectorstore.similarity_search_with_score(query, k=10)
# Returns chunks + cosine similarity scores
```

### 7.3 Augmentation Agent — Chunk-Ranker Tool

LLM-based reranking via the active provider — no PyTorch dependency, no extra reranker service:

```python
def llm_rerank(query: str, chunks: list[str]) -> list[dict]:
    """Returns top-3 chunks with relevance scores — scores flow to Reflection Agent
    and surface in the Streamlit Q&A tab alongside each source chunk."""
    numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(chunks))
    prompt = f"""
You are a relevance ranking assistant.
Query: "{query}"

Rank the following chunks by relevance to the query.
Return a JSON array of objects for the top 3, in order of relevance.
Each object must have: "index" (1-based int) and "score" (float 0.0-1.0).
Example: [{{"index": 3, "score": 0.95}}, {{"index": 7, "score": 0.82}}, {{"index": 1, "score": 0.71}}]

Chunks:
{numbered}
"""
    response = ranker_llm.invoke(prompt)
    ranked = json.loads(response.content)[:3]         # e.g. [{"index":3,"score":0.95}, ...]
    return [
        {"chunk": chunks[r["index"] - 1], "score": r["score"]}
        for r in ranked
    ]
```

The `score` from each ranked chunk is stored in `RAGState.reranked_chunks` as `list[dict]` and passed to the Generation Agent (which uses `chunk`) and the Reflection Agent (which uses `score` as a prior signal alongside its own triad scores).

### 7.4 Generation Agent — Response-Synthesizer Tool

```python
prompt = f"""
You are an invoice audit assistant. Answer using ONLY the context below.
If the answer is not in the context, say "I don't have that information."

Context:
{chr(10).join(top_3_chunks)}

Question: {query}
"""
answer = answer_llm.invoke(prompt).content
```

### 7.5 Reflection Agent — RAG Triad Scoring (LLM-as-judge)

Three focused prompts on the active provider — same three axes as TruLens RAG Triad, zero infrastructure overhead:

```python
# agents/rag/reflection_agent.py

CONTEXT_RELEVANCE_PROMPT = """
On a scale of 0.0 to 1.0, how relevant is the following context to the query?
Query: {query}
Context: {context}
Return ONLY a JSON object: {{"score": 0.0}}
"""

GROUNDEDNESS_PROMPT = """
On a scale of 0.0 to 1.0, how well is the answer supported by the context?
Penalise any claims not found in the context.
Context: {context}
Answer: {answer}
Return ONLY a JSON object: {{"score": 0.0}}
"""

ANSWER_RELEVANCE_PROMPT = """
On a scale of 0.0 to 1.0, how directly does the answer address the query?
Query: {query}
Answer: {answer}
Return ONLY a JSON object: {{"score": 0.0}}
"""

def evaluate_triad(query: str, context_chunks: list[str], answer: str) -> dict:
    context = "\n---\n".join(context_chunks)
    scores = {}
    for name, prompt_template, kwargs in [
        ("context_relevance", CONTEXT_RELEVANCE_PROMPT, {"query": query, "context": context}),
        ("groundedness",      GROUNDEDNESS_PROMPT,      {"context": context, "answer": answer}),
        ("answer_relevance",  ANSWER_RELEVANCE_PROMPT,  {"query": query, "answer": answer}),
    ]:
        resp = judge_llm.invoke(prompt_template.format(**kwargs))
        scores[name] = json.loads(resp.content)["score"]

    scores["passed"] = all(v >= 0.7 for v in scores.values())
    return scores
```

If `passed=False`, the Generation Agent is re-prompted once before the answer is returned (max 1 retry to avoid cost runaway). Scores surface in the Streamlit Q&A tab alongside every answer.

---

## 8. FastMCP Server

**Execution strategy:** MCP discovery is the **primary demo path** — agents call tools via `MultiServerMCPClient` at runtime, satisfying the spec requirement for dynamic tool discovery. Direct-wired calls are the **silent fallback** — each node imports the tool function directly and calls it only if the MCP client raises a connection error. This keeps the demo clean while ensuring a broken MCP server never crashes the pipeline.

```python
# Pattern applied in every agent node
async def extract_node(state: InvoiceState) -> InvoiceState:
    try:
        # Primary path — MCP discovery (what evaluators see)
        async with MultiServerMCPClient(MCP_CONFIG) as client:
            tools = {t.name: t for t in await client.get_tools()}
            result = await tools["data_harvester_tool"].ainvoke(
                {"file_path": state["file_path"]}
            )
    except Exception as mcp_err:
        logger.warning(f"MCP unavailable ({mcp_err}), using direct fallback")
        # Silent fallback — direct import
        result = data_harvester_tool(file_path=state["file_path"])
    return {**state, "raw_text": result}
```

All 10 tools registered as MCP endpoints — agents discover them at runtime:

```python
# mcp_server.py
from fastmcp import FastMCP

mcp = FastMCP("invoice-auditor")

@mcp.tool()
def invoice_watcher_tool(directory: str) -> list[dict]:
    """Scans for unprocessed invoice+meta file pairs."""

@mcp.tool()
def data_harvester_tool(file_path: str) -> str:
    """Extracts raw text from PDF (pdfplumber), DOCX (python-docx), PNG (pytesseract)."""

@mcp.tool()
def lang_bridge_tool(text: str, source_language: str) -> dict:
    """Translates to English. Returns {translated_text, translation_confidence}."""

@mcp.tool()
def data_completeness_checker_tool(invoice_dict: dict) -> dict:
    """Checks required fields against rules.yaml. Returns {missing_fields, status}."""

@mcp.tool()
def business_validation_tool(invoice_dict: dict, po_number: str) -> dict:
    """Calls ERP API, compares line items, returns {discrepancies, status}."""

@mcp.tool()
def insight_reporter_tool(validation_result: dict) -> str:
    """Generates HTML report, returns output file path."""

@mcp.tool()
def vector_indexer_tool(text: str, metadata: dict) -> bool:
    """Chunks and embeds invoice text into ChromaDB."""

@mcp.tool()
def semantic_retriever_tool(query: str, k: int = 10) -> list[dict]:
    """Semantic similarity search. Returns chunks with scores."""

@mcp.tool()
def chunk_ranker_tool(query: str, chunks: list[str]) -> list[str]:
    """Cross-encoder reranking. Returns top-3 most relevant chunks."""

@mcp.tool()
def response_synthesizer_tool(query: str, context_chunks: list[str]) -> str:
    """Generates grounded natural language answer using the active chat model."""
```

---

## 9. FastAPI ERP Mock

```
GET  /erp/vendor/{vendor_id}     → Vendor
GET  /erp/po/{po_number}         → PurchaseOrder with line_items[]
GET  /erp/sku/{item_code}        → SKU

POST /invoice/process            → triggers full pipeline for a file
GET  /invoice/{invoice_no}       → InvoiceState (pipeline status)
GET  /invoice/{invoice_no}/report → HTML report file
POST /rag/query                  → RAGQuery → RAGResponse
```

All ERP data served from `data/ERP_mockdata/*.json` via `Depends(get_erp_data)` — swappable for tests.

---

## 10. LangFuse Observability

Single `CallbackHandler` passed to every graph invocation — captures all nodes automatically:

```python
# observability.py
from langfuse.callback import CallbackHandler

langfuse_handler = CallbackHandler(
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
)

# Usage — same pattern for both pipeline and RAG graphs
result = pipeline_graph.invoke(
    initial_state,
    config={"callbacks": [langfuse_handler]}
)
```

**What gets traced automatically:**
- Every LangGraph node (start time, end time, input, output)
- Every LLM API call (prompt tokens, completion tokens, latency, cost)
- Tool calls via MCP
- Errors and retries

---

## 11. Streamlit UI

Three tabs covering all UI requirements:

```python
# streamlit_app.py
tabs = st.tabs(["Live Audit", "Human Review", "RAG Q&A"])

# Tab 1 — Live Audit Feed
# Polls /invoice/* endpoint, shows pipeline status per invoice

# Tab 2 — Human Review (HITL)
# Displays invoices with status = "manual_review"
# Shows discrepancy table from ValidationResult
# [Approve] / [Reject] / [Edit] buttons → resumes LangGraph via interrupt()

# Tab 3 — RAG Q&A
# Text input → POST /rag/query
# Displays answer + source chunks + RAG Triad score badges
```

**HITL flow using LangGraph interrupt():**

> ⚠️ `interrupt()` requires a **checkpointer** to persist state while waiting for human input. Wire this in Phase 3 (not Phase 5) — leaving it late risks the final hour being spent debugging state serialization.

```python
# agents/orchestrator.py — wire checkpointer at graph compile time (Phase 3)
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()   # swap for SqliteSaver("checkpoints.db") for persistence

pipeline_graph = graph_builder.compile(checkpointer=checkpointer)
```

Use explicit `interrupt()` inside `business_validate_node` only after discrepancies are known. That gives the reviewer enough context to approve, reject, or edit.

```python
# In business_validator_agent.py
from langgraph.types import interrupt, Command

if result.status == "manual_review":
    human_decision = interrupt({
        "invoice_no": invoice_no,
        "discrepancies": result.discrepancies,
        "message": "Discrepancies exceed tolerance. Approve or reject."
    })
    result.reviewer_decision = human_decision["decision"]
```

**Edit/correction re-entry flow:**

When a human edits field values (e.g. corrects a unit price, adds a missing item code), the correction must **re-enter the pipeline at `data_validate_node`**, not restart from the monitor. Restarting would re-run extraction and translation unnecessarily and overwrite the human's corrections.

```
Human edits in Streamlit
    │
    ▼
state["extracted"] is updated with corrected fields   ← Streamlit writes to state
    │
    ▼
graph.invoke(Command(resume={"decision": "edit", "corrections": edited_fields}))
    │
    ▼
orchestrator routes to data_validate_node             ← re-enters mid-graph
    │
    ▼
data_validate → biz_validate → report → index         ← runs from here forward only
```

```python
# In orchestrator.py — conditional edge after interrupt
def route_after_hitl(state: InvoiceState) -> str:
    decision = state["validation_result"].get("reviewer_decision")
    if decision == "edit":
        return "data_validate_node"   # re-enter at validation with corrected data
    return END                        # approve or reject → go to report
```

Corrections update only `state["extracted"]` — `state["raw_text"]` and `state["translated"]` are untouched.

---

## 12. Error Handling & Graceful Degradation

Every agent node wraps its logic in a `try/except` and routes errors into the graph state rather than raising — so one bad invoice never stops the pipeline processing the next one.

**Per-agent pattern:**
```python
def extract_node(state: InvoiceState) -> InvoiceState:
    try:
        raw_text = data_harvester_tool(state["file_path"])
        return {**state, "raw_text": raw_text}
    except FileNotFoundError:
        logger.error(f"File not found: {state['file_path']}")
        return {**state, "status": "reject", "next": END,
                "messages": [f"ERROR: file not found"]}
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return {**state, "status": "flag", "next": "report_node",
                "messages": [f"ERROR: extraction failed — {e}"]}
```

**Degradation rules per failure type:**

| Agent | Failure | Degradation |
|---|---|---|
| Extractor | pdfplumber fails on corrupt PDF | Retry with pytesseract (image fallback); if still fails → `flag` and skip to report |
| Translator | Active LLM provider timeout | Re-attempt once; if fails → mark `translation_confidence=0`, route to `manual_review` |
| Translator | Confidence < 0.70 | Route directly to `manual_review`; do not block pipeline |
| Business Validator | ERP endpoint 404 (no PO match) | Add to `discrepancies` as `"po_not_found"`; status = `flag` |
| Business Validator | ERP endpoint down | Log, mark status = `flag`, proceed to report with partial data |
| Reflection Agent | Active LLM provider triad call fails | Return `scores={"passed": False, "error": "evaluation_unavailable"}`; do not block Q&A |
| MCP client | Connection error | Silent fallback to direct-wired tool (see Section 8) |

**Graph-level error routing** — the `status` and `next` fields in `InvoiceState` control where a failed invoice goes:
```python
# In orchestrator.py
def route_after_extract(state: InvoiceState) -> str:
    if state.get("status") == "reject":
        return "report_node"       # write a rejection report, skip validation
    if not state.get("raw_text"):
        return "report_node"       # extraction produced nothing
    return "translate_node"        # happy path
```

---

## 13. RAI Guardrails

Applied at every agent boundary — not a single module but a cross-cutting concern:

| Layer | Where | What |
|---|---|---|
| Input | Before any LLM prompt | Sanitise prompt injection patterns (`ignore previous instructions`, `<\|im_start\|>`) |
| Input | Extractor output | Validate extracted JSON against Pydantic models before passing downstream |
| Output | Translation output | Route to `manual_review` if `translation_confidence < 0.70`; block auto-approval |
| Output | RAG generation | Reflection Agent blocks answer if any Triad score < 0.7 |
| Output | Reports | Mask PII — phone numbers, IBAN, bank account numbers |
| Interaction | Business Validator | Human confirmation required before auto-approving invoices above a value threshold |

---

## 14. Known Data Issues (Pre-code Fixes Required)

| Issue | File | Fix |
|---|---|---|
| meta.json declares `.docx`, file is `.pdf` | `INV_DE_004.meta.json` | Change `"attachments": ["INV_DE_004.docx"]` → `"INV_DE_004.pdf"` |
| No meta.json for local extra file | `Actual_invoice.pdf` | Move to `data/unmatched/` and exclude from capstone demo dataset |
| No item_code column | `INV_EN_002.pdf`, `INV_EN_005_scan.png` | Extractor must fuzzy-match description → SKU |
| Currency absent | `INV_EN_005_scan.png`, `INV_EN_006_malformed.pdf` | Infer from vendor context; flag as missing field |

---

## 15. Environment Variables

```bash
# .env
MODEL_PROVIDER=auto

AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_API_VERSION=<your-api-version>
AZURE_OPENAI_CHAT_DEPLOYMENT=invoice-auditor-chat
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=invoice-auditor-embedding

GEMINI_API_KEY=your_gemini_api_key
GEMINI_CHAT_MODEL=gemini-2.5-flash
GEMINI_EMBEDDING_MODEL=text-embedding-004

LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com

ERP_API_BASE=http://localhost:8000
MCP_SERVER_URL=http://localhost:8001/mcp

INCOMING_DIR=./data/incoming
REPORTS_DIR=./outputs/reports
LOG_FILE=./logs/invoice_auditor.log
CHROMA_DIR=./chroma_db
```

---

## 16. Dependencies

```toml
# pyproject.toml
[project]
name = "ai-invoice-auditor"
version = "0.1.0"
requires-python = ">=3.11"

dependencies = [
    # Orchestration
    "langgraph>=0.2",
    "langchain>=0.3",
    "langchain-google-genai",
    "langchain-openai",
    "langgraph-supervisor",

    # LLM
    "google-generativeai",
    "openai",

    # Extraction
    "pdfplumber",
    "python-docx",
    "pytesseract",
    "Pillow",

    # Backend
    "fastapi",
    "uvicorn[standard]",

    # RAG
    "langchain-chroma",
    "chromadb",
    # sentence-transformers removed — active provider handles reranking

    # MCP
    "fastmcp",
    "langchain-mcp-adapters",

    # Observability
    "langfuse",

    # UI
    "streamlit",

    # Data / validation
    "pydantic>=2.0",
    "pyyaml",

    # Templating
    "jinja2",

    # watchdog removed — os.listdir() scan is sufficient for demo
]
```

---

## 17. Build Order, Acceptance Criteria & Git Strategy

**Git strategy:** Work on `main` for scaffolding; each phase gets a `feature/phase-N-<name>` branch; merge to `main` via PR once the phase acceptance criteria passes.

---

**Phase 1 — Foundation (no LLM)**
1. Pydantic models (`models/`)
2. FastAPI ERP mock (`api/erp_router.py`)
3. Data Completeness Checker Tool
4. Business Validation Tool
5. Error handling skeleton — `try/except` pattern in place for all future nodes

✅ **Acceptance:** `GET /erp/po/PO-1001` returns correct line items; `data_completeness_checker_tool(inv_en_006_dict)` returns `{missing_fields: ["invoice_no","currency"], status: "flag"}`.

---

**Phase 2 — Extraction Pipeline**
6. Data Harvester Tool (pdfplumber + python-docx + pytesseract)
7. Extractor Agent + provider-selected structuring + post-extraction hallucination check
8. Lang-Bridge Tool + Translation Agent
9. Fix `INV_DE_004.meta.json`

✅ **Acceptance:** All 6 invoices extract without crash; `INV_ES_003` returns `translation_confidence ≥ 0.90`; `INV_EN_006_malformed` extraction returns `invoice_no=None, currency=None`.

---

**Phase 3 — Orchestration**
10. `InvoiceState` TypedDict
11. LangGraph pipeline graph — all 7 nodes (including index)
12. `MemorySaver` checkpointer + `interrupt()` wired ← **do not defer**
13. Reporting Agent + Jinja2 template
14. MCP server + MCP-primary / direct-fallback pattern per node

✅ **Acceptance:** Full pipeline run on `INV_EN_001` produces an HTML report at `outputs/reports/INV-1001_report.html` with status `manual_review` and SKU-002 discrepancy listed; `interrupt()` pauses graph and resumes on `Command(resume=...)`.

---

**Phase 4 — RAG**
15. Indexing Agent + ChromaDB (last node in pipeline graph)
16. Retrieval Agent
17. Augmentation Agent — provider-selected reranking returning `list[dict]` with scores
18. Generation Agent
19. Reflection Agent — 3 LLM-as-judge prompts
20. LangGraph RAG graph

✅ **Acceptance:** Query `"What is the total amount on INV-1001?"` returns correct answer (`$1617.00`) with all three triad scores ≥ 0.7; a nonsense query returns `"I don't have that information"` with low context relevance score.

---

**Phase 5 — Cross-cutting**
21. LangFuse callback on both graphs
22. RAI guardrails
23. Streamlit UI — 3 tabs, HITL edit flow re-entering at `data_validate_node`

✅ **Acceptance:** LangFuse dashboard shows a complete trace for a full pipeline run with all nodes visible; Streamlit Human Review tab shows discrepancy for `INV_EN_001`, human edits unit price, pipeline re-runs from `data_validate_node` and produces updated report.

---

## 18. Agile Plan & Deliverables

**Sprint cadence**

1. Sprint 1: foundation, ERP mock, rules-driven completeness validation
2. Sprint 2: extraction, translation, multilingual normalization
3. Sprint 3: LangGraph orchestration, MCP integration, HTML reporting
4. Sprint 4: RAG pipeline, HITL workflow, observability, demo hardening

**Backlog / user stories**

1. As an auditor, I want invoices ingested from `data/incoming/` so that email intake is simulated automatically.
2. As an auditor, I want multilingual invoices translated and normalized so that I can review them in English.
3. As a finance reviewer, I want invoice lines checked against ERP records so that discrepancies are surfaced automatically.
4. As a support agent, I want to ask questions over processed invoices so that I can resolve vendor queries quickly.
5. As a human reviewer, I want to correct extracted fields and resume the workflow so that exceptions are handled without restarting the pipeline.

**Demo plan**

1. Show one clean invoice auto-approved end to end.
2. Show one malformed invoice flagged by rules-driven completeness validation.
3. Show one discrepancy invoice paused for HITL review and resumed after correction.
4. Show one RAG question answered from indexed invoice content with triad scores visible.
5. Show LangFuse trace for the same run.

**Slide deck scope**  
Maximum 5 slides, aligned to the capstone deliverable:

1. Problem statement, business context, and objectives
2. Architecture and agent/tool mapping
3. End-to-end workflow and validation logic
4. RAG, HITL, guardrails, and observability
5. Demo results, rubric coverage, and next steps

---

## 19. Evaluation Coverage Map

| Rubric Component | Weight | Implementation |
|---|---|---|
| Functional completeness | 35% | 6 pipeline agents + 6 tools, all formats, all languages, rules.yaml-driven |
| RAG & Agentic integration | 25% | 5 RAG agents, RAG Triad, LangFuse, FastMCP dynamic discovery |
| UI & Human-in-the-loop | 15% | Streamlit 3-tab app, interrupt() HITL, corrections re-trigger validation |
| Code quality & modularity | 15% | Pydantic v2, DRY tool base class, no hardcoded thresholds, early returns |
| Agile methodology | 10% | Phased build order, incremental delivery per phase |
