# From RAG to Corrective RAG (CRAG) — A Step-by-Step Build Log

**Stack used:** LangChain + LangGraph + Mistral (`ministral-8b-latest`, `mistral-embed`) + FAISS + Tavily
**Corpus:** single PDF (`software_project_man_book.pdf`), loaded with `PyPDFLoader`
**Files:** `1_basic_rag.ipynb` → `6_ambiguous.ipynb`

This is meant to be the thing you re-read in six months when you need to build another RAG-family project and just want your brain re-loaded, not the tutor's.

---

## 1. What we actually needed, and why we broke it into 6 pieces

Plain RAG has one silent failure mode: **it always answers, even when the retrieved chunks are garbage.** The model will confidently hallucinate an answer from irrelevant context because nothing in the pipeline ever asks "was what I retrieved actually good enough?"

CRAG (Corrective RAG) is the fix: add a **grading step** after retrieval, and route differently depending on the grade — trust the internal docs, go fetch better knowledge from the web, or blend both. That's the entire idea in one sentence. Everything else is implementation detail.

We never built that in one shot. We built it as a *staircase*, each file adding exactly one new capability on top of the last, so at every step there was one clear thing to test and reason about:

| File | Adds | Core question it answers |
|---|---|---|
| 1. `basic_rag` | retrieve → generate | Does plain RAG work at all on this corpus? |
| 2. `retrieval_refinement` | decompose → filter → recompose | Can we shrink noisy context down to only the useful sentences? |
| 3. `retrieval_evaluator` | per-doc LLM scoring + CORRECT/INCORRECT/AMBIGUOUS verdict | Is the retrieved context actually good? |
| 4. `web_search_refinement` | real Tavily web search on INCORRECT | What do we do when internal docs are bad? |
| 5. `query_rewrite` | LLM rewrites the question into a search-engine query before web search | How do we search the web *well*, not just literally? |
| 6. `ambiguous` | AMBIGUOUS now also triggers web search, and refine blends both sources | What do we do about the messy middle case? |

This "add one capability, test it, then add the next" approach is the actual transferable skill here — not the specific CRAG code. Any agentic pipeline (not just RAG) is built this way: get the dumbest version running end-to-end first, then bolt on judgment/correction/retry logic one node at a time.

---

## 2. The mental model (matches your tutor's diagram)

```
Retrieval:       x (question) ──► retriever ──► d1, d2, ... (docs)

Knowledge        ┌─────────────────────────────────────────┐
Correction:       Retrieval Evaluator asks: "is this correct for x?"
                  │
        ┌─────────┼─────────┐
     CORRECT   AMBIGUOUS  INCORRECT
        │         │  \       │
        │         │   \      │
        ▼         ▼    ▼     ▼
   Knowledge   Knowledge   Knowledge
   Refinement  Refinement  Searching
   (decompose/  (both)     (rewrite → web
    filter/                 search → select)
    recompose)

Generation:   x + k_internal   x + k_in + k_ex   x + k_external
                    └───────────────┬───────────────┘
                                Generator
```

- **k_in** = refined internal knowledge (from your own PDF, via decompose→filter→recompose)
- **k_ex** = refined external knowledge (from the web, via rewrite→search→refine)
- CORRECT uses k_in only, INCORRECT uses k_ex only, AMBIGUOUS uses **both concatenated**

That three-way split at the bottom of the diagram is exactly what file 6's `refine()` function implements:

```python
if state.get("verdict") == "CORRECT":
    docs_to_use = state["good_docs"]
elif state.get("verdict") == "INCORRECT":
    docs_to_use = state["web_docs"]
else:  # AMBIGUOUS
    docs_to_use = state["good_docs"] + state["web_docs"]
```

Whenever you build a CRAG-style system in the future, this diagram *is* the plan. Draw it first, then write nodes that match its boxes.

---

## 3. Step-by-step build

### File 1 — Basic RAG (the floor)

```
retrieve → generate
```

- `PyPDFLoader` → `RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)` → `MistralAIEmbeddings` → `FAISS` → `retriever.as_retriever(k=4)`
- One node retrieves, one node generates. No judgment, no correction.
- **Why start here:** if this doesn't work — bad chunking, bad embeddings, corpus doesn't actually contain the answer — nothing built on top of it will work either. Always get the naive version running and sanity-checked before adding intelligence.

### File 2 — Retrieval refinement (compress before you generate)

Adds a **decompose → filter → recompose** step between retrieve and generate:

1. **Decompose**: regex-split the concatenated retrieved context into individual sentences (`decompose_to_sentences`).
2. **Filter**: run each sentence through an LLM classifier (`KeepOrDrop(keep: bool)`) asking "does this sentence help answer the question?"
3. **Recompose**: join the sentences that survived back into `refined_context`.

This is "strip-level" refinement — instead of trusting the entire retrieved chunk (which might be 80% irrelevant padding around one useful sentence), you throw away everything that isn't pulling weight. It's cheap, it's per-sentence, and it directly reduces hallucination surface area for the generator.

**Why this file exists on its own, before evaluation:** refinement and evaluation are two *separate* concerns. Refinement asks "of what I have, what's useful?" Evaluation (file 3) asks "do I even have enough?" Conflating them into one step makes debugging much harder — you can't tell if bad output is a retrieval problem or a filtering problem. Building them separately, then composing them, is the right instinct.

### File 3 — Retrieval evaluator (the actual "C" in CRAG)

This is where CRAG genuinely starts. Adds:

- `DocEvalScore(score: float, reason: str)` — a Pydantic schema for structured LLM output
- `eval_each_doc_node`: scores **every retrieved chunk individually** (not the whole context at once — this matters, see §4)
- Threshold logic: `UPPER_TH = 0.7`, `LOWER_TH = 0.3`
  - any chunk `> 0.7` → verdict `CORRECT`
  - all chunks `< 0.3` → verdict `INCORRECT`
  - otherwise → verdict `AMBIGUOUS`
- `good_docs` = chunks that scored `> LOWER_TH` (kept even in the ambiguous case, as "weakly relevant")
- Routing: CORRECT → refine → generate; INCORRECT → a `fail` placeholder (not yet real); AMBIGUOUS → an `ambiguous` end node (not yet real either)

At this stage, INCORRECT and AMBIGUOUS just dead-end with a message — the "correction" isn't built yet. That's deliberate: get the *evaluator* right and testable in isolation (does it actually classify well?) before building what happens after a bad verdict.

### File 4 — Web search refinement (the correction mechanism)

INCORRECT now does something instead of failing: `web_search_node` calls Tavily with the raw question, wraps each result into a `Document`, and stores them in `web_docs`. `refine()` is updated to pick `good_docs` if CORRECT, else `web_docs` — same decompose/filter/recompose logic from file 2, now source-agnostic.

This is the moment the system stops being "RAG with a grader bolted on" and becomes genuinely *corrective*: a bad verdict actually triggers a repair action, not just a different error message.

### File 5 — Query rewrite (search well, not literally)

Problem: sending the user's raw natural-language question straight to a search API is a weak search query. `"What are the latest changes to the PMBOK guide in 2026?"` is a fine question for an LLM, but a search engine wants keywords, not a sentence.

Adds `rewrite_query_node`: an LLM call with `WebQuery(query: str)` structured output that turns the question into a short (6–14 word) keyword query, and adds a recency hint like `(last 30 days)` if the question implies it. `web_search_node` now searches on `web_query`, falling back to the raw question if the rewrite came back empty.

Routing: INCORRECT now goes `rewrite_query → web_search → refine → generate` instead of straight to `web_search`.

**General lesson:** whenever you hand a raw user utterance to an external tool (search API, SQL, another agent), consider whether that tool actually wants natural language or a more structured query — and if not, put a small rewriting node in front of it. This pattern (rewrite-then-call) shows up constantly in agentic pipelines.

### File 6 — Ambiguous handling (closing the loop)

Two changes:

1. **Routing collapses**: `route_after_eval` now only distinguishes CORRECT vs. not-CORRECT. Both INCORRECT and AMBIGUOUS go through `rewrite_query → web_search → refine`.
2. **`refine()` becomes verdict-aware**, blending sources exactly as the diagram shows: CORRECT uses `good_docs` only, INCORRECT uses `web_docs` only, AMBIGUOUS uses **both**.

This is the complete CRAG pattern. Nothing dead-ends anymore — every verdict produces a real answer, backed by the most trustworthy knowledge available for that verdict.

---

## 4. LangGraph: how it's actually being used here

Every file in this series uses exactly three LangGraph building blocks. Once these three click, you can build almost any agentic graph.

### a) `State` — a `TypedDict`, not a class instance

```python
class State(TypedDict):
    question: str
    docs: List[Document]
    good_docs: List[Document]
    verdict: str
    reason: str
    strips: List[str]
    kept_strips: List[str]
    refined_context: str
    web_query: str
    web_docs: List[Document]
    answer: str
```

This is the **shared memory** that flows through every node. Key mental model: **nodes don't mutate state directly — they return a partial dict, and LangGraph merges it into the running state.** Look at any node:

```python
def retrieve_node(state: State) -> State:
    q = state["question"]
    return {"docs": retriever.invoke(q)}   # only returns the field it changed
```

It reads whatever fields it needs from `state`, and returns *only* the fields it's updating. That's why `State` grew one field per file (file 3 added `good_docs`/`verdict`/`reason`, file 4 added `web_docs`, file 5 added `web_query`) — the schema is a running ledger of "everything any node in this graph might need to read or write."

### b) Nodes and edges — the graph is just a flowchart in code

```python
g = StateGraph(State)
g.add_node("retrieve", retrieve_node)
g.add_node("eval_each_doc", eval_each_doc_node)
...
g.add_edge(START, "retrieve")
g.add_edge("retrieve", "eval_each_doc")
```

A node is any function `State -> partial dict`. An edge is just "after this node, run that node." This is deliberately boring — the power isn't in the edges, it's in...

### c) `add_conditional_edges` — this is where "corrective" logic lives

```python
def route_after_eval(state: State) -> str:
    if state["verdict"] == "CORRECT":
        return "refine"
    else:
        return "rewrite_query"

g.add_conditional_edges(
    "eval_each_doc",
    route_after_eval,
    {"refine": "refine", "rewrite_query": "rewrite_query"},
)
```

A conditional edge is a function that looks at the current `State` and returns a *key* naming which node to go to next. This is the single mechanism that turns a linear pipeline into a system with judgment and branching. Any time you want an agent to "decide what to do next based on what happened," this is the primitive you reach for — not an if/else buried inside a node, but a router function LangGraph calls between nodes.

**Rule of thumb for future projects:** if a decision changes *which nodes run next*, it belongs in a conditional-edge router. If a decision only changes *behavior within* one step (like `refine()` picking which docs to use), it can live inside the node itself, reading `state["verdict"]`. Both are used here — know the difference.

---

## 5. Pydantic: how it's actually being used here

Every LLM call in this pipeline that needs to be *machine-readable* (not just prose) uses the same pattern:

```python
class DocEvalScore(BaseModel):
    score: float
    reason: str

doc_eval_chain = doc_eval_prompt | llm.with_structured_output(DocEvalScore)
out = doc_eval_chain.invoke({"question": q, "chunk": d.page_content})
out.score   # a real float, not a string you have to parse
```

`with_structured_output(SomeModel)` tells the LLM provider (Mistral here) to constrain its output to match that schema, and LangChain parses the JSON back into an actual Pydantic object. This is what lets you write `out.score > UPPER_TH` instead of regex-scraping a number out of free text.

Four schemas appear across the series, each solving one narrow decision:

| Schema | Fields | Used for |
|---|---|---|
| `DocEvalScore` | `score: float, reason: str` | grading one retrieved chunk (file 3+) |
| `KeepOrDrop` | `keep: bool` | should this sentence survive filtering? (file 2+) |
| `WebQuery` | `query: str` | rewritten search query (file 5+) |

**Pattern to remember:** every one of these is a *tiny, flat* schema — one or two fields, no nesting. That's intentional and it's why `ministral-8b-latest` (a small, cheap model) is sufficient here — simple structured-output tasks don't need a bigger model. The rule of thumb from other projects in this same tutorial track: escalate to a bigger model (e.g. `ministral-14b-latest`) only when the schema gets genuinely complex — nested objects, lists of objects, many interdependent fields (like a multi-step `Plan` with a list of `Task` objects). A flat bool/float/string schema is squarely in "small model" territory.

**General lesson for future RAG/agent projects:** anywhere you're tempted to ask an LLM a yes/no or a score and then `if "yes" in response.lower()`, replace it with a one-field Pydantic model and `with_structured_output`. It's more reliable, and it composes directly into `if`/routing logic without string parsing.

---

## 6. The reusable checklist

When you next build a RAG-family project from scratch, this is the order of operations:

1. **Get naive RAG working first.** Load → chunk → embed → retrieve → generate, no judgment at all. Confirm the corpus actually contains answerable content.
2. **Add refinement** (decompose → LLM filter → recompose) as its own separate step, independent of any grading. This alone measurably reduces noise in what reaches the generator.
3. **Add an evaluator** that scores retrieved chunks individually (not the blob as a whole) and classifies a verdict from the score distribution (a clear "good enough" threshold + a clear "not good enough" threshold, with a middle bucket for the rest — don't force a binary).
4. **Wire the evaluator's verdict into a conditional edge**, not an if/else buried in a node. Bad verdicts should route to a *real* recovery path (web search, a different retriever, asking the user to clarify) — never a silent placeholder.
5. **If you're handing a query to an external tool** (search API, database, another system), add a rewrite step in front of it rather than passing the raw user question through.
6. **Decide explicitly how each verdict bucket sources its context** (internal only / external only / blended) — don't let this be implicit, write it as one clear `if/elif/else` in the refine step, matching a diagram you can actually draw.
7. **Every LLM decision that feeds control flow** (a score, a boolean, a routing choice) → a small flat Pydantic schema + `with_structured_output`. Never regex a decision out of free text.
8. Keep `State` as a single `TypedDict` that grows by exactly the fields each new capability needs — it doubles as living documentation of what the pipeline tracks.

That's the whole transferable pattern. The specific thresholds (0.7/0.3), the specific corpus, and the specific model names will change project to project — this shape won't.
