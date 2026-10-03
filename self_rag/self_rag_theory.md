# Self-RAG — Theory & Lecture Notes

This document is the written companion to the `self_rag/` notebooks (files 1–8). It explains **why** each piece exists, **what problem it solves**, and **how the code implements it**, in the order the project was built.

---

## 1. The Goal

Plain RAG has one blind spot: it always trusts its own retrieval and always trusts its own generation. It never asks:

- "Do I even need to retrieve anything for this question?"
- "Are the documents I retrieved actually relevant?"
- "Is my answer actually grounded in those documents, or did the model just make something up?"
- "Even if the answer is grounded, does it actually *answer* the question?"
- "If retrieval failed, should I try searching differently — or go to the web — instead of giving up?"

**Self-RAG** is the family of techniques that adds self-reflection/self-critique steps *inside* the RAG pipeline so the system checks its own work at each stage instead of blindly chaining retrieve → generate.

We built this up deliberately, one capability at a time, rather than writing the final graph in one shot — the same way CRAG was built. Each file adds exactly one new piece of reasoning on top of the previous file's graph, so you can see precisely which problem each node is solving.

### How we divided it into parts

| File | Capability added | Question it answers |
|---|---|---|
| 1 | Retrieval gate | "Do I need to retrieve at all?" |
| 2 | Relevance filter | "Which retrieved docs are actually relevant?" |
| 3 | Context-grounded generation + fallback | "Answer only from relevant docs; what if there are none?" |
| 4 | IsSUP (support check) | "Is the answer actually grounded in the context?" |
| 5 | Verify–revise loop | "If not grounded, fix the answer and re-check — with a retry cap" |
| 6 | IsUSE (usefulness check) | "Even if grounded, does the answer actually address the question?" |
| 7 | Query rewrite loop | "If not useful, rewrite the *search query* and retrieve again — with a retry cap" |
| 8 (separate branch) | Web-search fallback | "If internal docs never become relevant, fall back to the web instead of giving up" |

Files 1→7 form one continuous chain — each file's graph is the previous file's graph plus one more node/edge. File 8 is a **separate branch off of file 3/7's ideas**: it drops the IsSUP/IsUSE grounding-verification loop entirely and instead solves the "no relevant docs" problem by searching the web, rather than by revising the answer or rewriting the retrieval query against the same closed document set.

### The end-to-end picture (file 7 — the full verify/revise/rewrite chain)

```
                    ┌─────────────────┐
                    │  decide_retrieval │
                    └────────┬─────────┘
                 should_retrieve?  │
          ┌────────────no──────────┴─────yes───────────┐
          ▼                                             ▼
  ┌────────────────┐                             ┌───────────┐
  │ generate_direct │                             │  retrieve │◄──────────────┐
  └────────┬────────┘                             └─────┬─────┘               │
           │                                             ▼                    │
          END                                      ┌─────────────┐            │
                                                     │ is_relevant │            │
                                                     └──────┬──────┘            │
                                        relevant_docs exist? │                  │
                                 ┌───────────no───────────────┴────yes───┐       │
                                 ▼                                       ▼       │
                      ┌────────────────────┐                ┌──────────────────────┐
                      │  no_relevant_docs   │                │ generate_from_context │
                      └──────────┬──────────┘                └──────────┬────────────┘
                                 ▼                                       ▼
                                END                                 ┌────────┐
                                                                     │ is_sup │
                                                                     └───┬────┘
                                                        fully_supported?  │
                                        ┌────no, retries<MAX───────────────┴──yes / retries maxed──┐
                                        ▼                                                           ▼
                                ┌────────────────┐                                        ┌───────────────┐
                                │ revise_answer  │──────────► back to is_sup                │ accept_answer │
                                └────────────────┘                                        └──────┬────────┘
                                                                                                    ▼
                                                                                              ┌──────────┐
                                                                                              │  is_use   │
                                                                                              └────┬──────┘
                                                                              useful?               │
                                 ┌──────────no, rewrite_tries<MAX────────────────┴───yes / maxed─────┐
                                 ▼                                                                   ▼
                      ┌───────────────────┐                                                       END
                      │ rewrite_question  │────────► back to retrieve  (loop, capped at MAX_REWRITE_TRIES)
                      └────────────────────┘
                                 │ (maxed out)
                                 ▼
                       no_answer_found → END
```

File 8's graph replaces the right-hand IsSUP/IsUSE machinery with a loop back through retrieval via the web:

```
retrieve → is_relevant ──relevant?──yes──► generate_from_context → END
                 │
                 no
                 ▼
          rewrite_query → web_search ──► is_relevant (loop, UNCAPPED)
```

---

## 2. File-by-file breakdown

### File 1 — `1_self_rag_step1.ipynb`: the retrieval gate

**Problem it solves:** Not every question needs the document corpus. "What is 2+2?" or "Summarize what you just said" don't need retrieval at all — forcing retrieval on every query wastes a call and can inject irrelevant context into a question that didn't need it.

**New state fields:** `question`, `need_retrieval`, `docs`, `answer`.

**New node — `decide_retrieval`:**
```python
class RetrieveDecision(BaseModel):
    should_retrieve: bool
```
An LLM call (structured output) looks at the question and decides whether the internal knowledge base (Northbeam Robotics' PDFs) is actually needed to answer it.

**Conditional edge:** routes to `generate_direct` (if `should_retrieve` is False — answers directly from the model's own knowledge, no documents touched) or `retrieve` (if True — runs the retriever over the vectorstore and stores the raw `docs`).

**Where the graph ends:** Both branches terminate at `END`. At this stage, the `retrieve` branch does not even generate an answer yet — the point of file 1 is purely to prove the gating decision works before building anything on top of it.

**Supporting pipeline (same across every file):** the three Northbeam PDFs (`Company_Profile.pdf`, `Company_Policies.pdf`, `Product_and_Pricing.pdf`) are loaded with `PyPDFLoader`, chunked (`chunk_size=600, chunk_overlap=150`), cleaned, and embedded in batches (`embed_in_batches`, batch size 50, with retry/backoff) using `MistralAIEmbeddings(model="mistral-embed")`. The LLM throughout is `ChatMistralAI(model="ministral-8b-latest", temperature=0)`.

---

### File 2 — `2_self_rag_step2.ipynb`: the relevance filter

**What's new vs. file 1:** Retrieval by itself is not reliable — a vector search can return documents that are topically close but don't actually answer the question (e.g. semantically similar wording, wrong section of the handbook). File 2 adds a per-document relevance check so the system doesn't blindly trust everything the retriever returns.

**New state field:** `relevant_docs: List[Document]`.

**New node — `is_relevant`:**
```python
class RelevanceDecision(BaseModel):
    is_relevant: bool

def is_relevant(state: State):
    relevant_docs = []
    for doc in state["docs"]:
        decision = relevance_llm.invoke(
            is_relevant_prompt.format_messages(
                question=state["question"], document=doc.page_content
            )
        )
        if decision.is_relevant:
            relevant_docs.append(doc)
    return {"relevant_docs": relevant_docs}
```
This runs **one LLM call per retrieved chunk** — each chunk is independently judged relevant or not relevant to the question, and only the ones that pass survive into `relevant_docs`.

**Graph:** `retrieve → is_relevant → END`. Still no generation from the filtered docs yet — file 2's entire job is to prove the filter works in isolation.

---

### File 3 — `3_self_rag_step3.ipynb`: grounded generation + the "nothing survived" fallback

**What's new vs. file 2:** Now that we trust `relevant_docs`, we finally generate an answer — but strictly from that filtered context, never from the model's general knowledge. And we handle the case where the filter rejected everything.

**New state field:** `context: str`.

**New node — `generate_from_context`:** concatenates every doc in `relevant_docs` into one context block and prompts the model to answer *using only that context*, explicitly instructed to say `"No relevant document found."` if the context is empty or doesn't contain the answer.

**New node — `no_relevant_docs`:** a direct fallback message used when the relevance filter returned zero documents at all — no point even calling the generation LLM in that case.

**New routing function — `route_after_relevance`:**
```python
def route_after_relevance(state: State) -> Literal["generate_from_context", "no_relevant_docs"]:
    if state.get("relevant_docs") and len(state["relevant_docs"]) > 0:
        return "generate_from_context"
    return "no_relevant_docs"
```

**Graph:** `retrieve → is_relevant → (generate_from_context | no_relevant_docs) → END`. This is the first file where the retrieval path produces an actual, usable answer end-to-end.

---

### File 4 — `4_self_rag_step4.ipynb`: IsSUP — checking the answer is actually grounded

**What's new vs. file 3:** Telling the model "answer only from context" doesn't guarantee it listened. LLMs still paraphrase, infer, or add qualitative language ("the culture seems generous") that isn't literally present in the source text. File 4 adds a *post-generation* audit of the answer against the context — the first real self-reflection step in the pipeline.

**New state fields:** `issup: Literal["fully_supported", "partially_supported", "no_support"]`, `evidence: List[str]`.

**New node — `is_sup`:**
```python
class IsSUPDecision(BaseModel):
    issup: Literal["fully_supported", "partially_supported", "no_support"]
    evidence: List[str] = Field(default_factory=list)
```
The prompt is deliberately strict: any interpretive or qualitative word in the answer that isn't literally present in the context (e.g. "robust", "generous", "strong culture") downgrades the verdict — even if the underlying fact is roughly true, because the model is being graded on *literal grounding*, not on being roughly right. `evidence` lists the exact context snippets that support (or fail to support) the claim.

**Relevance prompt change:** the `is_relevant` prompt from file 2 is deliberately loosened to topic-level matching rather than exact-answer matching — strict "does this exactly answer the question" checking is now `is_sup`'s job, not the relevance filter's. This avoids double-strict filtering that would reject useful partial context too early.

**Graph (no new routing yet):** `generate_from_context → is_sup → END`. File 4's job is only to prove the audit itself produces sensible verdicts — the test questions in this file are specifically chosen to hit all three verdict categories (a plain fact → fully_supported; a "describe the culture" question → partially_supported; a factual-accuracy trap about which pricing tier includes a dedicated account manager → tests whether the model will hallucinate a benefit for the wrong tier).

---

### File 5 — `5_self_rag_step5.ipynb`: the verify-revise loop

**What's new vs. file 4:** Catching an ungrounded answer is only half the job — file 5 closes the loop by actually fixing it and re-checking, instead of just reporting the verdict and stopping.

**New state field:** `retries: int`.

**New routing function — `route_after_issup`:**
```python
MAX_RETRIES = 10

def route_after_issup(state: State) -> Literal["accept_answer", "revise_answer"]:
    if state.get("issup") == "fully_supported":
        return "accept_answer"
    if state.get("retries", 0) >= MAX_RETRIES:
        return "accept_answer"   # give up gracefully rather than looping forever
    return "revise_answer"
```

**New node — `revise_answer`:**
```python
def revise_answer(state: State):
    out = llm.invoke(revise_prompt.format_messages(
        question=state["question"], answer=state.get("answer", ""), context=state.get("context", "")
    ))
    return {"answer": out.content, "retries": state.get("retries", 0) + 1}
```
The key design detail: `revise_prompt` doesn't just say "try again" — it forces the model into a **strict quote-only output format** (bullet points that must be direct quotes lifted from the context, no original wording allowed). This matters because a loosely-worded "please fix this" instruction would let the model reintroduce the same kind of unsupported language that got it flagged in the first place. Forcing quote-only output makes the revision mechanically incapable of hallucinating again.

**Loop:** `generate_from_context → is_sup → (accept_answer | revise_answer → is_sup → ...)`, capped at `MAX_RETRIES = 10`. Because this is a real loop in the graph, `app.invoke()` now needs `config={"recursion_limit": 80}` — LangGraph's default recursion limit (25) would otherwise trip before the retry cap does, since each retry round-trip through `revise_answer → is_sup` consumes several graph steps.

---

### File 6 — `6_self_rag_step6.ipynb`: IsUSE — checking the answer is actually useful

**What's new vs. file 5:** IsSUP only checks *grounding* — "is every claim backed by the context?" It says nothing about *usefulness* — an answer can be perfectly grounded (every word quoted from real context) and still fail to address what the user actually asked, e.g. because the quoted material is tangentially related but doesn't contain the actual answer. File 6 adds a second, independent self-check for this.

**New state fields:** `isuse: Literal["useful", "not_useful"]`, `use_reason: str`.

**New node — `is_use`:**
```python
class IsUSEDecision(BaseModel):
    isuse: Literal["useful", "not_useful"]
    reason: str = Field(..., description="Short reason in 1 line.")
```
This is a deliberately separate concern from `is_sup`: a grounded-but-useless answer ("The document mentions office locations." when asked "What's the refund policy?") should fail `is_use` even though it may be perfectly grounded in real context.

**New routing function — `route_after_isuse`:**
```python
def route_after_isuse(state: State) -> Literal["END", "no_answer_found"]:
    if state.get("isuse") == "useful":
        return "END"
    return "no_answer_found"
```

**Rewiring:** `route_after_issup`'s `accept_answer` branch, which previously went straight to `END`, is now rewired to flow into `is_use` first. So the final path is: `generate_from_context → is_sup → (revise loop) → accept_answer → is_use → (END | no_answer_found)`.

---

### File 7 — `7_self_rag_step7.ipynb`: the query-rewrite loop

**What's new vs. file 6:** File 5's `revise_answer` loop fixes the *answer* against context that was already retrieved — it never questions whether the *retrieval itself* was the problem. If `is_use` says the answer isn't useful, the real issue might be that the retriever searched for the wrong thing entirely. File 7 adds a second, distinct kind of retry that goes back and searches again with a better query, rather than trying to patch an answer built on the wrong documents.

**New state fields:** `retrieval_query: str`, `rewrite_tries: int`.

**Changed node — `retrieve`:**
```python
def retrieve(state: State):
    q = state.get("retrieval_query") or state["question"]
    return {"docs": retriever.invoke(q)}
```
Now retrieval prefers a rewritten query if one exists, falling back to the raw question otherwise — this is what lets the rewrite loop actually change what gets searched for on each pass.

**New routing function — `route_after_isuse` (replaces file 6's version):**
```python
MAX_REWRITE_TRIES = 3

def route_after_isuse(state: State) -> Literal["END", "rewrite_question", "no_answer_found"]:
    if state.get("isuse") == "useful":
        return "END"
    if state.get("rewrite_tries", 0) >= MAX_REWRITE_TRIES:
        return "no_answer_found"
    return "rewrite_question"
```

**New node — `rewrite_question`:**
```python
class RewriteDecision(BaseModel):
    retrieval_query: str = Field(..., description="Rewritten query optimized for vector retrieval against internal company PDFs.")

def rewrite_question(state: State):
    decision = rewrite_llm.invoke(rewrite_for_retrieval_prompt.format_messages(
        question=state["question"], retrieval_query=state.get("retrieval_query", ""), answer=state.get("answer", "")
    ))
    return {
        "retrieval_query": decision.retrieval_query,
        "rewrite_tries": state.get("rewrite_tries", 0) + 1,
        "docs": [], "relevant_docs": [], "context": "",   # reset so the next pass starts clean
    }
```
Note the explicit reset of `docs`, `relevant_docs`, and `context` — without this, stale data from the failed pass could leak into the next retrieval cycle's relevance/generation steps.

**Full loop:** `rewrite_question → retrieve → is_relevant → generate_from_context → is_sup → (revise loop) → is_use → (END | rewrite_question again, capped at MAX_REWRITE_TRIES | no_answer_found)`. This is a **distinct retry mechanism from file 5's**: file 5 fixes the answer against context that's already fixed; file 7 fixes the search query and starts the whole downstream chain over.

**Test-cell note:** the tutor's original test cell seeded `initial_state["retrieval_query"]` with an unrelated leftover string while asking an unrelated question — almost certainly a copy-paste artifact in the tutor's own code rather than an intentional feature. The Mistral conversion uses `""` as the default instead, so the first retrieval pass always starts from the raw question rather than a stale seeded query.

---

### File 8 — `8_self_rag_web.ipynb`: the web-search fallback (separate branch)

This file is **not** file 8 in the 1→7 chain — it's an alternate ending that starts over from roughly file 3's point and takes a different path entirely. It **drops IsSUP and IsUSE completely** — there is no grounding/usefulness audit here at all. What it adds instead is a way to escape the "no relevant docs" dead-end by going to the open web rather than giving up or rewriting the query against the same closed PDF set.

**State fields (note what's absent):** `question`, `need_retrieval`, `docs`, `relevant_docs`, `context`, `answer`, `web_query` — no `issup`, no `isuse`, no `retries`, no `rewrite_tries`.

**New node — `rewrite_query_node`:**
```python
class WebQuery(BaseModel):
    query: str

rewrite_chain = rewrite_prompt | llm.with_structured_output(WebQuery)

def rewrite_query_node(state: State):
    out = rewrite_chain.invoke({"question": state["question"]})
    return {"web_query": out.query}
```
Turns the original question into a search-engine-style query (short keywords, not a full sentence) — different phrasing needs than the vector-retrieval query rewriting in file 7, since this one is going to Tavily, not the embedding index.

**New node — `web_search_node`:**
```python
tavily = TavilySearchResults(max_results=5)

def web_search_node(state: State):
    q = state.get("web_query") or state["question"]
    results = tavily.invoke({"query": q})
    docs = []
    for r in results or []:
        title = r.get("title", ""); url = r.get("url", "")
        content = r.get("content", "") or r.get("snippet", "")
        text = f"TITLE: {title}\nURL: {url}\nCONTENT:\n{content}"
        docs.append(Document(page_content=text, metadata={"source": "web", "url": url, "title": title}))
    return {"docs": docs}
```
Wraps each Tavily hit into a `Document` with the same shape the PDF retriever produces, so the web results can flow through the exact same `is_relevant` filter and `generate_from_context` node used for the internal documents — no separate generation logic needed for web vs. PDF answers.

**Changed routing — `route_after_relevance`:**
```python
def route_after_relevance(state: State) -> Literal["generate_from_context", "rewrite_query"]:
    if state.get("relevant_docs") and len(state["relevant_docs"]) > 0:
        return "generate_from_context"
    return "rewrite_query"
```
Where file 3's version routed "nothing relevant" to a dead-end (`no_relevant_docs → END`), this version routes it into the web-search escape hatch instead.

**Full loop:** `retrieve → is_relevant → (generate_from_context → END) | (rewrite_query → web_search → is_relevant → ...)`.

**⚠️ Design note — flagged but intentionally not fixed:** unlike file 7's `rewrite_tries` / `MAX_REWRITE_TRIES = 3`, this loop has **no retry counter and no cap at all**. If a question keeps failing the relevance filter even after a web search (e.g. the web search itself returns noise, or the rewritten query is still off-target), the graph will keep looping `rewrite_query → web_search → is_relevant` indefinitely, relying entirely on LangGraph's *default* `recursion_limit` (25) to eventually throw a `GraphRecursionError` rather than failing gracefully with a `no_answer_found`-style message. This mirrors the tutor's own notebook exactly — it's a real gap in the original design, carried through faithfully rather than silently patched, since the instruction was to convert the code, not redesign it.

**Requirement:** this file needs a `TAVILY_API_KEY` environment variable in addition to `MISTRAL_API_KEY`.

---

## 3. Summary table — state fields by file

| Field | Introduced in | Still used in web branch (file 8)? |
|---|---|---|
| `question`, `docs`, `answer` | 1 | ✅ |
| `need_retrieval` | 1 | ✅ |
| `relevant_docs` | 2 | ✅ |
| `context` | 3 | ✅ |
| `issup`, `evidence` | 4 | ❌ dropped |
| `retries` | 5 | ❌ dropped |
| `isuse`, `use_reason` | 6 | ❌ dropped |
| `retrieval_query`, `rewrite_tries` | 7 | ❌ dropped (replaced by `web_query`) |
| `web_query` | 8 | ✅ (file 8 only) |

This table is the clearest way to see the fork: files 1→7 keep accumulating self-checks on top of each other, while file 8 branches off, keeps only the base retrieval/relevance/generation skeleton, and solves the "retrieval came up empty" problem with an external data source instead of more internal self-critique.
