# query_knowledge_hub Modes

`query_knowledge_hub` supports two response modes:

- `contexts`: returns retrieved chunks and citation metadata. This is the default and keeps backward compatibility with existing MCP clients.
- `answer`: performs server-side grounded RAG QA: retrieval, optional rerank, context selection, answer generation, citation binding, guard checks, and structured metadata.

The EnterpriseRAG-Bench dataset used for local experiments is expected at:

```text
G:\data\RAG_dataset\EnterpriseRAG-Bench
```

The dataset contains `questions` and `documents` parquet files. It is suitable for evaluating answer mode because it includes gold answers and expected document IDs.

## Input Schema

```json
{
  "query": "How does hybrid search work in this project?",
  "collection": "project_docs",
  "top_k": 5,
  "mode": "answer",
  "include_sources": true,
  "include_citations": true,
  "answer_style": "bullet",
  "language": "en"
}
```

Fields:

- `mode`: `contexts` or `answer`. Defaults to `contexts`.
- `answer_style`: `concise`, `detailed`, or `bullet`. Used only in answer mode.
- `language`: `auto`, `zh`, or `en`.
- `include_sources`: include structured source list.
- `include_citations`: include structured citation list.

## When To Use Contexts

Use `contexts` when the MCP client should decide how to summarize or reason over retrieved chunks. This is best for agent workflows, code assistants, and clients that already have their own answer generation policy.

Example response metadata:

```json
{
  "mode": "contexts",
  "query": "How does hybrid search work?",
  "collection": "project_docs",
  "retrieval_status": "sufficient",
  "results": [
    {
      "rank": 1,
      "chunk_id": "chunk-1",
      "text": "...",
      "score": 0.82,
      "source": "docs/search.md",
      "page": 3,
      "citation": "[docs/search.md, p.3]"
    }
  ],
  "sources": [
    {
      "source": "docs/search.md",
      "page": 3,
      "chunk_id": "chunk-1"
    }
  ],
  "trace_id": "..."
}
```

## When To Use Answer

Use `answer` when the server should control final QA behavior, citations, refusal policy, and evaluation output.

Example response metadata:

```json
{
  "mode": "answer",
  "query": "How does hybrid search work?",
  "collection": "project_docs",
  "answer": "Hybrid search combines dense and sparse retrieval, then fuses the rankings [C1].",
  "retrieval_status": "sufficient",
  "confidence": "medium",
  "citations": [
    {
      "citation_id": "C1",
      "chunk_id": "chunk-1",
      "source": "docs/search.md",
      "page": 3,
      "snippet": "..."
    }
  ],
  "sources": [
    {
      "source": "docs/search.md",
      "page": 3,
      "chunk_id": "chunk-1"
    }
  ],
  "used_chunk_ids": ["chunk-1"],
  "warnings": []
}
```

No retrieval results:

```json
{
  "mode": "answer",
  "answer": "I could not retrieve enough relevant information from the knowledge base, so I cannot answer this question based on the current corpus.",
  "retrieval_status": "no_results",
  "confidence": "low",
  "citations": [],
  "sources": [],
  "used_chunk_ids": [],
  "warnings": ["NO_RETRIEVAL_RESULTS"]
}
```

## LLM Configuration

Answer mode reuses the existing `llm` settings and provider factory. Configure `config/settings.yaml` with an existing provider such as `openai`, `azure`, `deepseek`, `bailian`, or `ollama`.

```yaml
llm:
  provider: "openai"
  model: "gpt-4o-mini"
  base_url: "${LLM_BASE_URL:-https://api.openai.com/v1}"
  api_key: "${LLM_API_KEY:-}"
  temperature: 0.0
  max_tokens: 4096

response:
  answer_generation:
    enabled: true
    default_mode: "contexts"
    min_contexts: 1
    min_score: 0.2
    max_context_chars: 8000
    default_answer_style: "concise"
    timeout_seconds: 20
    hallucination_guard:
      enabled: true
```

Alibaba Cloud Bailian / Model Studio can be configured through its
OpenAI-compatible endpoint:

```yaml
llm:
  provider: "bailian"
  model: "qwen-plus"
  base_url: "${BAILIAN_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
  api_key: "${DASHSCOPE_API_KEY:-}"
  temperature: 0.0
  max_tokens: 4096
```

Contexts mode does not require an LLM.

## Guard And Citation Strategy

Answer mode asks the LLM to answer only from retrieved contexts and to cite claims with markers such as `[C1]`. A lightweight guard adds warnings when:

- no contexts exist;
- citation markers refer to missing contexts;
- an answer has contexts but no citation marker;
- the answer contains obvious unsupported inference phrases;
- retrieval is `no_results` or `insufficient`.

The first implementation is rule-based and does not use an NLI model.

## Current Limits

- Citation grounding is marker-based and does not prove every sentence is entailed.
- Insufficient retrieval returns a controlled context summary instead of forcing the LLM to speculate.
- LLM failures return a structured fallback answer and warning rather than failing the MCP call.
