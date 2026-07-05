# EnterpriseRAG-Bench 检索评测指南

本文档说明如何在当前项目中运行 EnterpriseRAG-Bench 的 document-level retrieval evaluation。该评测关注检索阶段是否召回了正确的 `dsid_xxx` document，不等同于官方的 end-to-end answer correctness evaluation。

## 数据准备

1. 下载 EnterpriseRAG-Bench。

   推荐放在本地固定路径，例如：

   ```powershell
   G:\data\RAG_dataset\EnterpriseRAG-Bench
   ```

2. 确认 questions 数据。

   新增脚本优先支持 `questions.jsonl`：

   ```text
   EnterpriseRAG-Bench/questions.jsonl
   ```

   如果本地是 HuggingFace parquet 结构，也可以直接传入数据集根目录，脚本会自动尝试读取：

   ```text
   EnterpriseRAG-Bench/data/questions/test.parquet
   ```

3. 解压或导出 documents。

   当前 `scripts/ingest.py` 面向文件摄取。如果你的 EnterpriseRAG-Bench documents 是 parquet，需要先把每条 document 导出为独立文本文件，并确保文件名或路径中包含 `doc_id`，例如：

   ```text
   exported_documents/dsid_0dd4dcc9.txt
   ```

   评测脚本会从检索结果的 `metadata.source_path`、`metadata.source`、`metadata.file_path`、`metadata.source_file`、`metadata.path`、`chunk_id` 或 `id` 中提取 `dsid_xxx`，并与 `expected_doc_ids` 对齐。

## 摄取文档

将导出的 documents 摄取到 collection，例如：

```powershell
python scripts/ingest.py `
  --path G:\data\RAG_dataset\EnterpriseRAG-Bench\exported_documents `
  --collection enterprise_rag `
  --config config/settings.yaml
```

如果只是检查将要处理的文件，可以先运行 dry-run：

```powershell
python scripts/ingest.py `
  --path G:\data\RAG_dataset\EnterpriseRAG-Bench\exported_documents `
  --collection enterprise_rag `
  --dry-run
```

## 运行评测

快速检查脚本参数：

```powershell
python scripts/run_enterprise_rag_eval.py --help
```

使用本地数据集根目录运行小样本评测：

```powershell
python scripts/run_enterprise_rag_eval.py `
  --questions-file G:\data\RAG_dataset\EnterpriseRAG-Bench `
  --config config/settings.yaml `
  --collection enterprise_rag `
  --top-k 10 `
  --modes dense bm25 hybrid hybrid_rerank `
  --max-questions 20 `
  --markdown
```

如果已经导出了 `questions.jsonl`，也可以直接指定该文件：

```powershell
python scripts/run_enterprise_rag_eval.py `
  --questions-file G:\data\RAG_dataset\EnterpriseRAG-Bench\questions.jsonl `
  --config config/settings.yaml `
  --collection enterprise_rag `
  --top-k 10 `
  --candidate-k 20 `
  --modes dense bm25 hybrid hybrid_rerank `
  --markdown
```

## 过滤参数

只评测指定 `question_type`：

```powershell
python scripts/run_enterprise_rag_eval.py `
  --questions-file G:\data\RAG_dataset\EnterpriseRAG-Bench `
  --collection enterprise_rag `
  --question-types basic semantic constrained `
  --max-questions 50
```

只评测指定 `source_types`：

```powershell
python scripts/run_enterprise_rag_eval.py `
  --questions-file G:\data\RAG_dataset\EnterpriseRAG-Bench `
  --collection enterprise_rag `
  --source-types github gmail linear
```

默认会跳过没有 `expected_doc_ids` 的问题。如需保留这类问题用于结果检查，可以使用：

```powershell
python scripts/run_enterprise_rag_eval.py `
  --questions-file G:\data\RAG_dataset\EnterpriseRAG-Bench `
  --collection enterprise_rag `
  --no-skip-no-ground-truth
```

## 输出结果

JSON 报告会写入：

```text
eval/results/{timestamp}_enterprise_rag.json
```

传入 `--markdown` 时，会额外写入：

```text
eval/results/{timestamp}_enterprise_rag.md
```

Markdown 汇总表包含：

```text
Mode | Evaluated | Recall@K | Precision@K | MRR@K | NDCG@K | Hit@K | Avg Latency ms
```

JSON 中每个 query 会保留：

- `question_id`
- `question_type`
- `source_types`
- `expected_doc_ids`
- `retrieved_doc_ids`
- `metrics`
- `error`
- `results`

其中 `results` 复用项目现有 `_serialise_result`，保留每个 retrieved chunk 的 `chunk_id`、`score`、`metadata` 和 `text_preview`。

## 指标说明

当前脚本计算 document-level IR metrics：

- `recall@k`
- `precision@k`
- `mrr@k`
- `ndcg@k`
- `hit@k`
- `latency_ms`

因为同一个 document 可能被切成多个 chunk，脚本会先从 retrieved chunks 中提取 `dsid_xxx`，再按顺序去重，最后与 `expected_doc_ids` 对齐。因此这里评估的是 document recall，不是 chunk-level recall。

## 边界说明

- 该脚本不会调用真实 LLM Judge，也不会评估 `gold_answer` 的 semantic correctness。
- 如果某个 query 检索失败，评测不会中断，错误会写入该 query 的 `error` 字段。
- 如果 collection 为空，脚本会在控制台输出 warning，并在报告中记录 `collection_record_count`。
- 如果需要官方 end-to-end evaluation，后续还需要基于项目的 answer generation 生成 `answers.jsonl`，再接入官方 judge 流程。

## 测试

轻量测试不依赖真实 Embedding API，也不依赖 ChromaDB 中已有数据：

```powershell
pytest tests/test_enterprise_rag_eval.py -q
```
