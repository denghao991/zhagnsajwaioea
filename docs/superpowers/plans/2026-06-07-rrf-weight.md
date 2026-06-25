# RRF 融合权重支持实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 RRF 合并添加可配置的 vector/BM25 权重，支持通过 `data/config.yaml` 调节两侧检索的贡献比例。

**Architecture:** `rrf_merge()` 增加 `alpha`（vector 权重）和 `beta`（BM25 权重）参数，默认均为 1.0（保持向后兼容）。权重值通过 config.py 从 YAML 加载，app.py 传入。

**Tech Stack:** Python, pytest

---

### Task 1: 修改 rrf_merge 支持权重参数

**Files:**
- Modify: `src/tiny_rag/retrieval/hybrid.py:6-41`
- Test: `tests/test_hybrid.py`

- [x] **Step 1: 写权重测试**

```python
def test_rrf_merge_with_weights():
    """vector 权重 2.0 时排序变化。"""
    vec = [
        _make_result("A", distance=0.3),
        _make_result("B", distance=0.5),
    ]
    bm25 = [
        _make_result("B", distance=0.5),
        _make_result("C", distance=0.5),
    ]
    merged = rrf_merge(vec, bm25, n_results=3, alpha=2.0, beta=1.0)
    assert [r["text"] for r in merged] == ["B", "A", "C"]


def test_rrf_merge_weight_defaults_to_one():
    """不传 alpha/beta 时与之前行为一致。"""
    vec = [_make_result("A"), _make_result("B")]
    bm25 = [_make_result("B"), _make_result("C")]
    default = rrf_merge(vec, bm25, n_results=3)
    explicit = rrf_merge(vec, bm25, n_results=3, alpha=1.0, beta=1.0)
    assert [r["text"] for r in default] == [r["text"] for r in explicit]
```

- [x] **Step 2: 运行测试确认失败** — FAILED (TypeError)

- [x] **Step 3: 实现权重参数** — hybrid.py 增加 alpha/beta 签名

- [x] **Step 4: 运行测试确认通过** — 6/6 PASS

- [x] **Step 5: 提交** — `043b8f2`

---

### Task 2: YAML + config.py 暴露权重配置

**Files:**
- Modify: `data/config.yaml:7-9`
- Modify: `src/tiny_rag/config.py:70-95`

- [x] **Step 1: data/config.yaml 增加 alpha/beta** — `alpha: 1.0, beta: 1.0`

- [x] **Step 2: config.py 增加 `VECTOR_ALPHA`、`BM25_BETA` 全局变量和加载逻辑**

- [x] **Step 3: 运行配置测试确认无回归** — 8/8 PASS

- [x] **Step 4: 提交** — `a02e435`

---

### Task 3: app.py 传入权重到 rrf_merge

**Files:**
- Modify: `src/tiny_rag/app.py:14,199`

- [x] **Step 1: import 添加 VECTOR_ALPHA, BM25_BETA**

- [x] **Step 2: rrf_merge 调用处传入 alpha/beta**

- [x] **Step 3: 全量测试** — 101/101 PASS

- [x] **Step 4: 提交** — `c8ae0a3`
