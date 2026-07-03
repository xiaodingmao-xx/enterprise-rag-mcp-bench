"""Lightweight rule-based question generation for metadata enrichment."""

from __future__ import annotations


def generate_rule_based_questions(
    *,
    title: str,
    summary: str = "",
    entities: list[str] | None = None,
    tags: list[str] | None = None,
    enabled: bool = True,
    max_questions: int = 3,
) -> list[str]:
    """Generate simple candidate user questions from metadata fields."""

    if not enabled or max_questions <= 0:
        return []

    subject = (title or "").strip()
    if not subject:
        for item in (entities or []) + (tags or []):
            if str(item).strip():
                subject = str(item).strip()
                break
    if not subject:
        return []

    subject = subject[:60].strip()
    lower_context = f"{title} {summary} {' '.join(tags or [])}".lower()

    templates = [f"{subject} 的作用是什么？"]
    if any(keyword in lower_context for keyword in ("config", "配置", "部署", "deployment", "docker")):
        templates.append(f"如何配置 {subject}？")
    if any(keyword in lower_context for keyword in ("error", "错误", "报错", "http", "exception")):
        templates.append(f"{subject} 报错如何排查？")
    if len(templates) < max_questions:
        templates.append(f"{subject} 有哪些关键注意事项？")

    output: list[str] = []
    seen: set[str] = set()
    for question in templates:
        if question in seen:
            continue
        seen.add(question)
        output.append(question)
        if len(output) >= max_questions:
            break
    return output
