"""Recall playground route (PRD §4.5 / §6 POST /api/recall).

The precision-debugging tool: runs the **real** ``Retriever.recall`` for the
ranked, scored results and ``Retriever.build_index`` for the ~500-token
SessionStart injection preview (FR-RET-3/4) — both through the existing scoped
retriever (no new retrieval path). The query's scope (``'global'`` /
``'project:<id>'`` / a bare project id, or ``None`` for the default composed
scope) is threaded into both calls so the operator sees exactly what a session in
that scope would retrieve and what would be injected.

Each result carries a short ``why`` derived from the lexical/entity overlap with
the query so the screen can show *why it surfaced*.
"""

from __future__ import annotations

from fastapi import APIRouter

from mnemozine.interfaces import RetrievalContext, RetrievedMemory
from mnemozine.schema.models import Scope
from mnemozine.web.deps import RetrieverDep, SettingsDep
from mnemozine.web.routes._read import memory_to_list_item, scope_to_obj
from mnemozine.web.schemas import (
    InjectionIndexPreview,
    RecallRequest,
    RecallResponse,
    ScoredMemory,
)

router = APIRouter(prefix="/api/recall", tags=["recall"])

# Tiny stopword set so the why-note keys on meaningful query terms only.
_STOPWORDS = frozenset(
    {"the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is", "are", "do", "i"}
)


def _why(query: str, memory: RetrievedMemory) -> str:
    """Explain why a unit surfaced: shared entities first, then lexical overlap.

    Purely lexical (offline) so the playground stays fast and dependency-free; the
    score itself comes from the retriever's real ranking.
    """

    q_terms = {w for w in query.lower().split() if w and w not in _STOPWORDS}
    shared_entities = [e for e in memory.memory.entities if e.lower() in q_terms]
    if shared_entities:
        return "shares entities: " + ", ".join(sorted(set(shared_entities)))
    content_terms = {w.strip(".,;:!?()[]") for w in memory.memory.content.lower().split()}
    overlap = sorted(q_terms & content_terms)
    if overlap:
        return "matched terms: " + ", ".join(overlap[:5])
    return f"semantic match (score {memory.score:.2f})"


def _context_for(scope: Scope | None, query: str) -> RetrievalContext:
    """Build the working context the index preview is computed against.

    Mirrors the SessionStart path: the recall query is the recent text (the most
    specific signal) and the scope, when explicit, composes the searched scopes.
    """

    scopes = [scope] if scope is not None else []
    project = scope.project_id if scope is not None and not scope.is_global else None
    return RetrievalContext(
        project=project,
        scopes=scopes,
        recent_text=query,
    )


@router.post("", response_model=RecallResponse, summary="Run recall() + index preview")
async def run_recall(
    req: RecallRequest, retriever: RetrieverDep, settings: SettingsDep
) -> RecallResponse:
    """Interactive recall + SessionStart index preview (PRD §4.5).

    Runs the live ``recall(query, scope?, top_k)`` for the ranked results, then —
    when requested — ``build_index`` for the ~500-token injection preview. Both go
    through the existing retriever so the playground reflects production behavior.
    """

    scope = scope_to_obj(req.scope)
    retrieved = await retriever.recall(req.query, scope, top_k=req.top_k)
    results = [
        ScoredMemory(
            memory=memory_to_list_item(r.memory),
            score=r.score,
            why=_why(req.query, r),
        )
        for r in retrieved
    ]

    index_preview: InjectionIndexPreview | None = None
    if req.include_index_preview:
        index = await retriever.build_index(_context_for(scope, req.query))
        index_preview = InjectionIndexPreview(
            text=index.text,
            token_estimate=index.token_estimate,
            token_budget=settings.inject.token_budget,
            global_count=index.global_count,
            project_count=index.project_count,
            cross_ref_hints=list(index.cross_ref_hints),
            entity_tags=list(index.entity_tags),
        )

    return RecallResponse(
        query=req.query,
        scope=req.scope,
        results=results,
        index_preview=index_preview,
    )


__all__ = ["router"]
