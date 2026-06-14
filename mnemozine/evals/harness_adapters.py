"""Minimal, self-contained Retriever / CrossReferencer / Extractor adapters.

The §9 metric runners are written against the :mod:`mnemozine.interfaces`
Protocols, so in production the orchestrator injects the *real* ``Retriever``,
``CrossReferencer`` and ``Extractor`` built by their owning modules. But the
EVAL harness must also run **offline against the conftest fakes** (PRD §9 "small
committed gold-set fixture so the harness runs offline against fakes"), and it
may **not** import a sibling module's concrete implementation.

So this module provides tiny, dependency-free adapters that satisfy the relevant
Protocols using only a ``StorageBackend`` (+ optional ``EmbeddingProvider`` /
``LLMProvider``). They are intentionally naive — just enough behavior to drive
the metrics deterministically against the in-memory fake. The real components,
when supplied to the runner, take precedence; these are the fallback so the
harness never needs a live FalkorDB/Ollama/Qwen.
"""

from __future__ import annotations

from collections.abc import Sequence

from mnemozine.interfaces import (
    Classification,
    CrossReference,
    InjectionIndex,
    RetrievalContext,
    RetrievedMemory,
    StorageBackend,
)
from mnemozine.schema.events import IngestEvent
from mnemozine.schema.models import MemoryType, MemoryUnit, Scope, ScopeDecision


class StorageBackedRetriever:
    """A naive :class:`~mnemozine.interfaces.Retriever` over a StorageBackend.

    Delegates scoped retrieval straight to ``StorageBackend.scoped_query`` and
    records access on the deliberate read paths (``scoped_retrieve`` / ``recall``)
    per the FR-MNT-3 access-recording contract. ``build_index`` is provided for
    completeness but is not exercised by the §9 metrics here.
    """

    def __init__(self, storage: StorageBackend) -> None:
        self._storage = storage

    async def scoped_retrieve(
        self, query: str, context: RetrievalContext, *, top_k: int = 10
    ) -> list[RetrievedMemory]:
        scopes = context.scopes or [Scope.global_()]
        results = await self._storage.scoped_query(
            query,
            scopes,
            entities=list(context.entities) or None,
            top_k=top_k,
        )
        for r in results:
            await self._storage.record_access(r.memory.id)
        return results

    async def recall(
        self, query: str, scope: Scope | None = None, *, top_k: int = 10
    ) -> list[RetrievedMemory]:
        scopes = [scope] if scope is not None else [Scope.global_()]
        results = await self._storage.scoped_query(query, scopes, top_k=top_k)
        for r in results:
            await self._storage.record_access(r.memory.id)
        return results

    async def build_index(
        self, context: RetrievalContext, *, token_budget: int | None = None
    ) -> InjectionIndex:
        scopes = context.scopes or [Scope.global_()]
        results = await self._storage.scoped_query(
            context.recent_text or "",
            scopes,
            entities=list(context.entities) or None,
            top_k=10,
        )
        # Category-split contract: "preferences" are global-scope memories; the
        # counts key on the controlled scope decision rather than the old type.
        globals_ = [r for r in results if r.memory.scope_decision is ScopeDecision.GLOBAL]
        text = "; ".join(r.memory.content for r in globals_[:3])
        return InjectionIndex(
            text=text,
            token_estimate=len(text.split()),
            global_count=len(globals_),
            project_count=sum(
                1 for r in results if r.memory.scope_decision is ScopeDecision.PROJECT
            ),
            entity_tags=list(context.entities),
        )


class GraphCrossReferencer:
    """A naive :class:`~mnemozine.interfaces.CrossReferencer` over a StorageBackend.

    Surfaces ``idea_seed`` memories that share at least one active entity with the
    working context (the explainable shared-entity path of FR-RET-6). Each
    surfaced connection carries a human-readable ``reason`` listing the shared
    entities. Dismissals are delegated to the backend's suppression store so they
    survive across calls (R2). Good enough to score cross-reference precision
    offline; the real engine adds weighting + a vector fallback.
    """

    def __init__(self, storage: StorageBackend) -> None:
        self._storage = storage

    async def find_related(
        self, context: RetrievalContext, *, max_suggestions: int | None = None
    ) -> list[CrossReference]:
        cap = max_suggestions if max_suggestions is not None else 2
        ctx_entities = {e.lower() for e in context.entities}
        context_key = context.project or "global"
        out: list[CrossReference] = []
        async for m in self._storage.iter_memories(active_only=True):
            if not m.cross_ref_candidate:
                continue
            shared = sorted({e for e in m.entities if e.lower() in ctx_entities})
            if not shared:
                continue
            if await self._storage.is_suppressed(m.id, context_key):
                continue
            score = len(shared) / max(len(ctx_entities), 1)
            out.append(
                CrossReference(
                    memory=m,
                    score=score,
                    reason=f"shares entities: {', '.join(shared)}",
                    shared_entities=shared,
                )
            )
        out.sort(key=lambda c: c.score, reverse=True)
        return out[:cap]

    async def suppress(self, memory_id: str, context_key: str) -> None:
        await self._storage.record_suppression(memory_id, context_key)


class KeywordExtractor:
    """A naive :class:`~mnemozine.interfaces.Extractor` for the classify path.

    Only ``classify`` is meaningfully implemented (the §9 classifier-accuracy
    path, R1): it labels a statement ``project_fact`` when it reads as
    project-specific ("this project", "the <proj> project", a version pin, a
    concrete datastore/runtime choice) and ``preference`` when it reads as a
    durable cross-project preference ("I prefer/always/like ..."). This is a
    deliberately simple heuristic so the harness has a working classifier offline;
    the real extractor uses the LLM. ``extract`` raises — the harness only needs
    ``classify``.
    """

    _PREF_MARKERS = (
        "i prefer",
        "i always",
        "i like",
        "i favor",
        "i avoid",
        "prefers ",
        "i use ",
        "i format ",
    )
    _FACT_MARKERS = (
        "this project",
        "the project",
        " pins ",
        " uses ",
        " targets ",
        " deploys ",
        "datastore",
    )

    async def extract(self, chunk: Sequence[IngestEvent]) -> list[MemoryUnit]:
        raise NotImplementedError("KeywordExtractor only supports classify() for the eval path")

    async def classify(self, statement: str, context: RetrievalContext) -> Classification:
        low = statement.lower()
        is_fact = any(m in low for m in self._FACT_MARKERS)
        is_pref = any(m in low for m in self._PREF_MARKERS)
        # A version pin or a named project + concrete tech reads as a fact even
        # if phrased with "I use ...".
        if is_fact and not (
            is_pref
            and not any(m in low for m in ("this project", "the project", " pins ", "datastore"))
        ):
            mtype = MemoryType.PROJECT_FACT
            scope = Scope.project(context.project) if context.project else Scope.global_()
        elif is_pref:
            mtype = MemoryType.PREFERENCE
            scope = Scope.global_()
        else:
            # Default: ambiguous statements lean preference (global) unless a
            # project is in context.
            mtype = MemoryType.PREFERENCE
            scope = Scope.global_()
        # Cheap entity guess: alpha tokens longer than 3 chars, de-duped.
        entities = sorted(
            {t.strip(".,:;()") for t in low.split() if t.strip(".,:;()").isalpha() and len(t) > 3}
        )[:6]
        # Map the heuristic's legacy type onto the category-split contract via the
        # documented MemoryType migration helpers.
        return Classification(
            scope_decision=mtype.scope_decision,
            scope=scope,
            category=mtype.category,
            cross_ref_candidate=mtype.is_cross_ref,
            entities=entities,
            confidence=0.7,
        )
