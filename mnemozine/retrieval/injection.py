"""Injection helpers for the Claude Code hooks (FR-RET-3 / FR-RET-5).

Two entry points, both built on :class:`ScopedRetriever.build_index` so they
share the same ~500-token budget envelope and the same compact-index format
contract:

* :func:`session_start_injection` (FR-RET-3) — fired on ``SessionStart``. Detects
  the working context from cwd / manifests / git-remote / recent turns and builds
  the full SessionStart index.
* :func:`mid_session_injection` (FR-RET-5) — fired on ``UserPromptSubmit`` as the
  conversation moves into new topics. It is finer-grained and *smaller* per the
  FR-RET-3 contract ("mid-session injections draw from the same budget envelope
  and should be finer-grained and even smaller per-injection"): it derives
  entities from the new prompt, uses the prompt as the retrieval query, and caps
  the budget to a fraction of the SessionStart budget.

Both return an :class:`~mnemozine.interfaces.InjectionIndex` whose ``text`` is
already truncated to budget and clearly delimited (advisory background, not a
directive). A hook script prints ``index.text`` when non-trivial.
"""

from __future__ import annotations

from mnemozine.config import Settings, get_settings
from mnemozine.interfaces import InjectionIndex, RetrievalContext
from mnemozine.retrieval.context import detect_context, entities_from_text
from mnemozine.retrieval.retriever import ScopedRetriever

# Mid-session injections are "even smaller per-injection" (FR-RET-3 contract):
# a fraction of the SessionStart budget so they stay lightweight as topics shift.
_MID_SESSION_BUDGET_FRACTION = 0.5


async def session_start_injection(
    retriever: ScopedRetriever,
    *,
    cwd: str | None = None,
    git_remote: str | None = None,
    recent_text: str | None = None,
    extra_entities: list[str] | None = None,
) -> InjectionIndex:
    """Build the SessionStart proactive injection index (FR-RET-3).

    Detects the working context (cwd / ``Cargo.toml`` / ``package.json`` /
    git-remote / recent turns) then delegates to
    :meth:`ScopedRetriever.build_index`, which enforces the full
    ``inject.token_budget`` (~500). The result is the compact index the
    ``SessionStart`` hook injects.
    """

    context = detect_context(
        cwd=cwd,
        git_remote=git_remote,
        recent_text=recent_text,
        extra_entities=extra_entities,
    )
    return await retriever.build_index(context)


async def mid_session_injection(
    retriever: ScopedRetriever,
    prompt: str,
    *,
    project: str | None = None,
    extra_entities: list[str] | None = None,
    settings: Settings | None = None,
    base_context: RetrievalContext | None = None,
) -> InjectionIndex:
    """Build a finer-grained, smaller mid-session injection (FR-RET-5).

    Derives entities from the new ``prompt`` (lexically, offline), scopes to the
    session ``project`` + global, and builds an index under a *reduced* budget
    (a fraction of ``inject.token_budget``) so mid-session injections stay even
    smaller than SessionStart per the FR-RET-3 contract.

    ``base_context`` may be supplied to reuse a SessionStart-derived context
    (scopes/project) — the prompt's entities are merged into it. Otherwise a
    context is composed from ``project`` + the prompt's entities.
    """

    cfg = settings or get_settings()
    budget = max(1, int(cfg.inject.token_budget * _MID_SESSION_BUDGET_FRACTION))

    prompt_entities = entities_from_text(prompt)
    if extra_entities:
        prompt_entities.extend(e.lower() for e in extra_entities)

    if base_context is not None:
        merged_entities = list(dict.fromkeys([*base_context.entities, *prompt_entities]))
        context = RetrievalContext(
            project=base_context.project,
            scopes=list(base_context.scopes),
            entities=merged_entities,
            recent_text=prompt,
        )
    else:
        # Compose a fresh context from the project + prompt entities. Reuse
        # detect_context's scope composition by passing no cwd reads (entities
        # come straight from the prompt) — but detect_context reads the fs, so
        # build the context directly here to stay prompt-scoped and side-effect
        # free for mid-session.
        from mnemozine.schema.models import Scope

        scopes = []
        if project:
            scopes.append(Scope.project(project))
        scopes.append(Scope.global_())
        context = RetrievalContext(
            project=project,
            scopes=scopes,
            entities=prompt_entities,
            recent_text=prompt,
        )

    return await retriever.build_index(context, token_budget=budget)
