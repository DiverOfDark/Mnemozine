"""An in-process fake of the Graphiti FalkorDB driver for storage contract tests.

The storage backend (:class:`mnemozine.storage.backend.GraphitiStorageBackend`)
talks to FalkorDB only through ``GraphitiClient.execute_query(cypher, **params)``.
This fake stands in for that seam: it interprets the *specific* Cypher statements
the backend emits against in-memory dict stores and returns FalkorDB-shaped
results (an object with a ``.result_set`` list of row-lists, where node/edge
values are plain dicts of properties).

It is deliberately **not** a general Cypher engine — it recognizes the handful of
statement shapes the backend uses by matching stable substrings. That is enough
to exercise the backend's real (de)serialization, the FR-MNT-1 4-way write
decision, scope/tier/entity filtering, cosine ranking, validity windows, tiering,
entity merge, edges, suppression, and sessions — all with no live FalkorDB or
Ollama. If the backend's Cypher changes shape, these tests fail loudly, which is
the point of a contract test.
"""

from __future__ import annotations

from typing import Any

from mnemozine.storage.cosine import cosine_similarity


class _Result:
    """Minimal FalkorDB QueryResult stand-in: rows live on ``result_set``."""

    def __init__(self, rows: list[list[Any]]) -> None:
        self.result_set = rows


class FakeFalkorDriver:
    """Dict-backed interpreter of the backend's Cypher (see module docstring)."""

    def __init__(self) -> None:
        self.memories: dict[str, dict[str, Any]] = {}
        self.entities: dict[str, dict[str, Any]] = {}
        self.edges: dict[str, dict[str, Any]] = {}
        # Persisted (memory)-[:MNEMOZINE_MENTIONS]->(entity) edges as a set of
        # (memory_id, entity_id) pairs (MERGE set-semantics: idempotent re-assert,
        # repointed in merge_entities).
        self.mentions: set[tuple[str, str]] = set()
        # Weighted entity-entity (entity)-[:MNEMOZINE_CO_MENTIONS]->(entity) edges
        # keyed on (from_id, to_id) -> {weight, shared, ...} (MERGE: idempotent
        # re-assert overwrites weight, never duplicates).
        self.co_mentions: dict[tuple[str, str], dict[str, Any]] = {}
        self.sessions: dict[tuple[str, str], dict[str, Any]] = {}
        self.suppressions: set[tuple[str, str]] = set()
        # Raw-chunk tier (offline re-extraction/reindex), keyed on content_hash
        # to mirror the backend's MERGE idempotency key (FR-ING-5).
        self.raw_chunks: dict[str, dict[str, Any]] = {}
        self.closed = False
        self.queries: list[str] = []

    async def close(self) -> None:
        self.closed = True

    # -- edge topology helpers (legacy-edge reproduction) ---------------------

    @staticmethod
    def _edge_view(e: dict[str, Any]) -> dict[str, Any]:
        """Project an internal edge record to the relation-props dict the backend reads.

        The fake ALWAYS keeps ``from_entity``/``to_entity`` on the internal record
        so it can answer the topology columns (``a.id``/``startNode(r).id`` …). But
        a *legacy* edge (the 2026-06-14 backfill / merge-rewired shape) stored only
        ``{id, relation, weight, valid_from}`` in FalkorDB — so for such an edge the
        relation-props object handed to ``_row_to_edge`` must OMIT from/to to
        reproduce the exact ``KeyError`` condition the read-side fix removes. The
        internal ``_legacy`` marker is itself never part of the FalkorDB shape, so
        it is always dropped from the view.
        """

        view = {k: v for k, v in e.items() if k != "_legacy"}
        if e.get("_legacy"):
            view.pop("from_entity", None)
            view.pop("to_entity", None)
        return view

    def add_legacy_edge(
        self,
        *,
        edge_id: str,
        from_entity: str,
        to_entity: str,
        relation: str = "relates",
        weight: float = 1.0,
        valid_from: str | None = None,
        valid_to: str | None = None,
    ) -> dict[str, Any]:
        """Insert a LEGACY-style edge (no from/to PROPS, only known via topology).

        Mirrors the 593-vs-2316 live split: the fake keeps the real endpoints
        internally (so it answers the topology columns), but marks the edge so the
        relation-props object the backend reads back omits ``from_entity`` /
        ``to_entity`` — exactly the shape that ``KeyError``'d before the fix.
        """

        e: dict[str, Any] = {
            "id": edge_id,
            "from_entity": from_entity,
            "to_entity": to_entity,
            "relation": relation,
            "weight": weight,
            "valid_from": valid_from,
            "valid_to": valid_to,
            "_legacy": True,
        }
        self.edges[edge_id] = e
        return e

    # The single seam the backend uses.
    async def execute_query(self, cypher: str, **params: Any) -> _Result:
        self.queries.append(cypher)
        c = cypher
        # --- vector index creation (no-op in the fake) -----------------------
        if "CREATE VECTOR INDEX" in c:
            return _Result([])

        # --- entity name_key range index creation (no-op in the fake) --------
        # ``ensure_entity_name_index`` emits CREATE INDEX FOR (e:MnemozineEntity)
        # ON (e.name_key); the fake resolves by scanning, so the index is a no-op.
        if "CREATE INDEX FOR (e:MnemozineEntity) ON (e.name_key)" in c:
            return _Result([])

        # --- index-backed vector KNN (the FR-RET-2 scoped_query path) --------
        # Mirror the real FalkorDB ``db.idx.vector.queryNodes`` contract: rank the
        # scope/tier/entity-filtered candidates by cosine *distance* (1 - cosine,
        # so 0 == identical, matching FalkorDB), return ``[node, score]`` rows in
        # ascending-distance order, then honor the over-fetch ``LIMIT $top_k``.
        if "db.idx.vector.queryNodes" in c:
            return self._vector_knn(c, params)

        # --- MnemozineRawChunk (the retained raw tier) -----------------------
        if ":MnemozineRawChunk" in c:
            return self._raw_chunk(c, params)
        # --- graph_snapshot default selection (degree-ranked) ----------------
        # The no-center default selection now degree-ranks entities over BOTH
        # structural layers (``OPTIONAL MATCH (e)-[r:MNEMOZINE_RELATES|
        # MNEMOZINE_CO_MENTIONS]-(...) WITH e, count(r) AS deg ORDER BY deg DESC,
        # e.id RETURN e LIMIT $cap``). Because it names MNEMOZINE_CO_MENTIONS it
        # would otherwise be routed to the co-mention handler; route it to the
        # entity handler explicitly (recognized by ``count(r) AS deg``) so the
        # degree-ranked selection is answered there.
        if "count(r) AS deg" in c and "RETURN e LIMIT $cap" in c:
            return self._entity_or_edge(c, params)
        # --- MNEMOZINE_CO_MENTIONS (weighted entity-entity co-mention edges) -
        # Checked BEFORE the bare :MnemozineEntity branch (the upsert + the
        # graph_snapshot aggregate both MATCH entity nodes) and BEFORE the
        # MNEMOZINE_MENTIONS branch (substring-distinct, but routed explicitly).
        if "MNEMOZINE_CO_MENTIONS" in c:
            return self._co_mention(c, params)
        # --- MNEMOZINE_MENTIONS (memory->entity mention edges + co-mention
        # derivations that read the mention layer) ---------------------------
        # Checked BEFORE the bare :MnemozineMemory / :MnemozineEntity branches
        # because the persist_mentions Cypher MATCHes both labels in one
        # statement; route it (and the mention-derived co_mention_pairs /
        # entity_mention_counts reads) to the dedicated mention handler.
        if "MNEMOZINE_MENTIONS" in c:
            return self._mentions(c, params)
        # --- relation registry (list_relations / merge_relations) ------------
        # The relation analogue of the category registry, grouped/merged over
        # MNEMOZINE_RELATES labels. Routed BEFORE the bare :MnemozineEntity /
        # MNEMOZINE_RELATES edge branches because the list/count statements have no
        # :MnemozineEntity label and the merge statement is a relation-relabel (not
        # an entity merge or an upsert), each recognized by a stable substring.
        if (
            ("RETURN coalesce(r.relation, 'relates') AS relation" in c)
            or ("[r:MNEMOZINE_RELATES {relation: $source}]" in c)
            or ("MERGE (a)-[nr:MNEMOZINE_RELATES {relation: $target}]" in c)
        ):
            return self._relation_norm(c, params)
        # --- MnemozineMemory --------------------------------------------------
        if ":MnemozineMemory" in c:
            return self._memory(c, params)
        # --- MnemozineEntity / edges -----------------------------------------
        if ":MnemozineEntity" in c:
            return self._entity_or_edge(c, params)
        if "MNEMOZINE_RELATES" in c:
            return self._edge_only(c, params)
        # --- suppression ------------------------------------------------------
        if ":MnemozineSuppression" in c:
            return self._suppression(c, params)
        # --- sessions ---------------------------------------------------------
        if ":MnemozineSession" in c:
            return self._session(c, params)
        raise AssertionError(f"FakeFalkorDriver: unhandled cypher:\n{cypher}")

    # -- memory ---------------------------------------------------------------

    def _memory(self, c: str, p: dict[str, Any]) -> _Result:
        if c.startswith("CREATE (m:MnemozineMemory) SET m = $props"):
            props = dict(p["props"])
            # The real backend now splits the embedding out of $props and stores
            # it as a typed vector via ``SET m.embedding = vecf32($embedding)`` in
            # the same statement; fold it back onto the node so the fake's stored
            # shape matches what the backend reads back.
            if "embedding" in p:
                props["embedding"] = list(p["embedding"])
            self.memories[props["id"]] = props
            return _Result([[props]])

        # --- store_stats aggregates (EMBEDDING-FREE Cypher COUNT/grouping) ----
        # totals + active in one pass.
        if "count(m) AS total" in c and "THEN 1 ELSE 0 END) AS active" in c:
            total = len(self.memories)
            active = sum(
                1 for m in self.memories.values() if m.get("valid_to") is None
            )
            return _Result([[total, active]])
        # grouped count: category / tier / scope-decision.
        if "coalesce(m.category, $default) AS k" in c:
            return self._group_count(
                lambda m: m.get("category") or p.get("default")
            )
        if "coalesce(m.tier, $default) AS k" in c:
            return self._group_count(lambda m: m.get("tier") or p.get("default"))
        if "CASE WHEN m.scope = $global THEN $g ELSE $p END AS k" in c:
            return self._group_count(
                lambda m: p["g"] if m.get("scope") == p["global"] else p["p"]
            )
        # by_source: group by the raw provenance blob (backend decodes source).
        if "m.provenance AS prov, count(m) AS n" in c:
            return self._group_count(lambda m: m.get("provenance"))

        # --- memory_growth: per-day grouped count over valid_from -------------
        # ``MATCH (m:MnemozineMemory) WHERE m.valid_from >= $since [AND scope...]
        # RETURN left(toString(m.valid_from), 10) AS day, count(m) AS n
        # ORDER BY day ASC``. ``valid_from`` is an ISO string that sorts lexically,
        # so the $since lower bound and the STARTS WITH scope roll-up are applied
        # exactly as the real Cypher would (global emits NO scope clause).
        if "left(toString(m.valid_from), 10) AS day" in c:
            since = str(p.get("since", ""))
            scope_eq = p.get("scope")
            scope_prefix = p.get("scope_prefix")
            day_counts: dict[str, int] = {}
            for m in self.memories.values():
                vf = str(m.get("valid_from") or "")
                if not vf or vf < since:
                    continue
                if scope_eq is not None or scope_prefix is not None:
                    sc = str(m.get("scope") or "")
                    if not (sc == scope_eq or sc.startswith(str(scope_prefix))):
                        continue
                day = vf[:10]
                day_counts[day] = day_counts.get(day, 0) + 1
            rows_g = sorted(day_counts.items())
            return _Result([[day, n] for day, n in rows_g])

        # --- query_memories: count over the filtered set ----------------------
        if "RETURN count(m) AS n" in c and "SET" not in c and "$src" not in c:
            n = sum(
                1 for m in self.memories.values() if self._memory_matches(c, m, p)
            )
            return _Result([[n]])

        # --- query_memories page / get_memory_display / graph idea-seeds ------
        # The display reads RETURN a field map (``{id: m.id, ...} AS v``); the
        # fake returns the stored props dict (the backend's _props() reads it as a
        # mapping), never selecting the embedding for the assertion's sake.
        if "AS v" in c and "RETURN {" in c:
            want_seed = "AS seed" in c
            rows = [
                m for m in self.memories.values() if self._memory_matches(c, m, p)
            ]
            # Keyed detail read: {id: $id} filter.
            if "{id: $id}" in c:
                rows = [m for m in self.memories.values() if m.get("id") == p.get("id")]
            if "ORDER BY v.valid_from DESC" in c:
                rows.sort(key=lambda m: str(m.get("valid_from") or ""), reverse=True)
            if "SKIP $offset" in c:
                off = int(p.get("offset", 0))
                rows = rows[off:]
            if "LIMIT $limit" in c:
                rows = rows[: int(p.get("limit", len(rows)))]
            # graph_snapshot idea-seed/memory_count scan is bounded in Cypher.
            if "LIMIT $mem_cap" in c:
                rows = rows[: int(p.get("mem_cap", len(rows)))]
            out: list[list[Any]] = []
            for m in rows:
                if want_seed:
                    out.append([m, bool(m.get("cross_ref_candidate", False))])
                else:
                    out.append([m])
            return _Result(out)

        # --- category registry (list_categories / merge_categories) ----------
        # ``list_categories``: count active memories grouped by category. Emitted
        # as ``... WHERE m.valid_to IS NULL RETURN m.category AS category,
        # count(m) AS n`` — return ``[category, count]`` rows.
        if "RETURN m.category AS category" in c:
            counts: dict[str, int] = {}
            for m in self.memories.values():
                if m.get("valid_to") is not None:
                    continue
                cat = m.get("category") or "fact"
                counts[cat] = counts.get(cat, 0) + 1
            return _Result([[cat, n] for cat, n in counts.items()])
        # ``merge_categories``: re-label every memory tagged $src to $tgt, return
        # the count. Emitted as ``... WHERE m.category = $src SET m.category = $tgt
        # RETURN count(m) AS n``.
        if "WHERE m.category = $src" in c and "SET m.category = $tgt" in c:
            n = 0
            for m in self.memories.values():
                if m.get("category") == p.get("src"):
                    m["category"] = p.get("tgt")
                    n += 1
            return _Result([[n]])

        # --- data-versioning: min over the memory tier (min_data_version) ------
        # Emitted as ``MATCH (m:MnemozineMemory) RETURN min(coalesce(m.data_version,
        # 0)) AS v``. Legacy nodes with no ``data_version`` prop coalesce to 0.
        if "min(coalesce(m.data_version, 0))" in c:
            if not self.memories:
                return _Result([[None]])
            lo = min(
                int(m.get("data_version") or 0) for m in self.memories.values()
            )
            return _Result([[lo]])

        # --- data-versioning: set_data_version (explicit stamp) ----------------
        # ``MATCH (m:MnemozineMemory) WHERE m.id IN $ids SET m.data_version =
        # $version RETURN count(m) AS n``.
        if "WHERE m.id IN $ids" in c and "SET m.data_version = $version" in c:
            n = 0
            for mid in p.get("ids", []):
                m = self.memories.get(mid)
                if m is not None:
                    m["data_version"] = p["version"]
                    n += 1
            return _Result([[n]])

        # --- reclassify_memory: SET any subset of scope/category/cross_ref + the
        # always-present data_version re-stamp ----------------------------------
        # Emitted as ``MATCH (m:MnemozineMemory {id: $id}) SET <fields> RETURN m``
        # where <fields> always includes ``m.data_version = $data_version`` and a
        # subset of ``m.scope = $scope`` / ``m.category = $category`` /
        # ``m.cross_ref_candidate = $cross_ref_candidate``.
        if "{id: $id}" in c and "RETURN m" in c and "SET m." in c and (
            "m.data_version = $data_version" in c
            or "m.scope = $scope" in c
            or "m.category = $category" in c
            or "m.cross_ref_candidate = $cross_ref_candidate" in c
        ):
            m = self.memories[p["id"]]
            if "m.data_version = $data_version" in c:
                m["data_version"] = p["data_version"]
            if "m.scope = $scope" in c:
                m["scope"] = p["scope"]
            if "m.category = $category" in c:
                m["category"] = p["category"]
            if "m.cross_ref_candidate = $cross_ref_candidate" in c:
                m["cross_ref_candidate"] = p["cross_ref_candidate"]
            return _Result([[m]])

        if "SET m.confidence" in c:
            m = self.memories[p["id"]]
            m["confidence"] = p["confidence"]
            m["last_accessed"] = p["last_accessed"]
            return _Result([[m]])
        if "SET m.valid_to" in c:
            m = self.memories[p["id"]]
            m["valid_to"] = p["valid_to"]
            return _Result([[m]])
        if "SET m.tier" in c:
            m = self.memories[p["id"]]
            m["tier"] = p["tier"]
            return _Result([[m]])
        if "SET m.embedding" in c:
            m = self.memories[p["id"]]
            m["embedding"] = p["embedding"]
            return _Result([[m]])
        if "SET m.access_count" in c:
            m = self.memories[p["id"]]
            m["access_count"] = int(m.get("access_count", 0)) + 1
            m["last_accessed"] = p["now"]
            return _Result([[m]])

        # MATCH ... RETURN m  (with optional WHERE filters)
        if "{id: $id}" in c and "RETURN m" in c and "SET" not in c:
            m = self.memories.get(p["id"])
            return _Result([[m]] if m else [])

        # Candidate / scoped / iter queries: evaluate the WHERE filters we emit.
        rows: list[list[Any]] = []
        for m in self.memories.values():
            if not self._memory_matches(c, m, p):
                continue
            rows.append([m])
        # honor LIMIT $cap for candidate fetch
        if "LIMIT $cap" in c and "cap" in p:
            rows = rows[: p["cap"]]
        return _Result(rows)

    def _group_count(self, key: Any) -> _Result:
        """Group the memory store by ``key(m)`` and return ``[k, count]`` rows."""

        counts: dict[Any, int] = {}
        for m in self.memories.values():
            k = key(m)
            counts[k] = counts.get(k, 0) + 1
        return _Result([[k, n] for k, n in counts.items()])

    def _vector_knn(self, c: str, p: dict[str, Any]) -> _Result:
        """Interpret the index-backed KNN scoped_query against the dict store.

        Applies the same scope/validity/tier/entity WHERE filters the backend
        emits (reusing :meth:`_memory_matches`), ranks by cosine distance to the
        query vector ``$qv``, and returns ``[node, distance]`` rows ascending —
        the shape the backend converts back to a cosine-similarity score. Honors
        the over-fetch ``LIMIT $top_k`` so behaviour matches FalkorDB's post-KNN
        cut. (The over-fetch ``$k`` is irrelevant in the fake since it ranks the
        full candidate set anyway — exactly the ordering the index would yield.)
        """

        qv = list(p.get("qv") or [])
        scored: list[tuple[float, dict[str, Any]]] = []
        for m in self.memories.values():
            if not self._memory_matches(c, m, p):
                continue
            distance = 1.0 - cosine_similarity(qv, list(m.get("embedding") or []))
            scored.append((distance, m))
        scored.sort(key=lambda t: t[0])
        top_k = p.get("top_k")
        if top_k is not None:
            scored = scored[:top_k]
        return _Result([[m, distance] for distance, m in scored])

    @staticmethod
    def _memory_matches(c: str, m: dict[str, Any], p: dict[str, Any]) -> bool:
        if "m.scope = $scope" in c and m.get("scope") != p.get("scope"):
            return False
        if "m.scope IN $scopes" in c and m.get("scope") not in p.get("scopes", []):
            return False
        if "m.valid_to IS NULL" in c and m.get("valid_to") is not None:
            return False
        if "m.valid_to IS NOT NULL" in c and m.get("valid_to") is None:
            return False
        if "m.tier = $hot" in c and m.get("tier") != p.get("hot"):
            return False
        if "m.tier = $tier" in c and m.get("tier") != p.get("tier"):
            return False
        if "any(e IN m.entities WHERE e IN $entities)" in c:
            if not (set(m.get("entities") or []) & set(p.get("entities") or [])):
                return False
        # --- Memories-table display filters (query_memories) -----------------
        if "m.category = $category" in c and m.get("category") != p.get("category"):
            return False
        if "m.provenance CONTAINS $source_needle" in c:
            if str(p.get("source_needle", "")) not in str(m.get("provenance") or ""):
                return False
        if "any(e IN m.entities WHERE toLower(e) = $entity)" in c:
            wanted = str(p.get("entity", "")).lower()
            if wanted not in {str(e).lower() for e in (m.get("entities") or [])}:
                return False
        if "toLower(m.content) CONTAINS $q" in c:
            if str(p.get("q", "")).lower() not in str(m.get("content") or "").lower():
                return False
        # graph_snapshot idea-seed scan: memories linking a kept entity by name.
        if "any(e IN m.entities WHERE toLower(e) IN $names)" in c:
            names = {str(n).lower() for n in (p.get("names") or [])}
            if not ({str(e).lower() for e in (m.get("entities") or [])} & names):
                return False
        if "m.valid_from < $valid_before" in c:
            vf = m.get("valid_from")
            if vf is None or not (str(vf) < str(p.get("valid_before"))):
                return False
        if "m.last_accessed IS NULL OR m.last_accessed < $unused_since" in c:
            la = m.get("last_accessed")
            if la is not None and not (str(la) < str(p.get("unused_since"))):
                return False
        # Data-versioning selection (iter_memories_below_version): legacy nodes
        # with no ``data_version`` prop coalesce to 0, so they are always below a
        # positive target version.
        if "coalesce(m.data_version, 0) < $version" in c:
            if int(m.get("data_version") or 0) >= int(p.get("version", 0)):
                return False
        return True

    # -- raw chunk (the retained raw tier) ------------------------------------

    def _raw_chunk(self, c: str, p: dict[str, Any]) -> _Result:
        """Interpret the RawChunk persist/iter Cypher against the dict store.

        ``persist_raw_chunk`` is idempotent on ``content_hash`` (the backend emits
        ``MERGE (c:MnemozineRawChunk {content_hash: $content_hash}) SET c = $props``):
        re-persisting the same hash overwrites the stored props in place rather
        than duplicating. ``iter_raw_chunks`` matches EXACTLY (no ancestor
        composition — a re-extraction must not widen scope), AND-combining the
        optional scope/session/source/since filters the backend emits.
        """

        if c.startswith("MERGE (c:MnemozineRawChunk {content_hash: $content_hash})"):
            props = dict(p["props"])
            self.raw_chunks[props["content_hash"]] = props
            return _Result([[props]])

        # --- store_stats: total raw-chunk nodes ------------------------------
        # Guard against set_chunk_data_version, which also RETURNs count(c) AS n
        # but mutates (SET c.data_version) and has its own handler below.
        if "RETURN count(c) AS n" in c and "SET" not in c:
            return _Result([[len(self.raw_chunks)]])

        # --- data-versioning: min over the raw-chunk tier (min_data_version) ---
        # ``MATCH (c:MnemozineRawChunk) RETURN min(coalesce(c.data_version, 0)) AS
        # v``. Legacy chunks with no ``data_version`` prop coalesce to 0.
        if "min(coalesce(c.data_version, 0))" in c:
            if not self.raw_chunks:
                return _Result([[None]])
            lo = min(
                int(ch.get("data_version") or 0) for ch in self.raw_chunks.values()
            )
            return _Result([[lo]])

        # --- data-versioning: set_chunk_data_version (explicit stamp) ----------
        # ``MATCH (c:MnemozineRawChunk) WHERE c.content_hash IN $hashes SET
        # c.data_version = $version RETURN count(c) AS n``.
        if "WHERE c.content_hash IN $hashes" in c and "SET c.data_version = $version" in c:
            n = 0
            for h in p.get("hashes", []):
                chunk = self.raw_chunks.get(h)
                if chunk is not None:
                    chunk["data_version"] = p["version"]
                    n += 1
            return _Result([[n]])

        # MATCH (c:MnemozineRawChunk)[ WHERE ...] RETURN c
        rows: list[list[Any]] = []
        for chunk in self.raw_chunks.values():
            if "c.scope = $scope" in c and chunk.get("scope") != p.get("scope"):
                continue
            if "c.session_id = $session_id" in c and chunk.get("session_id") != p.get(
                "session_id"
            ):
                continue
            if "c.source = $source" in c and chunk.get("source") != p.get("source"):
                continue
            if "c.ingested_at >= $since" in c:
                ing = chunk.get("ingested_at")
                if ing is None or not (str(ing) >= str(p.get("since"))):
                    continue
            # Data-versioning selection (iter_chunks_below_version): legacy chunks
            # with no ``data_version`` prop coalesce to 0 (always below a positive
            # target).
            if "coalesce(c.data_version, 0) < $version" in c:
                if int(chunk.get("data_version") or 0) >= int(p.get("version", 0)):
                    continue
            rows.append([chunk])
        return _Result(rows)

    # -- mentions (memory->entity MNEMOZINE_MENTIONS edges) -------------------

    def _mentions(self, c: str, p: dict[str, Any]) -> _Result:
        """Interpret the MNEMOZINE_MENTIONS persist + mention-derived reads.

        ``persist_mentions``: resolve every ``m.entities`` name to an entity by
        case-folded canonical-name OR alias match (mirroring ``get_entity``'s
        WHERE), MERGE the (memory, entity) edge into the mentions set, and RETURN
        the asserted count. Set semantics make it idempotent (a re-run re-asserts
        the same pairs, adds none).

        ``co_mention_pairs``: enumerate entity-id pairs co-mentioned by the same
        memory with their distinct-shared-memory counts (a<b, >= $min_shared).
        ``entity_mention_counts``: distinct-memory mention count per entity id.
        Each recognized by stable substring or it raises.
        """

        # merge_entities mention repoint: MATCH (m)-[r:MNEMOZINE_MENTIONS]->(s {id:$src})
        # MATCH (t {id:$tgt}) MERGE (m)-[:MNEMOZINE_MENTIONS]->(t) DELETE r — every
        # memory that mentioned the duplicate now mentions the survivor (set
        # semantics collapse a memory that mentioned BOTH onto one edge).
        if "-[r:MNEMOZINE_MENTIONS]->(s:MnemozineEntity {id: $src})" in c and "DELETE r" in c:
            src, tgt = p["src"], p["tgt"]
            self.mentions = {
                (mid, tgt if eid == src else eid) for (mid, eid) in self.mentions
            }
            return _Result([])

        # add_memory_mentions (inline per-memory seam): UNWIND $entity_ids AS eid
        # MATCH (m {id: $memory_id}) MATCH (e {id: eid}) MERGE
        # (m)-[r:MNEMOZINE_MENTIONS]->(e) RETURN count(r) AS n. Idempotent set
        # MERGE keyed on exact node ids; only edges whose endpoints both exist are
        # asserted, mirroring the backend's id-bound MATCH.
        if "UNWIND $entity_ids AS eid" in c and "RETURN count(r) AS n" in c:
            mid = p["memory_id"]
            if mid not in self.memories:
                return _Result([[0]])
            asserted = {
                (mid, eid) for eid in p.get("entity_ids", []) if eid in self.entities
            }
            self.mentions |= asserted
            return _Result([[len(asserted)]])

        # persist_mentions: MATCH (e)/(m) ... MERGE (m)-[r:MNEMOZINE_MENTIONS]->(e)
        # RETURN count(r) AS n
        if "MERGE (m)" in c and "RETURN count(r) AS n" in c:
            # Lower-cased name -> entity-id resolution (canonical + aliases).
            name_to_id: dict[str, str] = {}
            for e in self.entities.values():
                cn = str(e.get("canonical_name") or "")
                if cn:
                    name_to_id[cn.lower()] = e["id"]
                for alias in e.get("aliases") or []:
                    name_to_id.setdefault(str(alias).lower(), e["id"])
            asserted: set[tuple[str, str]] = set()
            for m in self.memories.values():
                for name in m.get("entities") or []:
                    eid = name_to_id.get(str(name).lower())
                    if eid is not None:
                        asserted.add((m["id"], eid))
            self.mentions |= asserted
            return _Result([[len(asserted)]])

        # co_mention_pairs: ... WITH a.id, b.id, count(DISTINCT m) AS shared
        # WHERE shared >= $min_shared RETURN aid, bid, shared
        if "count(DISTINCT m) AS shared" in c and "RETURN aid, bid, shared" in c:
            by_memory: dict[str, set[str]] = {}
            for mid, eid in self.mentions:
                by_memory.setdefault(mid, set()).add(eid)
            pair_counts: dict[tuple[str, str], int] = {}
            for eids in by_memory.values():
                ids = sorted(eids)
                for i in range(len(ids)):
                    for j in range(i + 1, len(ids)):
                        key = (ids[i], ids[j])
                        pair_counts[key] = pair_counts.get(key, 0) + 1
            min_shared = int(p.get("min_shared", 2))
            return _Result(
                [[a, b, n] for (a, b), n in pair_counts.items() if n >= min_shared]
            )

        # entity_mention_counts: MATCH (m)-[:MNEMOZINE_MENTIONS]->(e)
        # RETURN e.id AS eid, count(DISTINCT m) AS df
        if "count(DISTINCT m) AS df" in c and "RETURN e.id AS eid" in c:
            counts: dict[str, int] = {}
            for _mid, eid in self.mentions:
                counts[eid] = counts.get(eid, 0) + 1
            return _Result([[eid, df] for eid, df in counts.items()])

        raise AssertionError(f"FakeFalkorDriver: unhandled mentions cypher:\n{c}")

    # -- co-mention (entity-entity MNEMOZINE_CO_MENTIONS edges) ---------------

    def _co_mention(self, c: str, p: dict[str, Any]) -> _Result:
        """Interpret the MNEMOZINE_CO_MENTIONS upsert + graph_snapshot aggregate.

        ``upsert_co_mention``: MERGE the (from,to) co-mention edge keyed on the
        endpoint pair, SET weight + shared (re-assert, never sum) so it is
        idempotent. The graph_snapshot aggregate RETURNs the kept-id co-mention
        edges with their topology endpoints. Recognized by stable substring.
        """

        # merge_entities co-mention repoint (both directions) + self-loop drop:
        # fold every co-mention edge touching the duplicate ($src) onto the survivor
        # ($tgt), dropping self-loops and keeping the higher-weight edge on a
        # collision so no duplicate parallel co-mention edge remains.
        if "[r:MNEMOZINE_CO_MENTIONS]" in c and "DELETE r" in c:
            self._repoint_co_mention(c, p)
            return _Result([])

        # upsert_co_mention: MATCH (a {id:$lo})/(b {id:$hi}) MERGE
        # (a)-[r:MNEMOZINE_CO_MENTIONS {relation: $relation}]->(b) SET r.weight ...
        # Endpoints arrive canonical (lo <= hi); key on the canonical pair.
        if "MERGE (a)-[r:MNEMOZINE_CO_MENTIONS" in c and "SET r.weight = $weight" in c:
            frm, to = p["lo"], p["hi"]
            key = (frm, to) if frm <= to else (to, frm)
            existing = self.co_mentions.get(key, {})
            rec = {
                "id": existing.get("id") or p["id"],
                "from_entity": key[0],
                "to_entity": key[1],
                "relation": p["relation"],
                "weight": p["weight"],
                "shared": p["shared"],
                "valid_from": existing.get("valid_from") or p["valid_from"],
                "valid_to": None,
            }
            self.co_mentions[key] = rec
            return _Result([[rec]])

        # graph_snapshot co-mention aggregate: MATCH (a)-[r:MNEMOZINE_CO_MENTIONS]
        # ->(b) WHERE a.id IN $ids AND b.id IN $ids RETURN a.id AS source, ...
        if "a.id IN $ids AND b.id IN $ids" in c and "RETURN a.id AS source" in c:
            ids = set(p.get("ids") or [])
            rows = [
                [rec["from_entity"], rec["to_entity"], rec]
                for rec in self.co_mentions.values()
                if rec["from_entity"] in ids and rec["to_entity"] in ids
            ]
            return _Result(rows)

        raise AssertionError(f"FakeFalkorDriver: unhandled co-mention cypher:\n{c}")

    def _repoint_co_mention(self, c: str, p: dict[str, Any]) -> None:
        """Repoint co-mention edges off the duplicate ($src) onto the survivor ($tgt).

        Handles all three merge_entities co-mention statements (outgoing repoint,
        incoming repoint, self-loop drop) by rebuilding the ``co_mentions`` dict
        with every ``$src`` endpoint rewritten to ``$tgt``: self-loops are dropped
        and a collision keeps the higher-weight record, so no duplicate parallel
        co-mention edge survives (idempotent — a re-run with $src already gone is a
        no-op).
        """

        src, tgt = p["src"], p["tgt"]
        repointed: dict[tuple[str, str], dict[str, Any]] = {}
        for (a, b), rec in self.co_mentions.items():
            na = tgt if a == src else a
            nb = tgt if b == src else b
            if na == nb:  # self-loop (src<->tgt or src->src) — never co-mention self
                continue
            # Canonicalize (lo <= hi): a reversing repoint folds onto the survivor's
            # canonical edge (highest weight kept), never a parallel reversed dup.
            lo, hi = (na, nb) if na <= nb else (nb, na)
            new = dict(rec)
            new["from_entity"], new["to_entity"] = lo, hi
            prev = repointed.get((lo, hi))
            if prev is None or new.get("weight", 0.0) > prev.get("weight", 0.0):
                repointed[(lo, hi)] = new
        self.co_mentions = repointed

    # -- relation registry (list_relations / merge_relations) -----------------

    def _relation_norm(self, c: str, p: dict[str, Any]) -> _Result:
        """Interpret the relation-registry list + the relation relabel/merge.

        ``list_relations``: grouped count of active MNEMOZINE_RELATES edges by
        relation label (label missing -> 'relates').
        ``merge_relations`` is two statements: first a count of active source-
        relation edges (the relabelled total), then a MERGE-onto-(a,b,target) +
        DELETE-source relabel that combines weight via max and never leaves a
        duplicate parallel edge. Each recognized by stable substring or it raises.
        """

        active = [e for e in self.edges.values() if e.get("valid_to") is None]

        # list_relations: ... WHERE r.valid_to IS NULL
        # RETURN coalesce(r.relation, 'relates') AS relation, count(r) AS n
        if "RETURN coalesce(r.relation, 'relates') AS relation" in c:
            counts: dict[str, int] = {}
            for e in active:
                rel = e.get("relation") or "relates"
                counts[rel] = counts.get(rel, 0) + 1
            return _Result([[rel, n] for rel, n in counts.items()])

        # merge_relations count: ...[r:MNEMOZINE_RELATES {relation: $source}]...
        # WHERE r.valid_to IS NULL RETURN count(r) AS n
        if "[r:MNEMOZINE_RELATES {relation: $source}]" in c and "count(r) AS n" in c:
            n = sum(1 for e in active if e.get("relation") == p["source"])
            return _Result([[n]])

        # merge_relations relabel: MERGE (a)-[nr:MNEMOZINE_RELATES
        # {relation: $target}]->(b) ... DELETE r
        if "MERGE (a)-[nr:MNEMOZINE_RELATES {relation: $target}]" in c and "DELETE r" in c:
            source, target = p["source"], p["target"]
            for edge in list(self.edges.values()):
                if edge.get("valid_to") is not None or edge.get("relation") != source:
                    continue
                frm, to = edge["from_entity"], edge["to_entity"]
                existing = next(
                    (
                        e
                        for e in self.edges.values()
                        if e.get("valid_to") is None
                        and e["from_entity"] == frm
                        and e["to_entity"] == to
                        and e.get("relation") == target
                    ),
                    None,
                )
                if existing is None:
                    # No parallel target edge: relabel the source edge in place.
                    edge["relation"] = target
                else:
                    # MERGE found a target edge: combine weight (max), drop source.
                    existing["weight"] = max(
                        existing.get("weight", 0.0), edge.get("weight", 0.0)
                    )
                    del self.edges[edge["id"]]
            return _Result([])

        raise AssertionError(f"FakeFalkorDriver: unhandled relation cypher:\n{c}")

    # -- entity / edge --------------------------------------------------------

    def _entity_or_edge(self, c: str, p: dict[str, Any]) -> _Result:
        # --- store_stats: total entity nodes ---------------------------------
        if "RETURN count(e) AS n" in c:
            return _Result([[len(self.entities)]])

        # --- graph_snapshot: bounded entity selection ------------------------
        # Centered traversal: center + one-hop neighbors, capped at $cap.
        if "(c:MnemozineEntity {id: $center})" in c and "RETURN e LIMIT $cap" in c:
            keep: dict[str, dict[str, Any]] = {}
            center = self.entities.get(p["center"])
            if center is not None:
                keep[center["id"]] = center
                for edge in self.edges.values():
                    other_id = None
                    if edge["from_entity"] == center["id"]:
                        other_id = edge["to_entity"]
                    elif edge["to_entity"] == center["id"]:
                        other_id = edge["from_entity"]
                    if other_id is None:
                        continue
                    o = self.entities.get(other_id)
                    if o is None:
                        continue
                    if "o.type = $entity_type" in c and o.get("type") != p.get(
                        "entity_type"
                    ):
                        continue
                    keep[o["id"]] = o
            rows = [[e] for e in list(keep.values())[: p["cap"]]]
            return _Result(rows)
        # --- graph_snapshot default selection: DEGREE-RANKED bounded slice ----
        # The no-center default now ranks entities by incident structural degree
        # (RELATES edges + CO_MENTIONS) descending, tie-break on id, then takes the
        # top $cap — so the snapshot surfaces the connected structure, not an
        # arbitrary slice. Recognized by ``count(r) AS deg``; honors the optional
        # ``e.type = $entity_type`` filter, exactly like the plain branch below.
        if "count(r) AS deg" in c and "RETURN e LIMIT $cap" in c:
            ents = [
                e
                for e in self.entities.values()
                if "e.type = $entity_type" not in c
                or e.get("type") == p.get("entity_type")
            ]
            degree: dict[str, int] = {e["id"]: 0 for e in ents}
            for edge in self.edges.values():
                if edge["from_entity"] in degree:
                    degree[edge["from_entity"]] += 1
                if edge["to_entity"] in degree:
                    degree[edge["to_entity"]] += 1
            for rec in self.co_mentions.values():
                if rec["from_entity"] in degree:
                    degree[rec["from_entity"]] += 1
                if rec["to_entity"] in degree:
                    degree[rec["to_entity"]] += 1
            ents.sort(key=lambda e: (-degree[e["id"]], e["id"]))
            return _Result([[e] for e in ents[: p["cap"]]])
        # Plain bounded entity list (optional type filter), capped at $cap.
        if c.startswith("MATCH (e:MnemozineEntity)") and "RETURN e LIMIT $cap" in c:
            ents = [
                e
                for e in self.entities.values()
                if "e.type = $entity_type" not in c
                or e.get("type") == p.get("entity_type")
            ]
            return _Result([[e] for e in ents[: p["cap"]]])

        # --- graph_snapshot: structural edges among kept entities (1 query) --
        # The backend now RETURNs the endpoint ids from the matched topology
        # (``a.id AS source, b.id AS target, r``) so it never depends on the stored
        # from/to props. We answer ``source``/``target`` from the fake's internal
        # edge topology (which it always knows), and hand ``_edge_view(e)`` — which
        # omits from/to for a legacy edge — as the relation value.
        if "a.id IN $ids AND b.id IN $ids" in c and "RETURN a.id AS source" in c:
            ids = set(p.get("ids") or [])
            rows = [
                [e["from_entity"], e["to_entity"], self._edge_view(e)]
                for e in self.edges.values()
                if e["from_entity"] in ids and e["to_entity"] in ids
            ]
            return _Result(rows)

        # --- resolve_or_create_entity: name-keyed lookup ---------------------
        # MATCH (e) WHERE e.name_key = toLower($canonical_name) RETURN e LIMIT 1 —
        # the identity-by-normalized-name probe. Resolve by case-folded
        # canonical_name (the fake stores name_key on write, but matching on the
        # lowered canonical_name is equivalent and robust to legacy rows).
        if "e.name_key = toLower($canonical_name)" in c and "RETURN e LIMIT 1" in c:
            key = str(p["canonical_name"]).lower()
            for e in self.entities.values():
                stored_key = e.get("name_key") or str(
                    e.get("canonical_name") or ""
                ).lower()
                if stored_key == key:
                    return _Result([[e]])
            return _Result([])

        # --- backfill_entity_name_keys: stamp unset name_key in place --------
        # MATCH (e) WHERE e.name_key IS NULL SET e.name_key = toLower(...) RETURN
        # count(e). Touch only entities missing name_key (idempotent).
        if "e.name_key IS NULL" in c and "SET e.name_key = toLower(e.canonical_name)" in c:
            stamped = 0
            for e in self.entities.values():
                if e.get("name_key") is None:
                    e["name_key"] = str(e.get("canonical_name") or "").lower()
                    stamped += 1
            return _Result([[stamped]])

        if c.startswith("MERGE (e:MnemozineEntity {id: $id})"):
            e = self.entities.setdefault(p["id"], {"id": p["id"]})
            e["canonical_name"] = p["canonical_name"]
            e["aliases"] = list(p["aliases"])
            e["type"] = p["type"]
            # Maintain the storage-only name_key index invariant on every write
            # (the backend's SET e.name_key = toLower($canonical_name)).
            e["name_key"] = str(p["canonical_name"]).lower()
            return _Result([[e]])

        if c.startswith("MATCH (e:MnemozineEntity)") and "RETURN e LIMIT 1" in c:
            key = p["key"]
            for e in self.entities.values():
                if (
                    e["id"] == key
                    or e.get("canonical_name") == key
                    or key in (e.get("aliases") or [])
                ):
                    return _Result([[e]])
            return _Result([])

        if c.startswith("MATCH (e:MnemozineEntity) RETURN e"):
            return _Result([[e] for e in self.entities.values()])

        if "SET t.aliases = $aliases" in c:
            t = self.entities[p["tgt"]]
            t["aliases"] = list(p["aliases"])
            # The alias-update write (merge_entities + resolve_or_create_entity)
            # now ALSO re-asserts the name_key invariant on the survivor.
            if "t.name_key = toLower(t.canonical_name)" in c:
                t["name_key"] = str(t.get("canonical_name") or "").lower()
            return _Result([[t]])

        if c.startswith("MATCH (s:MnemozineEntity {id: $src}) DELETE s"):
            self.entities.pop(p["src"], None)
            return _Result([])

        # edge repointing during merge — repoint edges off src to tgt
        if "DELETE r" in c and "MNEMOZINE_RELATES" in c:
            self._repoint_edges(c, p)
            return _Result([])

        # neighbors traversal (RETURNs o, r + the topology endpoint ids src/dst)
        if "RETURN o, r" in c and "ORDER BY r.weight DESC" in c:
            return self._neighbors(p)

        # upsert_edge / edges_for_entity / create edge
        return self._edge_only(c, p)

    def _repoint_edges(self, c: str, p: dict[str, Any]) -> None:
        src, tgt = p["src"], p["tgt"]
        outgoing = "(s:MnemozineEntity {id: $src})-[r:MNEMOZINE_RELATES]->(o)" in c
        for edge in list(self.edges.values()):
            if outgoing and edge["from_entity"] == src:
                self._merge_repoint(edge, frm=tgt, to=edge["to_entity"])
                del self.edges[edge["id"]]
            elif (not outgoing) and edge["to_entity"] == src:
                self._merge_repoint(edge, frm=edge["from_entity"], to=tgt)
                del self.edges[edge["id"]]

    def _merge_repoint(self, edge: dict[str, Any], *, frm: str, to: str) -> None:
        for existing in self.edges.values():
            if (
                existing["from_entity"] == frm
                and existing["to_entity"] == to
                and existing["relation"] == edge["relation"]
            ):
                return  # MERGE found an existing edge; keep it
        new = dict(edge)
        new["from_entity"] = frm
        new["to_entity"] = to
        # Mint a fresh id for the MERGEd (nr) edge so the caller's subsequent
        # ``del self.edges[edge['id']]`` (deleting the source r) does not clobber
        # this just-created survivor edge — they would otherwise share one key.
        new["id"] = f"repoint:{frm}:{to}:{edge['relation']}"
        self.edges[new["id"]] = new

    def _edge_only(self, c: str, p: dict[str, Any]) -> _Result:
        # find active edge by (from,to,relation). The backend re-assert path knows
        # the endpoints from the incoming edge param (so it does NOT read them back
        # from props), hence we return the legacy-aware edge view here too.
        if "RETURN r LIMIT 1" in c and "{relation: $relation}" in c:
            for e in self.edges.values():
                if (
                    e["from_entity"] == p["from"]
                    and e["to_entity"] == p["to"]
                    and e["relation"] == p["relation"]
                    and e.get("valid_to") is None
                ):
                    return _Result([[self._edge_view(e)]])
            return _Result([])

        if "SET r.weight = $weight" in c and "{id: $id}" in c:
            e = self.edges[p["id"]]
            e["weight"] = p["weight"]
            return _Result([[e]])

        if c.startswith("MATCH (a:MnemozineEntity {id: $from_entity})") and "CREATE (a)-[r" in c:
            e = {
                "id": p["id"],
                "from_entity": p["from_entity"],
                "to_entity": p["to_entity"],
                "relation": p["relation"],
                "weight": p["weight"],
                "valid_from": p["valid_from"],
                "valid_to": p["valid_to"],
            }
            self.edges[e["id"]] = e
            return _Result([[e]])

        # prune_edge: SET r.valid_to ... RETURN r, startNode(r).id AS src,
        # endNode(r).id AS dst. The endpoint ids come from the fake's internal
        # topology; the relation value omits from/to for a legacy edge.
        if "SET r.valid_to = $valid_to RETURN r" in c:
            e = self.edges.get(p["id"])
            if e is None:
                return _Result([])
            e["valid_to"] = p["valid_to"]
            return _Result([[self._edge_view(e), e["from_entity"], e["to_entity"]]])

        # edges_for_entity: RETURN r, startNode(r).id AS src, endNode(r).id AS dst.
        if "RETURN r" in c and "{id: $id}" in c:
            eid = p["id"]
            active_only = "r.valid_to IS NULL" in c
            rows = [
                [self._edge_view(e), e["from_entity"], e["to_entity"]]
                for e in self.edges.values()
                if (e["from_entity"] == eid or e["to_entity"] == eid)
                and (not active_only or e.get("valid_to") is None)
            ]
            return _Result(rows)

        raise AssertionError(f"FakeFalkorDriver: unhandled edge cypher:\n{c}")

    def _neighbors(self, p: dict[str, Any]) -> _Result:
        eid = p["id"]
        rows: list[list[Any]] = []
        for e in self.edges.values():
            if e.get("valid_to") is not None:
                continue
            other_id = None
            if e["from_entity"] == eid:
                other_id = e["to_entity"]
            elif e["to_entity"] == eid:
                other_id = e["from_entity"]
            if other_id is None:
                continue
            other = self.entities.get(other_id)
            if other is None:
                continue
            # RETURN o, r, startNode(r).id AS src, endNode(r).id AS dst — the
            # endpoint ids come from the fake's internal topology; the relation
            # value omits from/to for a legacy edge (see _edge_view).
            rows.append([other, self._edge_view(e), e["from_entity"], e["to_entity"]])
        rows.sort(key=lambda r: r[1].get("weight", 0.0), reverse=True)
        return _Result(rows[: p["cap"]])

    # -- suppression ----------------------------------------------------------

    def _suppression(self, c: str, p: dict[str, Any]) -> _Result:
        key = (p["memory_id"], p["context_key"])
        if c.startswith("MERGE (s:MnemozineSuppression"):
            self.suppressions.add(key)
            return _Result([[{"memory_id": key[0], "context_key": key[1]}]])
        # MATCH ... RETURN s LIMIT 1
        return _Result([[{}]] if key in self.suppressions else [])

    # -- session --------------------------------------------------------------

    def _session(self, c: str, p: dict[str, Any]) -> _Result:
        key = (p["source"], p["session_id"])
        s = self.sessions.setdefault(key, {"source": key[0], "session_id": key[1]})
        s.update(
            {
                "project": p["project"],
                "started_at": p["started_at"],
                "ended_at": p["ended_at"],
                "raw_path": p["raw_path"],
            }
        )
        return _Result([[s]])


class FakeGraphitiClient:
    """Stand-in for :class:`GraphitiClient` wrapping a :class:`FakeFalkorDriver`.

    The backend only needs ``execute_query`` + ``close`` from the client, so this
    forwards to the fake driver — letting the contract test build a real
    ``GraphitiStorageBackend`` with no graphiti/FalkorDB import.
    """

    def __init__(self) -> None:
        self.driver = FakeFalkorDriver()

    async def execute_query(self, cypher: str, **params: Any) -> _Result:
        return await self.driver.execute_query(cypher, **params)

    async def ensure_entity_name_index(self) -> None:
        """Mirror :meth:`GraphitiClient.ensure_entity_name_index` (no-op create).

        The backend's ``backfill_entity_name_keys`` ensures the entity name_key
        range index before its SET pass; the fake routes the CREATE INDEX through
        ``execute_query`` (a no-op branch) so the real backend code path is
        exercised unchanged.
        """

        await self.execute_query(
            "CREATE INDEX FOR (e:MnemozineEntity) ON (e.name_key)"
        )

    async def close(self) -> None:
        await self.driver.close()
