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
        self.sessions: dict[tuple[str, str], dict[str, Any]] = {}
        self.suppressions: set[tuple[str, str]] = set()
        # Raw-chunk tier (offline re-extraction/reindex), keyed on content_hash
        # to mirror the backend's MERGE idempotency key (FR-ING-5).
        self.raw_chunks: dict[str, dict[str, Any]] = {}
        self.closed = False
        self.queries: list[str] = []

    async def close(self) -> None:
        self.closed = True

    # The single seam the backend uses.
    async def execute_query(self, cypher: str, **params: Any) -> _Result:
        self.queries.append(cypher)
        c = cypher
        # --- vector index creation (no-op in the fake) -----------------------
        if "CREATE VECTOR INDEX" in c:
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
        if "m.tier = $hot" in c and m.get("tier") != p.get("hot"):
            return False
        if "m.tier = $tier" in c and m.get("tier") != p.get("tier"):
            return False
        if "any(e IN m.entities WHERE e IN $entities)" in c:
            if not (set(m.get("entities") or []) & set(p.get("entities") or [])):
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

    # -- entity / edge --------------------------------------------------------

    def _entity_or_edge(self, c: str, p: dict[str, Any]) -> _Result:
        if c.startswith("MERGE (e:MnemozineEntity {id: $id})"):
            e = self.entities.setdefault(p["id"], {"id": p["id"]})
            e["canonical_name"] = p["canonical_name"]
            e["aliases"] = list(p["aliases"])
            e["type"] = p["type"]
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
            return _Result([[t]])

        if c.startswith("MATCH (s:MnemozineEntity {id: $src}) DELETE s"):
            self.entities.pop(p["src"], None)
            return _Result([])

        # edge repointing during merge — repoint edges off src to tgt
        if "DELETE r" in c and "MNEMOZINE_RELATES" in c:
            self._repoint_edges(c, p)
            return _Result([])

        # neighbors traversal
        if "RETURN o, r ORDER BY r.weight DESC" in c:
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
        self.edges[new["id"]] = new

    def _edge_only(self, c: str, p: dict[str, Any]) -> _Result:
        # find active edge by (from,to,relation)
        if "RETURN r LIMIT 1" in c and "{relation: $relation}" in c:
            for e in self.edges.values():
                if (
                    e["from_entity"] == p["from"]
                    and e["to_entity"] == p["to"]
                    and e["relation"] == p["relation"]
                    and e.get("valid_to") is None
                ):
                    return _Result([[e]])
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

        if "SET r.valid_to = $valid_to RETURN r" in c:
            e = self.edges.get(p["id"])
            if e is None:
                return _Result([])
            e["valid_to"] = p["valid_to"]
            return _Result([[e]])

        # edges_for_entity
        if "RETURN r" in c and "{id: $id}" in c:
            eid = p["id"]
            active_only = "r.valid_to IS NULL" in c
            rows = [
                [e]
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
            rows.append([other, e])
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

    async def close(self) -> None:
        await self.driver.close()
