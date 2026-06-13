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
        if c.startswith("CREATE (m:MnemozineMemory $props)"):
            props = dict(p["props"])
            self.memories[props["id"]] = props
            return _Result([[props]])

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
        return True

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
