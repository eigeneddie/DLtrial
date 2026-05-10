"""
rag_engine.py — Chilli Chiplets Thermal Co-Pilot RAG Engine
============================================================
Adapted from data/rag_app/rag_engine.py.

Key changes vs the original:
  - Single collection (chiplet_thermal_knowledge) instead of dual product/knowledge
  - System prompt repersonalized for 2.5D chiplet thermal co-design
  - Vision support: accepts an optional base64-encoded heatmap PNG and injects
    it as an image part into the Gemini API call so the LLM can "see" the map
  - Simulation state context automatically prepended to every query
"""

from __future__ import annotations

import base64
import json
import re
import requests
import sys
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchText
from langchain_community.embeddings import OllamaEmbeddings

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


# ── Retrieval result helpers ──────────────────────────────────────────────────

def _point_to_ctx(point, rank: int) -> dict:
    payload = point.payload or {}
    return {
        "rank"       : rank,
        "id"         : payload.get("id", str(point.id)),
        "title"      : payload.get("title", "Unknown"),
        "source_file": payload.get("source_file", ""),
        "content"    : payload.get("content", ""),
        "similarity" : point.score,
    }


# ── Scroll stop-words (for keyword-based equation retrieval) ──────────────────

_SCROLL_STOP_WORDS = frozenset({
    "what", "with", "from", "that", "this", "does", "used", "give",
    "show", "find", "tell", "have", "been", "will", "would", "could",
    "should", "when", "where", "which", "there", "their", "about",
    "into", "than", "then", "more", "also", "only", "just", "heat",
    "thermal", "using", "some", "such",
})

_EQUATION_KEYWORDS = frozenset({
    "equation", "formula", "formulae", "calculate", "calculation",
    "table", "coefficient", "number", "ratio", "flux", "bond",
    "nusselt", "efficiency", "resistance", "conductance", "correlation",
    "poisson", "fourier", "cte", "stress", "mismatch",
})


def _clean_answer_text(text: str) -> str:
    """Keep model output readable inside Streamlit chat cards."""
    if not text:
        return ""
    cleaned = text
    cleaned = re.sub(
        r"\$?([0-9]+(?:\.[0-9]+)?)\s*\^?\\circ\s*\\text\{C\}\$?",
        r"\1 °C",
        cleaned,
    )
    cleaned = re.sub(
        r"\$?([0-9]+(?:\.[0-9]+)?)\s*\^?\\circ\s*C\$?",
        r"\1 °C",
        cleaned,
    )
    cleaned = cleaned.replace("\\text{C}", "C")
    cleaned = cleaned.replace("\\text{cell}", "cell")
    cleaned = cleaned.replace("^\\circ", "°")
    cleaned = cleaned.replace("\\circ", "°")
    cleaned = cleaned.replace("\\Delta", "delta")
    cleaned = cleaned.replace("$", "")
    return cleaned.strip()


# ── Main RAG class ────────────────────────────────────────────────────────────

class ChipletRAG:
    """
    Retrieval-Augmented Generation engine for Chilli Chiplets.

    Usage:
        rag = ChipletRAG()
        answer = rag.ask(
            query        = "Why is the top-left corner failing?",
            sim_state    = {"peak_T": 127.4, "material": "aln", "cte_failures": 342, ...},
            heatmap_b64  = base64_encoded_png_string,   # optional
            provider     = "gemini",
        )
    """

    def __init__(self):
        self.client = None
        self.embeddings = None
        try:
            self.client = QdrantClient(path=config.QDRANT_PATH)
            self.embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL)
            self._ready = self.client.collection_exists(config.COLLECTION)
        except Exception:
            self._ready = False

    @property
    def ready(self) -> bool:
        """True if the Qdrant collection exists (i.e. ingest has been run)."""
        return self._ready

    def chunk_count(self) -> int:
        if not self.ready or self.client is None:
            return 0
        info = self.client.get_collection(config.COLLECTION)
        return info.points_count

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def _is_equation_query(self, query: str) -> bool:
        return any(kw in query.lower() for kw in _EQUATION_KEYWORDS)

    def _keyword_scroll(self, query: str, limit: int = 4) -> list[dict]:
        """Direct keyword scan — bypasses embeddings for math/equation chunks."""
        words = re.findall(r'\b[a-z]{4,}\b', query.lower())
        terms = [w for w in words if w not in _SCROLL_STOP_WORDS][:5]

        seen, results = set(), []
        for term in terms:
            filt = Filter(must=[FieldCondition(key="content", match=MatchText(text=term))])
            records, _ = self.client.scroll(
                collection_name=config.COLLECTION,
                scroll_filter=filt,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            for rec in records:
                pid = (rec.payload or {}).get("id", str(rec.id))
                if pid not in seen:
                    seen.add(pid)
                    payload = rec.payload or {}
                    results.append({
                        "rank"       : len(results) + 1,
                        "id"         : pid,
                        "title"      : payload.get("title", "Unknown"),
                        "source_file": payload.get("source_file", ""),
                        "content"    : payload.get("content", ""),
                        "similarity" : 0.5,
                    })
        return results

    def retrieve(self, query: str, top_k: int = config.TOP_K) -> list[dict]:
        """Semantic search + optional keyword scan for equations."""
        if not self.ready or self.client is None or self.embeddings is None:
            return []

        vec     = self.embeddings.embed_query(query)
        results = self.client.query_points(
            collection_name=config.COLLECTION,
            query=vec,
            limit=top_k,
        )
        contexts = [_point_to_ctx(p, i) for i, p in enumerate(results.points, 1)]

        # For equation/formula queries add keyword-scanned chunks
        if self._is_equation_query(query):
            seen = {c["id"] for c in contexts}
            for ctx in self._keyword_scroll(query):
                if ctx["id"] not in seen:
                    seen.add(ctx["id"])
                    contexts.append(ctx)

        return contexts

    # ── Context assembly ──────────────────────────────────────────────────────

    def _build_context_str(self, contexts: list[dict]) -> str:
        chunks = []
        for ctx in contexts:
            header = f"[Source: {Path(ctx['source_file']).name}] [Title: {ctx['title']}]"
            text   = ctx.get("content", "")[:config.MAX_CTX_CHARS]
            chunks.append(f"{header}\n{text}")
        return "\n\n---\n\n".join(chunks)

    @staticmethod
    def _sim_state_str(sim_state: dict) -> str:
        """Convert the live structured simulation state to compact JSON."""
        try:
            state_json = json.dumps(sim_state, indent=2, sort_keys=True)
        except TypeError:
            state_json = str(sim_state)
        return f"[STRUCTURED DESIGN STATE JSON]\n{state_json}"

    # ── Generation (Gemini with optional vision) ──────────────────────────────

    def _answer_gemini(
        self,
        query      : str,
        context_str: str,
        sim_str    : str,
        heatmap_b64: str | None,
        history    : list | None,
        api_key    : str,
    ) -> str:
        augmented = (
            f"{sim_str}\n\n"
            f"Retrieved thermal engineering literature:\n{context_str}\n\n"
            f"Engineer's question: {query}"
        )

        # Build the content parts for this turn
        user_parts: list[dict] = [{"text": augmented}]

        # Attach heatmap as an inline image if provided
        if heatmap_b64:
            user_parts.insert(0, {
                "inline_data": {
                    "mime_type": "image/png",
                    "data"     : heatmap_b64,
                }
            })

        # Build full conversation contents
        contents = []
        for msg in (history or []):
            role = "model" if msg["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})
        contents.append({"role": "user", "parts": user_parts})

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{config.GEMINI_MODEL}:generateContent?key={api_key}"
        )
        payload = {
            "system_instruction": {"parts": [{"text": config.COPILOT_SYSTEM_PROMPT}]},
            "contents"          : contents,
            "generationConfig"  : {"temperature": config.TEMPERATURE},
        }

        resp = requests.post(url, json=payload, timeout=90)
        if not resp.ok:
            raise RuntimeError(f"Gemini API error: {resp.status_code} {resp.text[:300]}")

        candidates = resp.json().get("candidates", [])
        if not candidates:
            return "No response from Gemini."
        parts = candidates[0].get("content", {}).get("parts", [])
        return "\n".join(p.get("text", "") for p in parts if p.get("text")).strip()

    def _answer_ollama(
        self,
        query: str,
        context_str: str,
        sim_str: str,
        history: list | None,
        model_name: str,
    ) -> str:
        context_block = context_str or "No literature context was retrieved. Use the structured design state and first principles."
        augmented = (
            f"{sim_str}\n\n"
            f"[RETRIEVED LITERATURE CONTEXT]\n{context_block}\n\n"
            f"[ENGINEER QUESTION]\n{query}\n\n"
            "Answer as a packaging thermal engineer, but be terse and grounded. "
            "Use only values present in the structured state for this design. "
            "Format exactly as three short sections: Direct answer, Evidence, First change. "
            "Use plain text units only, for example 149.6 °C and 14.9 °C/cell. "
            "Do not use LaTeX, dollar signs, superscripts, or equation formatting. "
            "Avoid tables, long background, and generic packaging lectures. "
            "Recommend only current app controls: move/resize/change power of components, "
            "switch substrate material, change TSV strip count, or change TSV orientation. "
            "Do not give exact TSV coordinate ranges because the app does not expose that control. "
            "Do not call a temperature safe/low/high unless the state includes a target limit. "
            "For first changes, prefer one controlled experiment and rerun validation. "
            "If validation metrics exist, mention trust level in one sentence."
        )

        messages = [{"role": "system", "content": config.COPILOT_SYSTEM_PROMPT}]
        for msg in (history or [])[-6:]:
            role = "assistant" if msg["role"] == "assistant" else "user"
            messages.append({"role": role, "content": msg["content"]})
        messages.append({"role": "user", "content": augmented})

        url = f"{config.OLLAMA_BASE_URL.rstrip('/')}/api/chat"
        payload = {
            "model": model_name or config.OLLAMA_CHAT_MODEL,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0.05,
                "num_predict": 500,
            },
        }
        resp = requests.post(url, json=payload, timeout=120)
        if not resp.ok:
            raise RuntimeError(f"Ollama API error: {resp.status_code} {resp.text[:300]}")
        content = resp.json().get("message", {}).get("content", "").strip()
        return _clean_answer_text(content) or "No response from Ollama."

    # ── Public interface ──────────────────────────────────────────────────────

    def ask(
        self,
        query      : str,
        sim_state  : dict | None  = None,
        heatmap_b64: str  | None  = None,
        history    : list | None  = None,
        provider   : str          = "gemini",
        api_key    : str          = "",
    ) -> str:
        """
        Ask the Co-Pilot a question.

        Parameters
        ----------
        query       : The engineer's question.
        sim_state   : Dict with current simulation metrics (peak_T, material, etc.)
        heatmap_b64 : Base64-encoded PNG of the thermal heatmap (optional vision).
        history     : List of {"role": "user"|"assistant", "content": "..."} dicts.
        provider    : "gemini" (default) | "openai" | "ollama"
        api_key     : Override API key (uses config default if empty).
        """
        contexts    = self.retrieve(query)
        context_str = self._build_context_str(contexts)
        sim_str     = self._sim_state_str(sim_state or {})
        key         = api_key or config.GEMINI_API_KEY

        if provider == "gemini":
            if not key:
                raise RuntimeError("GEMINI_API_KEY is not configured.")
            return self._answer_gemini(query, context_str, sim_str, heatmap_b64, history, key)

        if provider == "ollama":
            return self._answer_ollama(query, context_str, sim_str, history, api_key or config.OLLAMA_CHAT_MODEL)

        raise ValueError(f"Unsupported provider: {provider}")
