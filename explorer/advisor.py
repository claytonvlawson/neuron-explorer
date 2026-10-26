"""Runs the production Parent Advisor pipeline (retrieval + persona) against the LOCAL model.

The public bot at learnwithclay.ai generates on a remote API model, so its neurons are not observable. This
module keeps everything that makes the bot the bot -- the same vault index, the same BM25 retriever, the same
PARENT_SYSTEM persona, the same crisis force-inject, the same "Context:/Question:" prompt shape -- and points
the generation step at Qwen3.5-4B on this machine instead. What you see firing is the same prompt the bot
would answer; only the model differs.

Persona and crisis patterns are read out of `vault-advisor-cloud/parent_advisor_server.py` by AST parsing, not
by import: importing that module builds the index and makedirs /opt/advisor/logs-parent as a side effect.

Env:
  ADVISOR_SRC    directory holding retrieval.py and parent_advisor_server.py
  ADVISOR_INDEX  BM25 pickle to answer from (default: the parent advisor's, if present locally)
  ADVISOR_VAULT  vault that index was built from (citations read frontmatter off it)
  ADVISOR_TOP_K  default 5, same as production
"""
import ast
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get("ADVISOR_SRC", os.path.expanduser("~/vault-advisor-cloud"))

_CANDIDATES = [
    (os.environ.get("ADVISOR_INDEX"), os.environ.get("ADVISOR_VAULT")),
    # the droplet's own index and vault, rsynced here: same corpus the live bot answers from
    (os.path.join(os.path.dirname(HERE), "advisor-prod", "bm25.pkl"),
     os.path.join(os.path.dirname(HERE), "advisor-prod", "vault")),
    (os.path.expanduser("~/kb-advisor/clients/thesis-copilot/index/bm25.pkl"),
     os.path.expanduser("~/kb-advisor/clients/thesis-copilot/vault")),
    (os.path.expanduser("~/vault-advisor-cloud/index/bm25.pkl"),
     os.path.expanduser("~/vault-advisor-cloud/vault")),
]
INDEX, VAULT = next(((i, v) for i, v in _CANDIDATES if i and os.path.exists(i)),
                    (os.path.expanduser("~/kb-advisor/clients/thesis-copilot/index/bm25.pkl"),
                     os.path.expanduser("~/kb-advisor/clients/thesis-copilot/vault")))
TOP_K = int(os.environ.get("ADVISOR_TOP_K", "5"))


def _constants(names):
    """Pull top-level constant assignments out of the production server without importing it."""
    path = os.path.join(SRC, "parent_advisor_server.py")
    out = {}
    if not os.path.exists(path):
        return out
    tree = ast.parse(open(path, encoding="utf-8").read())
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        t = node.targets[0]
        if isinstance(t, ast.Name) and t.id in names:
            try:
                out[t.id] = ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                pass
    return out


_PROD = None


def production():
    """The real parent_advisor_server module, imported with its index load stubbed out.

    Importing it plainly would build a second BM25 over the same pickle (2.5 s and ~1 GB) and makedirs
    /opt/advisor/logs-parent. Stubbing `retrieval.load_index` for the duration of the import keeps the persona,
    the crisis regexes and the citation builders -- the parts that must not drift -- without either side effect.
    """
    global _PROD
    if _PROD is not None:
        return _PROD
    import importlib.util
    import tempfile
    import types
    sys.path.insert(0, SRC)
    real_path = os.path.join(SRC, "parent_advisor_server.py")
    stub = types.ModuleType("retrieval")
    stub.__dict__.update({k: v for k, v in vars(_load_retrieval()).items() if not k.startswith("__")})
    stub.load_index = lambda p: ([], None)
    saved = sys.modules.get("retrieval")
    sys.modules["retrieval"] = stub
    os.environ.setdefault("ADVISOR_LOG_DIR", tempfile.mkdtemp(prefix="advisor-import-"))
    os.environ.setdefault("ADVISOR_VAULT", VAULT)  # citation_for() reads frontmatter off this root
    try:
        spec = importlib.util.spec_from_file_location("pa_server", real_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        if saved is not None:
            sys.modules["retrieval"] = saved
        else:
            sys.modules.pop("retrieval", None)
    _PROD = mod
    return mod


def _prod_attr(name, default=None):
    try:
        return getattr(production(), name, default)
    except Exception:
        return default


# NOTE: these are resolved at the bottom of the file, after _load_retrieval exists. Resolving them here
# raised NameError only during import (functions are not yet bound), which silently fell back to an empty
# persona and a dead crisis regex while the running process looked fine on retry.


def _load_retrieval():
    if SRC not in sys.path:
        sys.path.insert(0, SRC)
    import retrieval  # noqa: E402  (the production module, byte-identical chunker/tokenizer)
    return retrieval


class Advisor:
    """Lazy: the index is ~8.5k chunks and takes a few seconds to rank-build, so only load on first question."""

    def __init__(self):
        self._retrieval = None
        self.chunks = None
        self.bm25 = None
        self.error = None
        self.loaded_ms = None

    def ensure(self):
        if self.chunks is not None or self.error:
            return
        import time
        t0 = time.perf_counter()
        try:
            self._retrieval = _load_retrieval()
            self.chunks, self.bm25 = self._retrieval.load_index(INDEX)
            self.loaded_ms = round((time.perf_counter() - t0) * 1000)
        except Exception as e:  # surfaced in /api/advisor/status and on ask
            self.error = f"{type(e).__name__}: {e}"

    def status(self):
        self.ensure()
        return {"ready": self.chunks is not None, "chunks": len(self.chunks or []), "index": INDEX,
                "vault": VAULT, "persona_chars": len(PERSONA), "crisis_patterns":
                len(_PATTERNS), "top_k": TOP_K, "error": self.error,
                "load_ms": self.loaded_ms}

    def is_crisis(self, q):
        return bool(_CRISIS_RE and _CRISIS_RE.search(q))

    def _crisis_chunks(self):
        return [c for c in (self.chunks or []) if c["source"] == CRISIS_SOURCE]

    def gather(self, question):
        """Production retrieval shape: BM25 top-k, with the guardrails note force-injected on a crisis match."""
        self.ensure()
        if self.error:
            raise RuntimeError(self.error)
        hits = self._retrieval.retrieve_context(question, self.bm25, self.chunks, k=TOP_K)
        crisis = self.is_crisis(question)
        if crisis:
            injected = self._crisis_chunks()
            if injected:
                srcs = {c["source"] for c in injected}
                hits = injected + [h for h in hits if h["source"] not in srcs][:TOP_K]
        return hits, crisis

    def messages(self, question, hits):
        """Byte-for-byte the production prompt shape (parent_advisor_server.answer_question)."""
        context = "\n\n".join("Source: %s\n%s" % (h["source"], h["text"]) for h in hits)
        return [
            {"role": "system", "content": PERSONA},
            {"role": "user", "content": "Context:\n%s\n\n%sQuestion: %s" % (context, "", question)},
        ]

    def citations(self, sources, answer=""):
        """The production citation builder: titles, DOIs and URLs read from vault frontmatter."""
        try:
            return production().build_citations_grounded(sources, answer)
        except Exception:
            return []


# ---------- resolved last, once every helper above is bound ----------
PERSONA = _prod_attr("PARENT_SYSTEM", "")
CRISIS_SOURCE = _prod_attr("GUARDRAILS_SOURCE", "LanguageDeprivationAgent/01 Agent Scope and Guardrails.md")
MAX_Q = int(_prod_attr("MAX_Q", 500) or 500)
_PATTERNS = _prod_attr("CRISIS_PATTERNS", []) or []
_CRISIS_RE = re.compile("|".join(_PATTERNS), re.IGNORECASE) if _PATTERNS else None
