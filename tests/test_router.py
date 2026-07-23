"""
Suite de tests de los módulos locales del router (sin llamadas de red).

Cubre:
  - router/ranker.py    → chunk_text, rank_chunks, build_outline
  - router/sanitizer.py → strip_html, clean_text
  - router/ledger.py    → record/get, diff, purge, check_files
  - router/inbox.py     → send/check/complete/history

Todas las DBs usan tmp_path de pytest. Cero red, cero side-effects fuera del tmp.
"""

import asyncio
import importlib.util
import json
import time

import pytest

_HAS_FASTEMBED = importlib.util.find_spec("fastembed") is not None

from router.ranker import build_outline, chunk_text, rank_chunks
from router.sanitizer import clean_text, strip_html
from router.ledger import FileLedger
from router.inbox import Inbox


# ─── ranker.chunk_text ────────────────────────────────────────────────────────

def _make_sectioned_text(n_sections: int, lines_per_section: int, keyword_section: int,
                         keyword: str) -> str:
    """Texto con N secciones markdown; la keyword aparece solo en una sección."""
    blocks = []
    for s in range(n_sections):
        body_lines = []
        for l in range(lines_per_section):
            if s == keyword_section and l == 0:
                body_lines.append(f"This paragraph is about {keyword} and nothing else here.")
            else:
                body_lines.append(f"Filler content line {l} of ordinary prose padding text.")
        blocks.append(f"# Section {s}\n" + "\n".join(body_lines))
    return "\n\n".join(blocks)


def test_chunk_text_respects_line_boundaries():
    """Cada chunk debe corresponder exactamente a las líneas [start_line, end_line]."""
    text = _make_sectioned_text(8, 40, keyword_section=3, keyword="quokka")
    lines = text.split("\n")
    chunks = chunk_text(text)

    assert len(chunks) > 1, "un texto grande debe producir varios chunks"
    for c in chunks:
        assert 1 <= c.start_line <= c.end_line <= len(lines)
        expected = "\n".join(lines[c.start_line - 1:c.end_line])
        assert c.text == expected, "el texto del chunk no coincide con sus líneas de origen"


def test_chunk_text_small_input_single_chunk():
    text = "línea uno\nlínea dos\nlínea tres"
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 3
    assert chunks[0].text == text


def test_chunk_text_empty_and_blank():
    assert chunk_text("") == []
    assert chunk_text("\n\n   \n") == []


# ─── ranker.rank_chunks ───────────────────────────────────────────────────────

def test_rank_chunks_finds_relevant_chunk_by_query():
    keyword = "quokka"
    text = _make_sectioned_text(8, 40, keyword_section=3, keyword=keyword)
    chunks_total = chunk_text(text)
    top_k = 2
    assert len(chunks_total) > top_k, "el test necesita más chunks que top_k para que rankee"

    top, engine = rank_chunks(text, f"tell me about the {keyword}", top_k=top_k)

    assert len(top) == top_k
    assert engine.startswith("bm25"), f"sin fastembed el motor debe ser bm25, fue {engine}"
    combined = "\n".join(c.text for c in top)
    assert keyword in combined, "el chunk relevante debe estar entre los top-k"


def test_rank_chunks_results_sorted_by_position():
    text = _make_sectioned_text(8, 40, keyword_section=5, keyword="quokka")
    top, _ = rank_chunks(text, "quokka", top_k=3)
    starts = [c.start_line for c in top]
    assert starts == sorted(starts), "los chunks devueltos deben venir en orden de lectura"


def test_rank_chunks_returns_all_when_few_chunks():
    text = "línea uno\nlínea dos"
    top, engine = rank_chunks(text, "cualquier cosa", top_k=4)
    assert engine == "all"
    assert len(top) == len(chunk_text(text))


def test_rank_chunks_empty_text():
    top, engine = rank_chunks("", "query", top_k=4)
    assert top == []
    assert engine == "none"


# ─── ranker.build_outline ─────────────────────────────────────────────────────

def test_build_outline_captures_structure():
    text = (
        "# Título principal\n"
        "algo de prosa que no debe aparecer\n"
        "def funcion_uno(x):\n"
        "    return x\n"
        "class MiClase:\n"
        "    pass\n"
        "## Subsección\n"
    )
    outline = build_outline(text)

    joined = "\n".join(outline)
    assert "def funcion_uno(x):" in joined
    assert "class MiClase:" in joined
    assert "# Título principal" in joined
    assert "## Subsección" in joined
    assert "algo de prosa que no debe aparecer" not in joined
    # Cada entrada lleva prefijo de línea Ln:
    for entry in outline:
        assert entry.startswith("L")
        assert ":" in entry


def test_build_outline_respects_max_items():
    text = "\n".join(f"def f{i}(): pass" for i in range(100))
    outline = build_outline(text, max_items=10)
    assert len(outline) == 10


# ─── sanitizer.strip_html ─────────────────────────────────────────────────────

def test_strip_html_removes_scripts_and_styles():
    html = (
        "<html><head><style>.x{color:red}</style></head>"
        "<body><script>alert('malware xyz')</script>"
        "<p>Contenido visible importante</p>"
        "<div>Segundo parrafo</div></body></html>"
    )
    out = strip_html(html)

    assert "Contenido visible importante" in out
    assert "Segundo parrafo" in out
    assert "alert" not in out
    assert "malware xyz" not in out
    assert "color:red" not in out
    assert "<script" not in out and "<style" not in out


def test_strip_html_unescapes_entities():
    out = strip_html("<p>1 &lt; 2 &amp;&amp; 3 &gt; 0</p>")
    assert "1 < 2 && 3 > 0" in out


# ─── sanitizer.clean_text ─────────────────────────────────────────────────────

def test_clean_text_preserves_code_indentation():
    code = "def f():\n    if True:\n        x  =  1\n        return  x"
    out = clean_text(code, is_code=True)

    assert "    if True:" in out, "la indentación de 4 espacios debe conservarse"
    assert "        x  =  1" in out, "en código los espacios internos no se colapsan"
    assert "        return  x" in out


def test_clean_text_collapses_spaces_in_prose():
    prose = "esto    tiene     muchos      espacios"
    out = clean_text(prose, is_code=False)
    assert out == "esto tiene muchos espacios"


def test_clean_text_does_not_touch_fenced_code():
    text = "prosa   con   espacios\n```\ncode    with    spaces\n```\nmas   prosa"
    out = clean_text(text, is_code=False)
    assert "code    with    spaces" in out, "el bloque cercado no debe colapsarse"
    assert "prosa con espacios" in out
    assert "mas prosa" in out


# ─── sanitizer.sanitize_file_content: la extensión manda sobre la heurística ──

def test_code_file_with_html_literals_is_never_stripped():
    """
    Regresión: un .py con literales HTML (scraper, template, fixtures) se daba
    por página web y se le borraban los tags → código corrupto. La fidelidad
    exacta es la promesa central del producto.
    """
    from router.sanitizer import sanitize_file_content
    code = (
        'import re\n'
        '_RE = re.compile(r"<!DOCTYPE\\s+html|<html\\b|<body\\b|<head\\b", re.I)\n'
        'TEMPLATE = "<div class=\'card\'><p>hola</p></div>"\n'
        'def render(): return TEMPLATE\n'
    )
    clean, was_html = sanitize_file_content(code, "scraper.py")

    assert was_html is False, "un archivo de código nunca debe procesarse como HTML"
    assert "<html\\b" in clean and "<body\\b" in clean, "el regex debe sobrevivir intacto"
    assert "<div class='card'><p>hola</p></div>" in clean, "el template debe sobrevivir intacto"
    assert clean.strip() == code.strip(), "cero pérdida: el código vuelve tal cual"


def test_real_html_file_is_still_stripped():
    """El arreglo no debe romper el caso legítimo: HTML de verdad sí se limpia."""
    from router.sanitizer import sanitize_file_content
    html = "<html><body><style>.x{color:red}</style><script>evil()</script><p>texto visible</p></body></html>"
    clean, was_html = sanitize_file_content(html, "page.html")

    assert was_html is True
    assert "texto visible" in clean
    assert "evil()" not in clean and "color:red" not in clean


def test_clean_text_normalizes_blank_lines_and_crlf():
    text = "a\r\n\r\n\r\n\r\nb"
    out = clean_text(text)
    assert "\r" not in out
    assert "a\n\nb" == out


# ─── ledger: record / get ─────────────────────────────────────────────────────

@pytest.fixture
def ledger(tmp_path):
    led = FileLedger(str(tmp_path / "ledger.db"))
    yield led
    led.close()


def test_ledger_record_and_get(ledger):
    content = "hola mundo\nsegunda linea"
    outline = ["L1:hola mundo"]
    ledger.record("/fake/path.py", content, outline, tokens=5)

    got = ledger.get("/fake/path.py")
    assert got is not None
    assert got["hash"] == FileLedger.hash(content)
    assert got["snapshot"] == content
    assert got["outline"] == outline
    assert got["tokens"] == 5
    assert got["reads"] == 1


def test_ledger_get_missing_returns_none(ledger):
    assert ledger.get("/no/existe.py") is None


def test_ledger_record_upsert_increments_reads(ledger):
    ledger.record("/p.py", "v1", [], tokens=1)
    ledger.record("/p.py", "v2 contenido nuevo", ["L1:v2"], tokens=2)

    got = ledger.get("/p.py")
    assert got["reads"] == 2
    assert got["snapshot"] == "v2 contenido nuevo"
    assert got["hash"] == FileLedger.hash("v2 contenido nuevo")
    assert got["tokens"] == 2


# ─── ledger: diff ─────────────────────────────────────────────────────────────

def test_ledger_diff_identical_returns_empty(ledger):
    assert ledger.diff("misma cosa", "misma cosa") == ""


def test_ledger_diff_reports_changes(ledger):
    # Archivo grande con un único cambio: el diff pesa poco respecto al total,
    # así que rinde y se devuelve (no None).
    base = [f"linea numero {i} con contenido estable de relleno" for i in range(60)]
    old = "\n".join(base)
    changed = list(base)
    changed[30] = "linea numero 30 MODIFICADA con texto nuevo"
    new = "\n".join(changed)

    d = ledger.diff(old, new)
    assert d is not None and d != ""
    assert "linea numero 30 MODIFICADA con texto nuevo" in d
    assert d.startswith("--- antes")


def test_ledger_diff_too_large_returns_none(ledger):
    old = "\n".join(f"linea original numero {i} con texto de relleno" for i in range(50))
    new = "totalmente distinto"
    # El diff (borra 50 líneas + agrega 1) pesa muchísimo más que `new` → no rinde.
    assert ledger.diff(old, new) is None


# ─── ledger: purge por edad ───────────────────────────────────────────────────

def test_ledger_purge_removes_old_entries(ledger):
    ledger.record("/viejo.py", "contenido", [], tokens=1)
    old_ts = time.time() - 40 * 86400  # 40 días atrás (> PURGE_AFTER_DAYS=30)
    ledger._db.execute("UPDATE file_ledger SET last_seen=? WHERE path=?", (old_ts, "/viejo.py"))
    ledger._db.commit()

    result = ledger.purge()
    assert result["expired"] >= 1
    assert ledger.get("/viejo.py") is None


def test_ledger_purge_keeps_recent_entries(ledger):
    ledger.record("/nuevo.py", "contenido", [], tokens=1)
    result = ledger.purge()
    assert result["expired"] == 0
    assert ledger.get("/nuevo.py") is not None


# ─── ledger: check_files ──────────────────────────────────────────────────────

def test_ledger_check_files_detects_states(ledger, tmp_path):
    f = tmp_path / "archivo.py"
    content = "print('hola')\n"
    f.write_text(content, encoding="utf-8")
    good_hash = FileLedger.hash(content)

    result = ledger.check_files([
        {"path": str(f), "hash": good_hash},
        {"path": str(f), "hash": "hash_incorrecto"},
        {"path": str(tmp_path / "no_existe.py"), "hash": "x"},
    ])

    states = {(r["path"], r["state"]) for r in result}
    assert (str(f), "unchanged") in states
    assert (str(f), "changed") in states
    assert (str(tmp_path / "no_existe.py"), "deleted") in states


# ─── inbox: send / check / complete / history ─────────────────────────────────

@pytest.fixture
def inbox(tmp_path):
    ib = Inbox(str(tmp_path / "inbox.db"))
    yield ib
    ib.close()


def test_inbox_send_returns_id_and_check_lists_it(inbox):
    oid = inbox.send("migrar tests a pytest", to_client="code", from_client="cowork",
                     checkpoint="cp-1")
    assert isinstance(oid, int) and oid > 0

    pending = inbox.check("code")
    assert len(pending) == 1
    order = pending[0]
    assert order["id"] == oid
    assert order["to"] == "code"
    assert order["from"] == "cowork"
    assert order["message"] == "migrar tests a pytest"
    assert order["checkpoint"] == "cp-1"


def test_inbox_routing_any_and_targeted(inbox):
    inbox.send("para code", to_client="code")
    inbox.send("para cualquiera", to_client="any")

    code_view = inbox.check("code")
    desktop_view = inbox.check("desktop")

    code_msgs = {o["message"] for o in code_view}
    desktop_msgs = {o["message"] for o in desktop_view}

    # code ve lo suyo + lo dirigido a 'any'
    assert code_msgs == {"para code", "para cualquiera"}
    # desktop solo ve 'any', no lo dirigido a 'code'
    assert desktop_msgs == {"para cualquiera"}


def test_inbox_complete_moves_to_history(inbox):
    oid = inbox.send("hacer algo", to_client="code")
    assert inbox.complete(oid, result="34/34 verdes") is True

    assert inbox.check("code") == []

    hist = inbox.history()
    assert len(hist) == 1
    assert hist[0]["id"] == oid
    assert hist[0]["result"] == "34/34 verdes"


def test_inbox_complete_twice_is_false(inbox):
    oid = inbox.send("una vez", to_client="code")
    assert inbox.complete(oid) is True
    assert inbox.complete(oid) is False, "una orden ya completada no puede completarse de nuevo"


def test_inbox_complete_unknown_id_is_false(inbox):
    assert inbox.complete(99999) is False


def test_inbox_history_respects_limit(inbox):
    for i in range(5):
        oid = inbox.send(f"orden {i}", to_client="code")
        inbox.complete(oid, result=f"r{i}")
    hist = inbox.history(limit=3)
    assert len(hist) == 3
    # history ordena por done_at DESC → la última completada primero
    assert hist[0]["message"] == "orden 4"


# ─── inbox: cliente 'design' + assets de handoff ──────────────────────────────

def test_inbox_design_client_roundtrip_with_assets(inbox):
    # Claude Code le deja una orden a Claude Design, con material de handoff
    oid = inbox.send(
        "hero de la landing, dark + acento cyan",
        to_client="design", from_client="code", checkpoint="landing-v2",
        assets=["/proj/brief.md", "https://cdn/wireframe.png"],
    )
    pending = inbox.check("design")
    assert len(pending) == 1
    order = pending[0]
    assert order["to"] == "design"
    assert order["from"] == "code"
    assert order["assets"] == ["/proj/brief.md", "https://cdn/wireframe.png"]

    # Design completa y devuelve sus propios assets (el export del mockup)
    assert inbox.complete(oid, result="mockup listo, 2 variantes",
                          assets=["https://figma/hero", "/exports/hero.png"]) is True

    hist = inbox.history()
    assert hist[0]["result"] == "mockup listo, 2 variantes"
    assert hist[0]["result_assets"] == ["https://figma/hero", "/exports/hero.png"]


def test_inbox_assets_default_empty_and_normalizes_string(inbox):
    # sin assets → lista vacía en el check
    oid = inbox.send("sin material", to_client="design")
    assert inbox.check("design")[0]["assets"] == []
    inbox.complete(oid)
    assert inbox.history()[0]["result_assets"] == []

    # un asset como string suelto se normaliza a lista
    oid2 = inbox.send("un solo asset", to_client="design", assets="/x/spec.md")
    assert inbox.check("design")[0]["assets"] == ["/x/spec.md"]


def test_inbox_migration_adds_columns_to_legacy_db(tmp_path):
    # DB "vieja" sin las columnas nuevas → el __init__ debe migrarla sin romper
    import sqlite3
    dbp = str(tmp_path / "legacy_inbox.db")
    con = sqlite3.connect(dbp)
    con.execute("""
        CREATE TABLE inbox (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            to_client TEXT NOT NULL DEFAULT 'any',
            from_client TEXT NOT NULL DEFAULT 'unknown',
            message TEXT NOT NULL,
            checkpoint TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            result TEXT,
            created_at REAL NOT NULL,
            done_at REAL
        )
    """)
    con.execute("INSERT INTO inbox (message, created_at) VALUES ('orden vieja', ?)",
                (1_700_000_000.0,))
    con.commit(); con.close()

    ib = Inbox(dbp)  # dispara _migrate
    try:
        cols = {r[1] for r in ib._db.execute("PRAGMA table_info(inbox)").fetchall()}
        assert "assets" in cols and "result_assets" in cols
        # la orden vieja sigue visible y con assets vacíos
        pend = ib.check()
        assert len(pend) == 1 and pend[0]["assets"] == []
        # y podemos mandar una nueva con assets sobre la DB migrada
        oid = ib.send("nueva", to_client="design", assets=["/a.png"])
        assert ib.check("design")[0]["assets"] == ["/a.png"] or \
               any(o["id"] == oid for o in ib.check("design"))
    finally:
        ib.close()


# ─── inbox: no acumular duplicados ────────────────────────────────────────────

def test_inbox_send_exact_duplicate_reuses_existing_id(inbox):
    oid1 = inbox.send("migrar tests a pytest", to_client="code")
    oid2 = inbox.send("migrar tests a pytest", to_client="code")  # idéntica, aún pendiente
    assert oid1 == oid2, "una orden idéntica pendiente no debe crear una fila nueva"
    assert len(inbox.check("code")) == 1


def test_inbox_send_duplicate_ignores_whitespace_and_case(inbox):
    oid1 = inbox.send("Migrar  tests   a pytest", to_client="code")
    oid2 = inbox.send("migrar tests a pytest", to_client="code")
    assert oid1 == oid2
    assert len(inbox.check("code")) == 1


def test_inbox_send_same_message_different_client_is_not_duplicate(inbox):
    oid1 = inbox.send("mismo mensaje", to_client="code")
    oid2 = inbox.send("mismo mensaje", to_client="design")
    assert oid1 != oid2
    assert len(inbox.check("code")) == 1
    assert len(inbox.check("design")) == 1


def test_inbox_send_after_complete_is_not_duplicate(inbox):
    oid1 = inbox.send("tarea repetible", to_client="code")
    inbox.complete(oid1, result="listo")
    oid2 = inbox.send("tarea repetible", to_client="code")  # la anterior ya no está pendiente
    assert oid2 != oid1
    assert len(inbox.check("code")) == 1


def test_inbox_send_allow_duplicate_bypasses_guard(inbox):
    oid1 = inbox.send("forzada", to_client="code")
    oid2 = inbox.send("forzada", to_client="code", allow_duplicate=True)
    assert oid1 != oid2
    assert len(inbox.check("code")) == 2


def test_inbox_find_pending_duplicate_returns_none_when_no_match(inbox):
    inbox.send("algo", to_client="code")
    assert inbox.find_pending_duplicate("code", "otra cosa") is None
    assert inbox.find_pending_duplicate("design", "algo") is None  # destino distinto, no matchea


# ─── server: detección de código desactualizado en el proceso vivo ────────────
# Un proceso MCP no recarga módulos solo: si server.py/router/*.py cambian en
# disco después del arranque, sigue corriendo la versión vieja hasta reiniciar.
# _code_staleness() compara mtimes contra el boot_time del proceso para hacer
# esa desincronización observable vía router_status en lugar de tener que
# inferirla a mano.

@pytest.fixture(scope="module")
def server_module():
    """Importa server.py como módulo (requiere el paquete `mcp` instalado)."""
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "individra_server", Path(__file__).parent.parent / "server.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_code_staleness_false_right_after_boot(server_module):
    result = server_module._code_staleness(server_module._stats["start_time"])
    assert result == {"stale": False}


def test_code_staleness_true_when_boot_predates_files(server_module):
    # boot_time muy en el pasado → cualquier archivo watched "cambió después"
    result = server_module._code_staleness(0.0)
    assert result["stale"] is True
    assert result["changed_file"] in ("server.py", "inbox.py", "ledger.py", "ranker.py", "sanitizer.py", "__init__.py")
    assert isinstance(result["changed_ago_s"], int)
    assert "reiniciá el MCP" in result["hint"]


def test_code_staleness_false_when_boot_is_in_the_future(server_module):
    # boot_time posterior a cualquier mtime real → nada "cambió después"
    result = server_module._code_staleness(time.time() + 3600)
    assert result == {"stale": False}


# ─── ledger: cache de ranking por query (#1) ──────────────────────────────────

def test_normalize_query_collapses_spaces_and_casefolds():
    assert FileLedger.normalize_query("  Buscar   Los  Webhooks ") == "buscar los webhooks"
    assert FileLedger.normalize_query("MISMA") == FileLedger.normalize_query("misma")


def test_ledger_query_cache_roundtrip_and_key_distinction(ledger):
    ledger.put_query_cache("h1", "webhooks", 4, [(1, 10), (20, 25)])
    assert ledger.get_query_cache("h1", "webhooks", 4) == [(1, 10), (20, 25)]
    # distinto top_k, query o hash → miss (son parte de la clave)
    assert ledger.get_query_cache("h1", "webhooks", 2) is None
    assert ledger.get_query_cache("h1", "otra", 4) is None
    assert ledger.get_query_cache("h2", "webhooks", 4) is None


def test_ledger_query_cache_invalidates_by_content_hash(ledger):
    # el hash del contenido ES la clave → contenido nuevo = hash nuevo = miss
    h_old = FileLedger.hash("contenido viejo")
    h_new = FileLedger.hash("contenido nuevo distinto")
    ledger.put_query_cache(h_old, "q", 4, [(1, 5)])
    assert ledger.get_query_cache(h_old, "q", 4) == [(1, 5)]
    assert ledger.get_query_cache(h_new, "q", 4) is None


# ─── ledger: cache de vectores de embeddings (#2) ─────────────────────────────

def test_ledger_chunk_vectors_roundtrip(ledger):
    vecs = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    ledger.put_chunk_vectors("hashA", vecs)
    got = ledger.get_chunk_vectors("hashA")
    assert got is not None and len(got) == 2
    for orig, back in zip(vecs, got):
        assert all(abs(a - b) < 1e-6 for a, b in zip(orig, back)), "round-trip float32 debe conservar valores"


def test_ledger_chunk_vectors_missing_returns_none(ledger):
    assert ledger.get_chunk_vectors("no_existe") is None


def test_ledger_chunk_vectors_replaces_on_reput(ledger):
    ledger.put_chunk_vectors("h", [[1.0], [2.0], [3.0]])
    ledger.put_chunk_vectors("h", [[9.0]])  # reemplaza, no acumula
    got = ledger.get_chunk_vectors("h")
    assert len(got) == 1
    assert abs(got[0][0] - 9.0) < 1e-6


def test_ledger_purge_removes_orphan_caches(ledger):
    # file_hash sin entrada en file_ledger → cachés huérfanos, se limpian en purge
    ledger.put_query_cache("orphan", "q", 4, [(1, 5)])
    ledger.put_chunk_vectors("orphan", [[1.0, 2.0]])
    ledger.purge()
    assert ledger.get_query_cache("orphan", "q", 4) is None
    assert ledger.get_chunk_vectors("orphan") is None


def test_ledger_purge_keeps_caches_for_tracked_file(ledger):
    content = "algo de contenido trackeado"
    h = FileLedger.hash(content)
    ledger.record("/tracked.py", content, [], tokens=1)  # mete el hash en file_ledger
    ledger.put_query_cache(h, "q", 4, [(1, 3)])
    ledger.put_chunk_vectors(h, [[0.5, 0.5]])
    ledger.purge()
    assert ledger.get_query_cache(h, "q", 4) == [(1, 3)]
    assert ledger.get_chunk_vectors(h) is not None


# ─── ranker: reuso de vectores no cambia el ranking ───────────────────────────

def test_rank_chunks_vector_store_param_does_not_change_bm25(ledger):
    """Pasar file_hash/vector_store no altera el ranking (backward-compatible)."""
    text = _make_sectioned_text(8, 40, keyword_section=3, keyword="quokka")
    top_a, eng_a = rank_chunks(text, "quokka", top_k=2)
    top_b, eng_b = rank_chunks(text, "quokka", top_k=2, file_hash="h", vector_store=ledger)
    assert eng_a == eng_b
    assert [(c.start_line, c.end_line) for c in top_a] == [(c.start_line, c.end_line) for c in top_b]


@pytest.mark.skipif(not _HAS_FASTEMBED, reason="fastembed no instalado — #2 no aplica")
def test_rank_chunks_cached_vectors_same_ranking(ledger):
    text = _make_sectioned_text(8, 40, keyword_section=3, keyword="quokka")
    # 1ª pasada puebla el cache de vectores; 2ª lo reusa → mismo ranking y scores
    top1, _ = rank_chunks(text, "quokka marsupial", top_k=3, file_hash="hv", vector_store=ledger)
    top2, _ = rank_chunks(text, "quokka marsupial", top_k=3, file_hash="hv", vector_store=ledger)
    assert [(c.start_line, c.end_line, c.score) for c in top1] == \
           [(c.start_line, c.end_line, c.score) for c in top2]


# ─── server: cache de query end-to-end en router_smart_read ───────────────────

def _smart_read(server_module, **kw) -> dict:
    params = server_module.SmartReadInput(**kw)
    return json.loads(asyncio.run(server_module.router_smart_read(params)))


def test_smart_read_query_cache_hit_returns_same_ranges(server_module, tmp_path, monkeypatch):
    led = FileLedger(str(tmp_path / "led.db"))
    monkeypatch.setattr(server_module, "ledger", led)
    f = tmp_path / "big.py"
    f.write_text(_make_sectioned_text(8, 40, keyword_section=3, keyword="quokka"), encoding="utf-8")

    r1 = _smart_read(server_module, file_path=str(f), query="quokka", top_k=2)
    assert r1["status"] == "chunks"
    assert r1["cache_hit"] is False
    ranges1 = [c["lines"] for c in r1["chunks"]]

    r2 = _smart_read(server_module, file_path=str(f), query="quokka", top_k=2)
    assert r2["cache_hit"] is True
    assert r2["engine"] == "cache"
    assert [c["lines"] for c in r2["chunks"]] == ranges1, "el cache debe devolver los mismos ranges"
    led.close()


def test_smart_read_query_cache_invalidated_on_file_change(server_module, tmp_path, monkeypatch):
    led = FileLedger(str(tmp_path / "led2.db"))
    monkeypatch.setattr(server_module, "ledger", led)
    f = tmp_path / "big2.py"
    f.write_text(_make_sectioned_text(8, 40, keyword_section=3, keyword="quokka"), encoding="utf-8")

    assert _smart_read(server_module, file_path=str(f), query="quokka", top_k=2)["cache_hit"] is False
    assert _smart_read(server_module, file_path=str(f), query="quokka", top_k=2)["cache_hit"] is True

    # el archivo cambia → nuevo hash → el cache queda inalcanzable
    f.write_text(_make_sectioned_text(9, 45, keyword_section=6, keyword="quokka"), encoding="utf-8")
    r3 = _smart_read(server_module, file_path=str(f), query="quokka", top_k=2)
    assert r3["cache_hit"] is False, "un archivo modificado debe invalidar el cache de query"
    led.close()


def test_smart_read_force_full_bypasses_query_cache(server_module, tmp_path, monkeypatch):
    led = FileLedger(str(tmp_path / "led3.db"))
    monkeypatch.setattr(server_module, "ledger", led)
    f = tmp_path / "big3.py"
    f.write_text(_make_sectioned_text(8, 40, keyword_section=3, keyword="quokka"), encoding="utf-8")

    # popula el cache con una lectura normal
    assert _smart_read(server_module, file_path=str(f), query="quokka", top_k=2)["cache_hit"] is False
    # con force_full el cache no se usa aunque exista
    r = _smart_read(server_module, file_path=str(f), query="quokka", top_k=2, force_full=True)
    assert r["cache_hit"] is False
    led.close()


# ─── server: config en caliente (router_config.json) ─────────────────────────

@pytest.fixture
def config_at(server_module, tmp_path, monkeypatch):
    """Apunta _CONFIG_PATH a un tmp y resetea el cache de config (antes y después)."""
    path = tmp_path / "router_config.json"
    monkeypatch.setattr(server_module, "_CONFIG_PATH", path)
    server_module._config_cache["mtime"] = None
    server_module._config_cache["cfg"] = dict(server_module._CONFIG_DEFAULTS)
    yield path
    server_module._config_cache["mtime"] = None
    server_module._config_cache["cfg"] = dict(server_module._CONFIG_DEFAULTS)


def test_config_defaults_without_file(server_module, config_at):
    # (a) sin archivo → defaults exactos
    cfg = server_module._load_config()
    assert cfg == server_module._CONFIG_DEFAULTS
    assert cfg["full_return_max_tokens"] == 1500
    assert cfg["default_top_k"] == 4
    assert cfg["diff_max_ratio"] == 0.6
    assert cfg["cache_enabled"] is True


def test_config_file_overrides_value(server_module, config_at):
    # (b) el archivo overridea solo lo que declara; claves desconocidas se ignoran
    config_at.write_text(json.dumps({"default_top_k": 7, "clave_desconocida": 1}), encoding="utf-8")
    cfg = server_module._load_config()
    assert cfg["default_top_k"] == 7
    assert cfg["full_return_max_tokens"] == 1500, "lo no declarado mantiene su default"
    assert "clave_desconocida" not in cfg


def test_config_mtime_change_is_picked_up_without_reimport(server_module, config_at):
    # (c) editar el archivo se toma en la siguiente llamada, sin reimportar el módulo
    import os
    config_at.write_text(json.dumps({"default_top_k": 5}), encoding="utf-8")
    assert server_module._load_config()["default_top_k"] == 5

    config_at.write_text(json.dumps({"default_top_k": 9}), encoding="utf-8")
    os.utime(config_at, (time.time() + 10, time.time() + 10))  # mtime inequívocamente distinto
    assert server_module._load_config()["default_top_k"] == 9


def test_config_invalid_json_keeps_last_good(server_module, config_at):
    import os
    config_at.write_text(json.dumps({"default_top_k": 6}), encoding="utf-8")
    assert server_module._load_config()["default_top_k"] == 6
    config_at.write_text("{esto no es json", encoding="utf-8")
    os.utime(config_at, (time.time() + 10, time.time() + 10))
    assert server_module._load_config()["default_top_k"] == 6, "config rota no debe tirar los valores buenos"


def test_smart_read_cache_disabled_via_config(server_module, tmp_path, monkeypatch, config_at):
    # (d) cache_enabled=false → nunca hay cache_hit aunque se repita la query
    led = FileLedger(str(tmp_path / "led4.db"))
    monkeypatch.setattr(server_module, "ledger", led)
    config_at.write_text(json.dumps({"cache_enabled": False}), encoding="utf-8")
    f = tmp_path / "big4.py"
    f.write_text(_make_sectioned_text(8, 40, keyword_section=3, keyword="quokka"), encoding="utf-8")

    assert _smart_read(server_module, file_path=str(f), query="quokka", top_k=2)["cache_hit"] is False
    assert _smart_read(server_module, file_path=str(f), query="quokka", top_k=2)["cache_hit"] is False
    led.close()


def test_smart_read_default_top_k_comes_from_config(server_module, tmp_path, monkeypatch, config_at):
    led = FileLedger(str(tmp_path / "led5.db"))
    monkeypatch.setattr(server_module, "ledger", led)
    config_at.write_text(json.dumps({"default_top_k": 2}), encoding="utf-8")
    f = tmp_path / "big5.py"
    f.write_text(_make_sectioned_text(8, 40, keyword_section=3, keyword="quokka"), encoding="utf-8")

    r = _smart_read(server_module, file_path=str(f), query="quokka")  # sin top_k explícito
    assert r["status"] == "chunks"
    assert len(r["chunks"]) == 2, "sin top_k explícito debe usar default_top_k de la config"
    led.close()


# ─── server: code_stale inyectado en todas las tools (throttled) ──────────────

@pytest.fixture
def fresh_stale_cache(server_module):
    """Resetea el throttle de staleness antes y después de cada test."""
    server_module._stale_cache["ts"] = 0.0
    server_module._stale_cache["result"] = None
    yield
    server_module._stale_cache["ts"] = 0.0
    server_module._stale_cache["result"] = None


def test_smart_read_payload_has_code_stale_when_boot_is_old(
        server_module, tmp_path, monkeypatch, fresh_stale_cache):
    # (e) boot "viejo" → cualquier tool avisa code_stale en su payload
    led = FileLedger(str(tmp_path / "led6.db"))
    monkeypatch.setattr(server_module, "ledger", led)
    monkeypatch.setitem(server_module._stats, "start_time", 0.0)
    f = tmp_path / "chico.py"
    f.write_text("print('hola')", encoding="utf-8")

    r = _smart_read(server_module, file_path=str(f))
    assert "code_stale" in r
    assert r["code_stale"]["changed_file"] is not None
    assert "reconect" in r["code_stale"]["hint"]
    led.close()


def test_smart_read_payload_omits_code_stale_when_fresh(
        server_module, tmp_path, monkeypatch, fresh_stale_cache):
    # (f) CERO SORPRESAS: boot reciente (código sin cambios después) → el campo NO existe
    led = FileLedger(str(tmp_path / "led7.db"))
    monkeypatch.setattr(server_module, "ledger", led)
    monkeypatch.setitem(server_module._stats, "start_time", time.time() + 3600)
    f = tmp_path / "chico2.py"
    f.write_text("print('hola')", encoding="utf-8")

    r = _smart_read(server_module, file_path=str(f))
    assert "code_stale" not in r, "en operación normal el campo no debe existir en la respuesta"
    led.close()


def test_stale_check_is_throttled(server_module, monkeypatch, fresh_stale_cache):
    # El chequeo de mtimes corre 1 vez por ventana; entre medio sirve el cache
    calls = {"n": 0}
    real = server_module._code_staleness

    def counting(boot):
        calls["n"] += 1
        return real(boot)

    monkeypatch.setattr(server_module, "_code_staleness", counting)
    for _ in range(10):
        server_module._stale_throttled()
    assert calls["n"] == 1, "10 llamadas dentro de la ventana deben statear archivos una sola vez"


# ─── schema versioning (PRAGMA user_version) ──────────────────────────────────

def test_ledger_fresh_db_gets_schema_version(tmp_path):
    from router.ledger import LEDGER_SCHEMA_VERSION
    led = FileLedger(str(tmp_path / "v.db"))
    try:
        assert led._db.execute("PRAGMA user_version").fetchone()[0] == LEDGER_SCHEMA_VERSION
    finally:
        led.close()


def test_ledger_pre_versioning_db_is_migrated(tmp_path):
    # DB "vieja" (schema actual pero user_version=0) → al abrir queda versionada
    import sqlite3
    from router.ledger import LEDGER_SCHEMA_VERSION
    dbp = str(tmp_path / "old.db")
    con = sqlite3.connect(dbp)
    con.execute("CREATE TABLE file_ledger (path TEXT PRIMARY KEY, hash TEXT NOT NULL, snapshot TEXT, outline TEXT, tokens INTEGER, first_seen REAL, last_seen REAL, reads INTEGER DEFAULT 1)")
    con.commit(); con.close()
    led = FileLedger(dbp)
    try:
        assert led._db.execute("PRAGMA user_version").fetchone()[0] == LEDGER_SCHEMA_VERSION
        led.record("/x.py", "contenido", [], tokens=1)  # y sigue operativa
        assert led.get("/x.py") is not None
    finally:
        led.close()


def test_inbox_fresh_db_gets_schema_version(tmp_path):
    from router.inbox import INBOX_SCHEMA_VERSION
    ib = Inbox(str(tmp_path / "vi.db"))
    try:
        assert ib._db.execute("PRAGMA user_version").fetchone()[0] == INBOX_SCHEMA_VERSION
    finally:
        ib.close()


# ─── ledger.recent_files (base del arranque en frío) ──────────────────────────

def test_ledger_recent_files_orders_and_states(ledger, tmp_path):
    fa = tmp_path / "a.py"
    fb = tmp_path / "b.py"
    fa.write_text("contenido a", encoding="utf-8")
    fb.write_text("contenido b", encoding="utf-8")
    ledger.record(str(fa), "contenido a", [], tokens=1)
    time.sleep(0.02)  # last_seen distinguible
    ledger.record(str(fb), "contenido b", [], tokens=1)
    fb.write_text("contenido b MODIFICADO", encoding="utf-8")

    recent = ledger.recent_files(10)
    assert [r["path"] for r in recent] == [str(fb), str(fa)], "orden: último leído primero"
    states = {r["path"]: r["state"] for r in recent}
    assert states[str(fa)] == "unchanged"
    assert states[str(fb)] == "changed"


# ─── server: arranque en frío (digest determinista) ───────────────────────────

def _checkpoint(server_module, **kw) -> dict:
    params = server_module.CheckpointInput(**kw)
    return json.loads(asyncio.run(server_module.router_checkpoint(params)))


def test_resume_without_checkpoints_returns_reconstructed_digest(
        server_module, tmp_path, monkeypatch):
    led = FileLedger(str(tmp_path / "dig.db"))
    ib = Inbox(str(tmp_path / "dig_inbox.db"))
    monkeypatch.setattr(server_module, "ledger", led)
    monkeypatch.setattr(server_module, "inbox", ib)
    monkeypatch.setattr(server_module, "_checkpoints_dir", tmp_path / "no_checkpoints")

    f = tmp_path / "trabajado.py"
    f.write_text("codigo", encoding="utf-8")
    led.record(str(f), "codigo", [], tokens=2)
    oid = ib.send("tarea vieja", to_client="code")
    ib.complete(oid, result="hecha")

    r = _checkpoint(server_module, action="resume")
    assert r["status"] == "resumed"
    assert r["mode"] == "reconstructed_activity", "el LLM debe saber que NO es un checkpoint intencional"
    assert "NO un checkpoint intencional" in r["note"]
    assert any(rf["path"] == str(f) for rf in r["recent_files"])
    assert any(o["result"] == "hecha" for o in r["recent_orders"])
    led.close(); ib.close()


def test_resume_with_checkpoint_does_not_use_digest(server_module, tmp_path, monkeypatch):
    monkeypatch.setattr(server_module, "_checkpoints_dir", tmp_path / "cps")
    saved = _checkpoint(server_module, action="save", name="real-cp",
                        summary="checkpoint real guardado por alguien")
    assert saved["status"] == "saved"
    r = _checkpoint(server_module, action="resume")
    assert r.get("mode") is None, "habiendo checkpoint real, el digest no debe activarse"
    assert r["summary"] == "checkpoint real guardado por alguien"


# ─── server: MCP prompts (/resume, /handoff, /inbox) ──────────────────────────

def test_prompts_exist_and_reference_the_tools(server_module):
    r = server_module.prompt_resume()
    assert "router_checkpoint" in r and "router_inbox" in r
    assert "reconstructed_activity" in r, "el prompt debe advertir sobre el modo digest"

    h = server_module.prompt_handoff(to="design", message="hacer el hero")
    assert "router_checkpoint" in h and "router_inbox" in h
    assert "'design'" in h and "hacer el hero" in h

    i = server_module.prompt_inbox()
    assert "action='check'" in i and "action='complete'" in i


def test_prompt_handoff_without_args_asks_the_user(server_module):
    h = server_module.prompt_handoff()
    assert "preguntale al usuario" in h


# ─── rules: reglas de proyecto con procedencia (módulo) ───────────────────────

from router import rules as R


def test_rules_add_and_load_with_provenance(tmp_path):
    rule, created = R.add_rule(tmp_path, "nunca usar Redux", source_client="code", checkpoint="cp-arch")
    assert created is True
    assert rule["text"] == "nunca usar Redux"
    assert rule["source_client"] == "code"
    assert rule["checkpoint"] == "cp-arch"
    assert rule["id"] == 1

    loaded = R.load_rules(tmp_path)
    assert len(loaded) == 1 and loaded[0]["text"] == "nunca usar Redux"
    # se persiste como JSON legible en la raíz del proyecto
    assert R.rules_path(tmp_path).is_file()


def test_rules_add_dedup_is_case_and_space_insensitive(tmp_path):
    R.add_rule(tmp_path, "Los tests van en tests/")
    rule2, created2 = R.add_rule(tmp_path, "los   tests  van en tests/")
    assert created2 is False, "una regla equivalente no debe duplicarse"
    assert len(R.load_rules(tmp_path)) == 1
    assert rule2["id"] == 1


def test_rules_remove(tmp_path):
    R.add_rule(tmp_path, "regla A")
    r2, _ = R.add_rule(tmp_path, "regla B")
    assert R.remove_rule(tmp_path, r2["id"]) is True
    assert [r["text"] for r in R.load_rules(tmp_path)] == ["regla A"]
    assert R.remove_rule(tmp_path, 999) is False


def test_rules_load_missing_and_corrupt_are_safe(tmp_path):
    assert R.load_rules(tmp_path) == []  # sin archivo
    R.rules_path(tmp_path).write_text("{roto", encoding="utf-8")
    assert R.load_rules(tmp_path) == []  # corrupto → [], no lanza


def test_rules_walk_up_discovery(tmp_path):
    # reglas en la raíz del proyecto; un archivo anidado debe encontrarlas
    R.add_rule(tmp_path, "regla raíz")
    nested = tmp_path / "src" / "deep"
    nested.mkdir(parents=True)
    f = nested / "modulo.py"
    f.write_text("x = 1", encoding="utf-8")
    found = R.rules_for_file(str(f))
    assert len(found) == 1 and found[0]["text"] == "regla raíz"


def test_rules_sync_to_claudemd_delimited_and_idempotent(tmp_path):
    R.add_rule(tmp_path, "nunca usar Redux", source_client="code")
    md = R.sync_to_claudemd(tmp_path)
    content = md.read_text(encoding="utf-8")
    assert R.CLAUDEMD_START in content and R.CLAUDEMD_END in content
    assert "nunca usar Redux" in content

    # segundo sync no duplica la sección ni pisa contenido del usuario
    content = "# Mi proyecto\n\nNotas mías.\n\n" + content
    md.write_text(content, encoding="utf-8")
    R.add_rule(tmp_path, "otra regla")
    R.sync_to_claudemd(tmp_path)
    final = md.read_text(encoding="utf-8")
    assert final.count(R.CLAUDEMD_START) == 1, "no debe duplicar la sección"
    assert "Notas mías." in final, "no debe pisar el contenido del usuario"
    assert "otra regla" in final


# ─── rules: tool router_rules + inyección en payloads ─────────────────────────

def _rules_tool(server_module, **kw) -> dict:
    params = server_module.RulesInput(**kw)
    return json.loads(asyncio.run(server_module.router_rules(params)))


def test_router_rules_add_list_remove(server_module, tmp_path):
    r = _rules_tool(server_module, action="add", project_dir=str(tmp_path),
                    text="nunca usar Redux", from_client="code")
    assert r["status"] == "added"
    assert _rules_tool(server_module, action="add", project_dir=str(tmp_path),
                       text="nunca usar Redux")["status"] == "duplicate"

    listed = _rules_tool(server_module, action="list", project_dir=str(tmp_path))
    assert len(listed["rules"]) == 1

    rid = r["rule"]["id"]
    assert _rules_tool(server_module, action="remove", project_dir=str(tmp_path),
                       rule_id=rid)["status"] == "removed"
    assert _rules_tool(server_module, action="list", project_dir=str(tmp_path))["rules"] == []


def test_router_rules_promote_from_checkpoint(server_module, tmp_path, monkeypatch):
    monkeypatch.setattr(server_module, "_checkpoints_dir", tmp_path / "cps")
    _checkpoint(server_module, action="save", name="arch",
                summary="decisiones de arquitectura", decisions=["nunca usar Redux"])
    r = _rules_tool(server_module, action="promote", project_dir=str(tmp_path),
                    checkpoint="arch", from_client="code")
    assert r["status"] == "promoted"
    assert r["rule"]["text"] == "nunca usar Redux"
    assert r["rule"]["checkpoint"] == "arch", "la procedencia debe apuntar al checkpoint de origen"


def test_router_rules_sync_to_claudemd_flag(server_module, tmp_path):
    r = _rules_tool(server_module, action="add", project_dir=str(tmp_path),
                    text="regla sincronizada", sync_to_claudemd=True)
    assert "claudemd" in r
    md_content = (tmp_path / "CLAUDE.md").read_text(encoding="utf-8")
    assert "regla sincronizada" in md_content


def test_smart_read_injects_project_rules(server_module, tmp_path, monkeypatch):
    led = FileLedger(str(tmp_path / "rl.db"))
    monkeypatch.setattr(server_module, "ledger", led)
    R.add_rule(tmp_path, "nunca usar Redux", source_client="code")
    f = tmp_path / "chico.py"
    f.write_text("print('hola')", encoding="utf-8")

    r = _smart_read(server_module, file_path=str(f))
    assert "project_rules" in r
    assert "nunca usar Redux" in r["project_rules"]
    led.close()


def test_smart_read_omits_project_rules_when_none(server_module, tmp_path, monkeypatch):
    led = FileLedger(str(tmp_path / "rl2.db"))
    monkeypatch.setattr(server_module, "ledger", led)
    f = tmp_path / "chico2.py"
    f.write_text("print('hola')", encoding="utf-8")

    r = _smart_read(server_module, file_path=str(f))
    assert "project_rules" not in r, "sin reglas, el campo no debe existir (disciplina de tokens)"
    led.close()
