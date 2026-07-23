"""
Suite de tests de los módulos locales del router (sin llamadas de red).

Cubre:
  - router/ranker.py    → chunk_text, rank_chunks, build_outline
  - router/sanitizer.py → strip_html, clean_text
  - router/ledger.py    → record/get, diff, purge, check_files
  - router/inbox.py     → send/check/complete/history

Todas las DBs usan tmp_path de pytest. Cero red, cero side-effects fuera del tmp.
"""

import time

import pytest

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
