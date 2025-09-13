"""Streamlit GUI for Gleichungs Generator."""

from __future__ import annotations

import json
import math
from io import BytesIO
from pathlib import Path

import streamlit as st
import sympy as sp
from PIL import Image, ImageDraw, ImageFont

from gleichungs_generator import (
    DEFAULT_CONFIG,
    Equation,
    Problem,
    beautify_equation,
    generate_equations,
    render_arbeitsblatt,
    render_loesungsblatt,
    validate,
    x,
)

MAX_ABS = 120

# ---------------------------------------------------------------------------
# Hilfsfunktionen für Checks und Darstellung
# ---------------------------------------------------------------------------


def lcm_leq_limit(eq: Equation, limit: int = MAX_ABS) -> tuple[bool, int]:
    """Prüft ob kgV aller Nenner ≤ limit ist."""
    expr = eq.sympy_eq.lhs - eq.sympy_eq.rhs
    expr_together = sp.together(expr)
    denominators: list[int] = []
    for term in sp.Add.make_args(expr_together):
        _, denom = sp.fraction(term)
        if denom != 1:
            denominators.append(int(denom))
    if not denominators:
        return True, 1
    lcm_val = denominators[0]
    for d in denominators[1:]:
        lcm_val = math.lcm(lcm_val, d)
    return lcm_val <= limit, lcm_val


def numbers_leq_limit(eq: Equation, limit: int = MAX_ABS) -> tuple[bool, int]:
    """Prüft ob alle Zahlen ≤ limit sind."""

    max_val = 0

    def _expr_ok(expr: sp.Expr) -> bool:
        nonlocal max_val
        for atom in expr.atoms(sp.Rational):
            fr = sp.Rational(atom)
            max_val = max(max_val, abs(fr.p), abs(fr.q))
            if abs(fr.p) > limit or abs(fr.q) > limit:
                return False
        return True

    if not _expr_ok(eq.sympy_eq.lhs) or not _expr_ok(eq.sympy_eq.rhs):
        return False, int(max_val)
    lhs_exp = sp.expand(eq.sympy_eq.lhs)
    rhs_exp = sp.expand(eq.sympy_eq.rhs)
    if not _expr_ok(lhs_exp) or not _expr_ok(rhs_exp):
        return False, int(max_val)
    expr = sp.expand(sp.together(eq.sympy_eq.lhs - eq.sympy_eq.rhs))
    if expr.has(x):
        poly = sp.Poly(expr, x)
        for c in poly.all_coeffs():
            c_val = abs(float(c))
            max_val = max(max_val, c_val)
            if c_val > limit:
                return False, int(max_val)
    return True, int(max_val)


def equation_to_unicode(eq: Equation) -> str:
    return (
        sp.pretty(eq.sympy_eq.lhs, use_unicode=True)
        + " = "
        + sp.pretty(eq.sympy_eq.rhs, use_unicode=True)
    )


def equation_to_png(eq: Equation) -> bytes:
    txt = equation_to_unicode(eq)
    font = ImageFont.load_default()
    dummy = Image.new("RGB", (1, 1))
    d1 = ImageDraw.Draw(dummy)
    bbox = d1.multiline_textbbox((0, 0), txt, font=font)
    width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    img = Image.new("RGB", (width + 10, height + 10), "white")
    draw = ImageDraw.Draw(img)
    draw.multiline_text((5, 5), txt, fill="black", font=font)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def problem_checks(prob: Problem) -> dict[str, tuple[bool, str]]:
    eq = prob.equation
    expr = sp.expand(eq.sympy_eq.lhs - eq.sympy_eq.rhs)
    linear = sp.degree(expr, x) == 1
    unique = validate(eq)
    lcm_ok, lcm_val = lcm_leq_limit(eq)
    numbers_ok, max_val = numbers_leq_limit(eq)
    checks: dict[str, tuple[bool, str]] = {
        "Linearität": (linear, "" if linear else "Grad≠1"),
        "Eindeutige Lösung": (unique, "" if unique else "Keine eindeutige Lösung"),
        "LCM≤120": (lcm_ok, "" if lcm_ok else f"kgV={lcm_val}"),
        "Zahlengrenze≤120": (numbers_ok, "" if numbers_ok else f"max={max_val}"),
    }
    return checks


# ---------------------------------------------------------------------------
# Session State Defaults
# ---------------------------------------------------------------------------

if "config" not in st.session_state:
    st.session_state["config"] = DEFAULT_CONFIG.copy()
if "problems" not in st.session_state:
    st.session_state["problems"] = []
if "preview_mode" not in st.session_state:
    st.session_state["preview_mode"] = "unicode"
if "logs" not in st.session_state:
    st.session_state["logs"] = []
if "seed_used" not in st.session_state:
    st.session_state["seed_used"] = None
if "page" not in st.session_state:
    st.session_state["page"] = "Start"

# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

pages = ["Start", "Einstellungen", "Vorschau", "Export", "Protokoll"]
st.sidebar.radio("Seite", pages, key="page")

page = st.session_state["page"]

# ---------------------------------------------------------------------------
# Startseite
# ---------------------------------------------------------------------------

if page == "Start":
    st.title("Gleichungs‑Generator")
    st.write(
        "Erzeuge Arbeits- und Lösungsblätter für lineare Gleichungen mit einem Klick."
    )
    if st.button("Neuen Satz generieren"):
        st.session_state["page"] = "Einstellungen"
        st.experimental_rerun()

# ---------------------------------------------------------------------------
# Einstellungen
# ---------------------------------------------------------------------------

if page == "Einstellungen":
    st.title("Einstellungen")
    cfg = st.session_state["config"].copy()

    with st.form("config_form"):
        cfg["seed"] = st.number_input("Seed", value=cfg.get("seed", 12345), step=1)
        counts = cfg.get("counts", {})
        cols = st.columns(5)
        for i, lvl in enumerate(["L1", "L2", "L3", "L4", "L5"]):
            counts[lvl] = cols[i].slider(lvl, 0, 50, counts.get(lvl, 0))
        cfg["counts"] = counts
        cfg["prefer_fraction"] = st.slider(
            "Bruchquote", 0.5, 0.95, cfg.get("prefer_fraction", 0.85)
        )
        cfg["visual_complexity"] = st.selectbox(
            "Darstellung",
            ["mixed", "clean"],
            index=["mixed", "clean"].index(cfg.get("visual_complexity", "mixed")),
        )
        st.subheader("Numerische Leitplanken (≤ 120)")
        cfg["strict_limits"] = st.checkbox(
            "Limit strikt erzwingen (alle Schritte)",
            value=cfg.get("strict_limits", True),
        )
        st.info("kgV-Grenze (≤120) und automatisches Resampling")
        cfg["max_resamples"] = st.number_input(
            "Max. Resamples pro Aufgabe", 1, 500, cfg.get("max_resamples", 200)
        )
        st.subheader("Difficulty-Aufträge (D-OPS)")
        dops_options = [f"D-OPS-{i}" for i in range(1, 26)]
        cfg["dops"] = st.multiselect(
            "D-OPS auswählen", dops_options, default=cfg.get("dops", [])
        )
        cfg["max_dops"] = st.selectbox(
            "Max. Kombinationen pro Aufgabe",
            [1, 2, 3],
            index=[1, 2, 3].index(cfg.get("max_dops", 3)),
        )
        st.caption("Priorität: {1,7,20} > {3,4,5,6,17,24} > Rest")
        st.subheader("Bruch-/Ausgabe-Optionen")
        st.caption("Lösungen: Unechter Bruch und gemischte Zahl (fest)")
        st.caption("Dezimaldarstellung deaktiviert")
        st.subheader("Dateien")
        cfg["output_dir"] = st.text_input(
            "Ausgabeverzeichnis", cfg.get("output_dir", ".")
        )
        cfg["arbeits_filename"] = st.text_input(
            "Arbeitsblatt-Dateiname", cfg.get("arbeits_filename", "arbeitsblatt.docx")
        )
        cfg["loesung_filename"] = st.text_input(
            "Lösungsblatt-Dateiname", cfg.get("loesung_filename", "loesungsblatt.docx")
        )
        submitted = st.form_submit_button("Aufgabenset generieren")

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Konfiguration als JSON speichern",
            json.dumps(cfg, indent=2).encode("utf-8"),
            file_name="config.json",
        )
    with col2:
        uploaded = st.file_uploader("Konfiguration laden", type="json")
        if uploaded:
            cfg = json.load(uploaded)
            st.session_state["config"] = cfg
            st.experimental_rerun()

    if submitted:
        total = sum(cfg["counts"].values())
        if total > 200:
            st.error("Maximale Aufgabenanzahl 200 überschritten")
        else:
            st.session_state["config"] = cfg
            st.session_state["seed_used"] = cfg["seed"]
            problems = generate_equations(cfg)
            st.session_state["problems"] = problems
            logs = []
            for i, p in enumerate(problems, 1):
                logs.append(
                    {
                        "id": i,
                        "equation": p.equation.text,
                        "excluded": [str(e) for e in p.equation.excluded],
                        "resamples": p.resamples,
                        "dops": ",".join(p.dops),
                    }
                )
            st.session_state["logs"] = logs
            if len(problems) < total:
                st.warning("Zielanzahl nicht erreicht. Nächster Versuch?")
            st.session_state["page"] = "Vorschau"
            st.experimental_rerun()

# ---------------------------------------------------------------------------
# Vorschau
# ---------------------------------------------------------------------------

if page == "Vorschau":
    st.title("Vorschau")
    problems: list[Problem] = st.session_state.get("problems", [])
    if not problems:
        st.info("Keine Aufgaben generiert.")
    else:
        st.radio(
            "Render-Modus",
            ["unicode", "png"],
            key="preview_mode",
            format_func=lambda x: "Unicode-Pretty"
            if x == "unicode"
            else "PNG-Schnappschuss",
        )
        total = len(problems)
        avg_res = sum(p.resamples for p in problems) / total if total else 0
        from collections import Counter

        dops_counter: Counter[str] = Counter()
        for p in problems:
            dops_counter.update(p.dops)
        st.info(
            f"Akzeptiert: {total} | Ø Resamples: {avg_res:.2f} | "
            f"D-OPS: {dict(dops_counter)}"
        )
        for i, prob in enumerate(problems, 1):
            st.subheader(f"Aufgabe {i}")
            if st.session_state["preview_mode"] == "unicode":
                st.text(equation_to_unicode(prob.equation))
            else:
                st.image(equation_to_png(prob.equation))
            checks = problem_checks(prob)
            parts = []
            for name, (ok, reason) in checks.items():
                title = reason
                icon = "✅" if ok else "❌"
                if not ok and prob.resamples:
                    title = (title + f"; resampled {prob.resamples}×").strip(";")
                span = f"<span title='{title}'>{icon}{name}</span>"
                parts.append(span)
            st.markdown(" ".join(parts), unsafe_allow_html=True)
            if prob.resamples:
                st.caption(f"Resamples: {prob.resamples}")
            if st.checkbox("Lösungsschritte anzeigen", key=f"steps_{i}"):
                for step in prob.steps:
                    equation = beautify_equation(
                        step.lhs,
                        step.rhs,
                        st.session_state["config"]["visual_complexity"],
                    )
                    st.write(f"- {step.description_de}: {equation}")

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

if page == "Export":
    st.title("Export")
    problems: list[Problem] = st.session_state.get("problems", [])
    cfg = st.session_state.get("config", {})
    if not problems:
        st.info("Keine Aufgaben zum Export.")
    else:
        out_dir = Path(cfg.get("output_dir", "."))
        arbeits = out_dir / cfg.get("arbeits_filename", "arbeitsblatt.docx")
        loesung = out_dir / cfg.get("loesung_filename", "loesungsblatt.docx")
        if st.button("Arbeitsblatt.docx erzeugen"):
            render_arbeitsblatt(problems, str(arbeits))
            st.success(f"Arbeitsblatt gespeichert: {arbeits}")
        if st.button("Lösungsblatt.docx erzeugen"):
            render_loesungsblatt(problems, str(loesung), cfg)
            st.success(f"Lösungsblatt gespeichert: {loesung}")
        st.write(f"Anzahl Aufgaben: {len(problems)}")

# ---------------------------------------------------------------------------
# Protokoll
# ---------------------------------------------------------------------------

if page == "Protokoll":
    st.title("Protokoll")
    logs = st.session_state.get("logs", [])
    if logs:
        st.table(logs)
    else:
        st.info("Keine Protokolle verfügbar.")
