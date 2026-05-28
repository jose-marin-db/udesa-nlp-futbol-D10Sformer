"""Shared UX components for the D10Sformer Lab app."""

from __future__ import annotations

import streamlit as st


def inject_lab_css(path) -> None:
    st.markdown(f"<style>{path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def page_hero(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="lab-hero">
          <div class="lab-hero-kicker">D10Sformer Lab</div>
          <h1>{title}</h1>
          <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_intro(eyebrow: str, title: str, body: str, tip: str = "") -> None:
    tip_html = (
        f'<div class="lab-tip"><strong>Tip:</strong> {tip}</div>'
        if tip
        else ""
    )
    st.markdown(
        f"""
        <div class="lab-section">
          <div class="lab-eyebrow">{eyebrow}</div>
          <h2>{title}</h2>
          <p>{body}</p>
          {tip_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_strip(items: list[tuple[str, str]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        col.markdown(
            f"""
            <div class="lab-metric">
              <div class="lab-metric-val">{value}</div>
              <div class="lab-metric-lbl">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def step_pills(labels: list[str], active: int) -> None:
    cols = st.columns(len(labels))
    for i, (col, label) in enumerate(zip(cols, labels)):
        cls = "lab-pill active" if i == active else "lab-pill"
        col.markdown(f'<div class="{cls}">{label}</div>', unsafe_allow_html=True)
