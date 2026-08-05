"""Visual identity for the Streamlit dashboard: a coach's film-room
terminal, not a generic sports app -- dark, screen-lit, built for a
quick scan on waiver night. Same token system as the shareable HTML
dashboard artifact, so the daily tool and the portfolio piece read as
one product rather than two.

The display face (DIN Condensed) is read live from macOS's own font
file if present and embedded as a data URI -- it is NOT bundled into
this repo. It ships with Apple's OS under Apple's own font license,
which covers using it on a Mac that has it installed, not
redistributing the binary in a git repo. Everything falls back
gracefully to a system condensed sans if the file isn't there (e.g.
running this on Linux or a Mac without it) -- the design degrades, it
doesn't break.
"""

from __future__ import annotations

import base64
from pathlib import Path

GROUND = "#12181a"
SURFACE = "#1b2225"
LINE = "#2c3639"
LINE_SOFT = "#232b2d"
INK = "#e9ece8"
INK_DIM = "#8d9894"
INK_FAINT = "#5f6b68"
ACCENT = "#c9973e"
ACCENT_INK = "#1b140a"
TIER_HIGH = "#5b9279"
TIER_MEDIUM = "#7c98a8"
WARN = "#c1613d"

_DIN_CONDENSED_PATH = Path("/System/Library/Fonts/Supplemental/DIN Condensed Bold.ttf")
_FONT_FALLBACK_STACK = '"Arial Narrow", "Helvetica Neue", Arial, sans-serif'


def _font_face_css() -> str:
    if not _DIN_CONDENSED_PATH.exists():
        return ""
    encoded = base64.b64encode(_DIN_CONDENSED_PATH.read_bytes()).decode("ascii")
    return f"""
    @font-face {{
        font-family: "DIN Cond";
        src: url(data:font/ttf;base64,{encoded}) format("truetype");
        font-weight: 700;
        font-display: swap;
    }}
    """


def inject_css() -> str:
    """Returns a <style> block. Caller passes this to st.markdown with
    unsafe_allow_html=True, once, near the top of the page."""
    return f"""
    <style>
    {_font_face_css()}

    :root {{
        --font-display: "DIN Cond", {_FONT_FALLBACK_STACK};
        --font-mono: ui-monospace, "SF Mono", "Cascadia Code", Menlo, monospace;
    }}

    [data-testid="stApp"], [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
        background: {GROUND};
    }}
    [data-testid="stMainBlockContainer"] {{
        max-width: 1100px;
        padding-top: 2.5rem;
    }}
    [data-testid="stHeader"] {{
        background: transparent;
    }}

    [data-testid="stHeading"] h1 {{
        font-family: var(--font-display);
        font-weight: 700 !important;
        letter-spacing: 0.01em;
        color: {INK};
    }}
    [data-testid="stCaptionContainer"] {{
        color: {INK_FAINT} !important;
    }}

    [data-testid="stMetric"] {{
        background: {SURFACE};
        border: 1px solid {LINE};
        border-radius: 3px;
        padding: 12px 16px 10px;
    }}
    [data-testid="stMetricValue"] {{
        font-family: var(--font-display) !important;
        font-variant-numeric: tabular-nums;
        color: {INK} !important;
    }}
    [data-testid="stMetricLabel"] p {{
        font-size: 11px !important;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        color: {INK_FAINT} !important;
    }}

    [data-testid="stBaseButton-secondary"] {{
        background: {SURFACE};
        border: 1px solid {LINE};
        color: {INK_DIM};
        border-radius: 999px;
    }}
    [data-testid="stBaseButton-secondary"]:hover {{
        border-color: {ACCENT};
        color: {ACCENT};
    }}

    [data-testid="stMultiSelect"] span[data-baseweb="tag"] {{
        background-color: {ACCENT} !important;
        color: {ACCENT_INK} !important;
    }}

    hr {{
        border-color: {LINE} !important;
    }}

    .edge-table-scroll {{
        overflow-x: auto;
        border: 1px solid {LINE};
        border-radius: 3px;
    }}
    .edge-table {{
        width: 100%;
        min-width: 760px;
        border-collapse: collapse;
        font-size: 13.5px;
        color: {INK};
    }}
    .edge-table thead th {{
        text-align: left;
        font-size: 10.5px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: {INK_FAINT};
        font-weight: 600;
        padding: 10px 14px;
        border-bottom: 1px solid {LINE};
        background: {SURFACE};
    }}
    .edge-table tbody tr {{
        border-bottom: 1px solid {LINE_SOFT};
    }}
    .edge-table tbody tr:last-child {{ border-bottom: none; }}
    .edge-table tbody tr:hover {{ background: {SURFACE}; }}
    .edge-table td {{ padding: 12px 14px; vertical-align: top; }}

    .edge-rank {{
        font-family: var(--font-display);
        font-weight: 700;
        font-size: 19px;
        color: {INK_FAINT};
        font-variant-numeric: tabular-nums;
    }}
    .edge-table tr[data-rank="1"] .edge-rank {{ color: {ACCENT}; }}

    .edge-player-name {{ font-weight: 600; color: {INK}; }}
    .edge-player-meta {{ font-size: 11.5px; color: {INK_FAINT}; margin-top: 2px; }}
    .edge-pos-pill {{
        display: inline-block;
        font-size: 10.5px;
        font-weight: 700;
        letter-spacing: 0.03em;
        color: {INK_DIM};
        border: 1px solid {LINE};
        border-radius: 3px;
        padding: 1px 5px;
        margin-right: 6px;
    }}

    .edge-trend-cell {{ display: flex; align-items: center; gap: 8px; }}
    .edge-sparkline {{ flex-shrink: 0; }}

    .edge-score-trail {{
        font-family: var(--font-mono);
        font-variant-numeric: tabular-nums;
        font-size: 12.5px;
        color: {INK_DIM};
        white-space: nowrap;
    }}
    .edge-score-trail .final {{ font-size: 16px; font-weight: 600; color: {INK}; }}
    .edge-score-trail .op {{ color: {INK_FAINT}; padding: 0 3px; }}
    .edge-score-trail .mult-up {{ color: {TIER_HIGH}; }}
    .edge-score-trail .mult-down {{ color: {WARN}; }}

    .edge-tier {{
        display: inline-block;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        padding: 3px 9px;
        border-radius: 999px;
        white-space: nowrap;
    }}
    .edge-tier-high {{ background: rgba(91, 146, 121, 0.16); color: {TIER_HIGH}; }}
    .edge-tier-medium {{ background: rgba(124, 152, 168, 0.16); color: {TIER_MEDIUM}; }}
    .edge-tier-low {{ background: transparent; color: {INK_FAINT}; border: 1px solid {LINE}; }}

    .edge-explanation {{
        display: inline-block;
        color: {INK_DIM};
        font-size: 12.5px;
        line-height: 1.5;
        max-width: 30ch;
    }}
    .edge-warn-note {{ display: block; margin-top: 5px; color: {WARN}; }}
    .edge-warn-note::before {{ content: "⚑ "; }}

    .edge-panel {{
        border: 1px solid {LINE};
        border-radius: 3px;
        padding: 14px 16px;
    }}
    .edge-panel-alert {{ border-color: rgba(193, 97, 61, 0.35); }}
    .edge-roster-row {{
        display: flex;
        justify-content: space-between;
        font-size: 12.5px;
        padding: 5px 0;
        border-bottom: 1px solid {LINE_SOFT};
    }}
    .edge-roster-row:last-child {{ border-bottom: none; }}
    .edge-roster-row .bye {{ color: {INK_FAINT}; font-family: var(--font-mono); }}
    .edge-unresolved-row {{
        font-size: 12.5px;
        padding: 5px 0;
        border-bottom: 1px solid {LINE_SOFT};
        color: {INK_DIM};
    }}
    .edge-unresolved-row:last-child {{ border-bottom: none; }}
    .edge-unresolved-row .name {{ color: {WARN}; font-weight: 600; }}
    .edge-unresolved-row .note {{ display: block; color: {INK_FAINT}; margin-top: 1px; }}
    .edge-empty-note {{ font-size: 12.5px; color: {INK_FAINT}; font-style: italic; }}
    </style>
    """


def sparkline_svg(values: list[float], width: int = 60, height: int = 22) -> str:
    """A tiny inline trend line for the exact metric a row's explanation
    cites -- the product's whole thesis (usage moves before points do)
    made visible in the UI, not just stated in a sentence."""
    if len(values) < 2:
        return ""

    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    pad = 3
    n = len(values)
    xs = [pad + i * (width - 2 * pad) / (n - 1) for i in range(n)]
    ys = [height - pad - (v - lo) / span * (height - 2 * pad) for v in values]

    color = TIER_HIGH if values[-1] >= values[0] else INK_FAINT
    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))

    return (
        f'<svg class="edge-sparkline" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-hidden="true">'
        f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round"/>'
        f'<circle cx="{xs[-1]:.1f}" cy="{ys[-1]:.1f}" r="2.5" fill="{color}"/>'
        f"</svg>"
    )
