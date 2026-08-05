"""P1: Streamlit dashboard for the free-agent rankings (requirements 4/5),
with position filters and an on-demand refresh — the interactive
alternative to `python -m edge_engine.ranking.roster_fit`'s CLI table.

Usage:
    streamlit run app.py
"""

from __future__ import annotations

import html as html_lib
from datetime import date

import pandas as pd
import streamlit as st

import theme
from edge_engine.model.config import load_model_config
from edge_engine.paths import PLAYER_WEEK_PATH
from edge_engine.ranking.output import build_free_agent_rankings
from edge_engine.ranking.roster_fit import apply_roster_fit
from edge_engine.ranking.usage_trend import get_usage_trend
from edge_engine.roster.interface import get_default_source

st.set_page_config(page_title="Edge Engine — Free Agent Rankings", layout="wide")


def _html(fragment: str) -> None:
    """st.markdown() still treats 4+ leading spaces per line as a code
    block even with unsafe_allow_html=True, which swallows raw HTML as
    literal text instead of rendering it. Flatten indentation first."""
    flat = "\n".join(line.strip() for line in fragment.strip().splitlines())
    st.markdown(flat, unsafe_allow_html=True)


_html(theme.inject_css())


@st.cache_data(show_spinner="Scoring free agents against your roster...")
def load_rankings():
    config = load_model_config()
    source = get_default_source()
    league_config = source.get_league_config()
    rostered = source.get_rostered_players()
    meta = source.get_roster_meta()

    candidates, unresolved = build_free_agent_rankings(config)
    bye_weeks = {p.player_id: p.bye_week for p in source.get_free_agents() if p.player_id}
    results = apply_roster_fit(candidates, league_config, rostered, bye_weeks)
    player_week = pd.read_parquet(PLAYER_WEEK_PATH)

    return results, unresolved, rostered, league_config, meta, player_week, config.trailing_window


def _esc(text: str) -> str:
    return html_lib.escape(str(text))


def render_ranking_table(filtered, player_week, trailing_window) -> str:
    rows_html = []
    for i, r in enumerate(filtered, start=1):
        c = r.candidate
        # Same (season, week) the explanation text was generated from
        # upstream (RankedFreeAgent carries it), so the sparkline can
        # never silently disagree with the sentence next to it.
        trend = get_usage_trend(player_week, c.player_id, c.season, c.week, trailing_window)
        spark = theme.sparkline_svg(trend.values) if trend.values else ""

        mult_class = "mult-up" if r.scarcity_multiplier > 1 else ("mult-down" if r.scarcity_multiplier < 1 else "")
        tier_class = {"High": "edge-tier-high", "Medium": "edge-tier-medium"}.get(c.confidence_tier, "edge-tier-low")
        warn_html = f'<span class="edge-warn-note">{_esc(r.bye_week_collision)}</span>' if r.bye_week_collision else ""

        rows_html.append(f"""
        <tr data-rank="{i}">
          <td class="edge-rank">{i}</td>
          <td>
            <div class="edge-player-name">{_esc(c.name)}</div>
            <div class="edge-player-meta"><span class="edge-pos-pill">{_esc(c.position)}</span>{_esc(c.team)}</div>
          </td>
          <td><div class="edge-trend-cell">{spark}<span class="edge-explanation">{_esc(c.explanation)}</span></div></td>
          <td>
            <div class="edge-score-trail">
              {c.predicted_score:.1f}
              <span class="op">×</span><span class="{mult_class}">{r.scarcity_multiplier:.2f}</span>
              <span class="op">=</span>
              <span class="final">{r.roster_fit_score:.1f}</span>
            </div>
          </td>
          <td><span class="edge-tier {tier_class}">{_esc(c.confidence_tier)}</span></td>
          <td class="edge-explanation">{warn_html or "&mdash;"}</td>
        </tr>
        """)

    return f"""
    <div class="edge-table-scroll">
      <table class="edge-table">
        <thead>
          <tr>
            <th>#</th><th>Player</th><th>Usage trend</th><th>Opportunity → Fit</th><th>Tier</th><th>Flags</th>
          </tr>
        </thead>
        <tbody>{''.join(rows_html)}</tbody>
      </table>
    </div>
    """


st.title("Edge Engine")
st.caption("Usage moves before points do — free agent opportunity rankings for your league.")

col_refresh, _ = st.columns([1, 5])
if col_refresh.button("↻ Refresh", help="Re-run the model against the current roster_state/ files"):
    load_rankings.clear()

results, unresolved, rostered, league_config, meta, player_week, trailing_window = load_rankings()

staleness = (date.today() - meta.as_of_date).days
age_label = "Current" if staleness <= 0 else f"{staleness}d old"
status_cols = st.columns([2, 2, 2, 2])
status_cols[0].metric("Roster data as of", f"Week {meta.as_of_week}")
status_cols[1].metric("Scoring", league_config.scoring.ppr_type.replace("_", " ").upper())
status_cols[2].metric(f"{league_config.waivers.system} remaining", f"${meta.remaining_faab:.0f}")
status_cols[3].metric("Data age", age_label, delta=None if staleness <= 7 else "stale", delta_color="inverse")

if staleness > 7:
    st.warning(f"Roster data is {staleness} days old — update `data/roster_state/` before trusting this.")

st.divider()

flagged_count = sum(1 for r in results if r.candidate.confidence_tier in ("High", "Medium"))
collision_count = sum(1 for r in results if r.bye_week_collision)

tile_cols = st.columns(4)
tile_cols[0].metric("Free Agents Ranked", len(results))
tile_cols[1].metric("Medium+ Confidence", flagged_count)
tile_cols[2].metric("Bye Week Collisions", collision_count)
tile_cols[3].metric("Unresolved Names", len(unresolved))

st.divider()

positions = sorted({r.candidate.position for r in results})
selected_positions = st.multiselect("Filter by position", positions, default=positions)

filtered = [r for r in results if r.candidate.position in selected_positions]

if not filtered:
    _html('<p class="edge-empty-note">No candidates match this filter.</p>')
else:
    table_html = render_ranking_table(filtered, player_week, trailing_window)
    _html(table_html)

st.write("")
col_roster, col_unresolved = st.columns(2)

with col_roster:
    st.subheader("Your Roster")
    if rostered:
        rows = "".join(
            f"""<div class="edge-roster-row">
                  <span><span class="edge-pos-pill">{_esc(p.position)}</span>{_esc(p.name)} · {_esc(p.team)}</span>
                  <span class="bye">bye {p.bye_week if p.bye_week else '?'}</span>
                </div>"""
            for p in rostered
        )
        _html(f'<div class="edge-panel">{rows}</div>')
    else:
        _html('<p class="edge-empty-note">No rostered players on file.</p>')

with col_unresolved:
    st.subheader("Needs Attention")
    if unresolved:
        rows = "".join(
            f"""<div class="edge-unresolved-row">
                  <span class="name">{_esc(p.name)}</span> ({_esc(p.position)}, {_esc(p.team)})
                  <span class="note">{_esc(p.match_note)}</span>
                </div>"""
            for p in unresolved
        )
        _html(f'<div class="edge-panel edge-panel-alert">{rows}</div>')
    else:
        _html('<p class="edge-empty-note">Everything on the roster and free-agent files resolved cleanly.</p>')

st.divider()
st.caption(
    "Opportunity score: XGBoost mean-regression on trailing usage trends, trained on 2018–2023, "
    "validated on the held-out 2024 season — beats a rolling-average baseline by ~10% MAE with a "
    "72% single-week hit rate (82% over the following 3 weeks). Fit score reflects position scarcity "
    "against your league's starting lineup only; it is not a FAAB bid recommendation. Bye-week and "
    "injury-context flags are surfaced for your judgment, not resolved automatically."
)
