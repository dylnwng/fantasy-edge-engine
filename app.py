"""Streamlit dashboard: the interactive front end for every capability.

    streamlit run app.py

Four tabs, because each answers exactly one question and a screen that
answers three answers none of them:

  Waiver Wire  — who should I pick up
  My Roster    — what is actually wrong with my roster (Phase 3)
  Trade        — what does the usage data say about each side (Phase 4)
  Draft Board  — who's left, and where is the market using the wrong window

Rendering rules inherited from the project's central invariant (context is
surfaced, never baked into a score): context renders as text or badges and
never as a number, data gaps render above the results rather than as a
footnote, and "no signal" is a designed state rather than blank space.
"""

from __future__ import annotations

import html as html_lib
from datetime import date

import pandas as pd
import streamlit as st

import theme
from edge_engine.insights.divergence import NEUTRAL_BAND
from edge_engine.insights.report import build_report
from edge_engine.model.config import load_model_config
from edge_engine.model.injury_context import load_injury_reports
from edge_engine.model.predict import score_as_of_week
from edge_engine.model.scoring import compute_points_for_seasons
from edge_engine.paths import PLAYER_WEEK_PATH
from edge_engine.ranking.output import build_free_agent_rankings
from edge_engine.ranking.roster_fit import apply_roster_fit
from edge_engine.ranking.usage_trend import get_usage_trend
from edge_engine.roster.interface import get_default_source
from edge_engine.trade.compare import compare_trade

st.set_page_config(page_title="Edge Engine", layout="wide")


def _html(fragment: str) -> None:
    """st.markdown() still treats 4+ leading spaces per line as a code
    block even with unsafe_allow_html=True, which swallows raw HTML as
    literal text instead of rendering it. Flatten indentation first."""
    flat = "\n".join(line.strip() for line in fragment.strip().splitlines())
    st.markdown(flat, unsafe_allow_html=True)


_html(theme.inject_css())


def _esc(text: str) -> str:
    return html_lib.escape(str(text))


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


@st.cache_data(show_spinner="Loading usage history...")
def load_usage():
    """Usage + points for the most recent season that actually has data.

    The ESPN league year rolls over months before a snap is played, so in
    the offseason the league's own season has nothing in it. Falling back
    to the last season played (and saying so) beats an empty screen."""
    player_week = pd.read_parquet(PLAYER_WEEK_PATH)
    league_config = get_default_source().get_league_config()

    season = league_config.season
    note = None
    if player_week[player_week["season"] == season].empty:
        season = int(player_week["season"].max())
        note = (
            f"League year {league_config.season} has no usage data yet (season not started). "
            f"Showing {season}, the most recent season played."
        )

    available = player_week[player_week["season"] == season]
    points = compute_points_for_seasons([season], league_config.scoring)
    merged = available.merge(points, on=["season", "week", "player_id"], how="inner")
    latest_week = int(available["week"].max())
    return merged, season, latest_week, note


def _gaps_block(gaps: list[str]) -> None:
    """Data gaps render ABOVE results, visually distinct. A user reading a
    partial report must know it's partial before reading it."""
    if not gaps:
        return
    items = "".join(f"<li>{_esc(g)}</li>" for g in gaps)
    _html(
        f'<div class="edge-panel edge-panel-alert"><strong>Data gaps</strong>'
        f'<ul style="margin:6px 0 0 18px">{items}</ul></div>'
    )


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


def divergence_bar(divergence: float | None) -> str:
    """Diverging bar centred at zero. A point estimate never renders
    without its scale, and the reader should see who the outliers are
    without parsing a single digit."""
    if divergence is None:
        return "&mdash;"
    scale = 3.0
    pct = min(abs(divergence) / scale, 1.0) * 50
    if divergence < 0:
        bar = f'<div style="width:{pct}%;background:#e05252;height:10px;margin-left:{50 - pct}%"></div>'
    else:
        bar = f'<div style="width:{pct}%;background:#2a8f4e;height:10px;margin-left:50%"></div>'
    return (
        f'<div style="position:relative;background:#1a1a1a;height:10px;width:150px">'
        f'<div style="position:absolute;left:50%;width:1px;height:10px;background:#666"></div>{bar}</div>'
    )


# ---------------------------------------------------------------- header

st.title("Edge Engine")
st.caption("Usage moves before points do — waivers, roster diagnosis, trades and the draft board.")

col_refresh, _ = st.columns([1, 5])
if col_refresh.button("↻ Refresh", help="Re-run everything against the current data"):
    load_rankings.clear()
    load_usage.clear()

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

tab_waivers, tab_roster, tab_trade, tab_draft = st.tabs(
    ["Waiver Wire", "My Roster", "Trade", "Draft Board"]
)

# ---------------------------------------------------------------- waivers

with tab_waivers:
    st.subheader("Who should I pick up?")

    flagged_count = sum(1 for r in results if r.candidate.confidence_tier in ("High", "Medium"))
    collision_count = sum(1 for r in results if r.bye_week_collision)

    tile_cols = st.columns(4)
    tile_cols[0].metric("Free Agents Ranked", len(results))
    tile_cols[1].metric("Medium+ Confidence", flagged_count)
    tile_cols[2].metric("Bye Week Collisions", collision_count)
    tile_cols[3].metric("Unresolved Names", len(unresolved))

    positions = sorted({r.candidate.position for r in results})
    selected_positions = st.multiselect("Filter by position", positions, default=positions)
    actionable_only = st.checkbox(
        "Actionable tiers only (High / Medium)", value=True,
        help="A live run produced 293 candidates — 5 High, 38 Medium, 249 Low. The Low tier is "
             "the tool's own '$1 speculative claim' bucket.",
    )

    filtered = [r for r in results if r.candidate.position in selected_positions]
    if actionable_only:
        filtered = [r for r in filtered if r.candidate.confidence_tier in ("High", "Medium")]

    if not filtered:
        _html('<p class="edge-empty-note">No candidates match this filter.</p>')
    else:
        _html(render_ranking_table(filtered, player_week, trailing_window))

    if unresolved:
        st.subheader("Needs Attention")
        rows = "".join(
            f"""<div class="edge-unresolved-row">
                  <span class="name">{_esc(p.name)}</span> ({_esc(p.position)}, {_esc(p.team)})
                  <span class="note">{_esc(p.match_note)}</span>
                </div>"""
            for p in unresolved
        )
        _html(f'<div class="edge-panel edge-panel-alert">{rows}</div>')

# ---------------------------------------------------------------- roster

with tab_roster:
    st.subheader("What is actually wrong with my roster?")

    if not rostered:
        st.info(
            "The active roster source returned 0 rostered players — nothing to diagnose. "
            "In the offseason this is expected (the league hasn't drafted yet)."
        )
    else:
        merged, usage_season, latest_week, season_note = load_usage()
        window = st.slider("Usage window (games)", 2, 6, 3)
        week = st.slider("Through week", 1, latest_week, latest_week)

        config = load_model_config()
        report = build_report(
            rostered=rostered,
            player_week=merged,
            points=merged["points"],
            scored=score_as_of_week(usage_season, week + 1, config),
            lineup_slots=league_config.lineup_slots,
            injuries=load_injury_reports([usage_season]),
            season=usage_season,
            week=week,
            window=window,
        )
        _gaps_block(([season_note] if season_note else []) + report.data_gaps)

        st.markdown("**Bye-adjusted scarcity** — the week your depth actually breaks")
        st.dataframe(
            pd.DataFrame([
                {
                    "Pos": s.position, "Startable": s.startable_now, "Need": s.required,
                    "Now": s.surplus_now,
                    "Worst wk": s.worst_week if s.worst_week is not None else "—",
                    "Then": s.worst_week_surplus if s.worst_week_surplus is not None else "—",
                }
                for s in sorted(report.scarcity, key=lambda x: x.position)
            ]),
            hide_index=True, use_container_width=True,
        )

        st.markdown(f"**Usage vs production** — outside ±{NEUTRAL_BAND:.1f} SD is a real signal")
        signalled = [p for p in report.players if p.signal in ("undervalued", "overvalued")]
        neutral = [p for p in report.players if p.signal == "neutral"]

        if not signalled:
            _html('<p class="edge-empty-note">No player is distinguishable from his own usage '
                  'this week. That is an answer, not an empty result.</p>')
        for p in signalled:
            cols = st.columns([2, 2, 5])
            cols[0].markdown(f"**{_esc(p.name)}** · {_esc(p.position)}")
            cols[1].markdown(divergence_bar(p.divergence), unsafe_allow_html=True)
            cols[2].caption(" ".join(p.context))
        if neutral:
            with st.expander(f"{len(neutral)} player(s) show no distinguishable signal"):
                for p in neutral:
                    st.caption(f"{p.name} ({p.position}) — {p.divergence:+.2f} SD")

        if report.bye_collisions:
            st.markdown("**Bye collisions**")
            for c in report.bye_collisions:
                st.caption(f"Week {c.week}: {c.position} — {', '.join(c.players)}")
        for note in report.exposure_notes:
            st.caption(f"• {note}")

# ---------------------------------------------------------------- trade

with tab_trade:
    st.subheader("What does the usage data say about this trade?")
    st.caption(
        "No verdict, no fairness score, no winner. A rest-of-season model was built and "
        "rejected (it didn't beat a season-to-date average — see EVALUATION.md), so this "
        "shows observed usage and production rather than a projection it couldn't validate."
    )

    merged, usage_season, latest_week, season_note = load_usage()
    if season_note:
        _gaps_block([season_note])

    known_names = sorted({n for n in merged["player_name"].dropna().unique()})
    c1, c2 = st.columns(2)
    out_names = c1.multiselect("You give up", known_names, key="trade_out")
    in_names = c2.multiselect("You receive", known_names, key="trade_in")

    if not out_names and not in_names:
        _html('<p class="edge-empty-note">Pick at least one player on either side.</p>')
    else:
        comparison = compare_trade(
            out_names, in_names, merged, merged["points"],
            season=usage_season, week=latest_week, window=3,
        )
        _gaps_block(comparison.data_gaps)

        for label, side in (("You give up", comparison.outgoing), ("You receive", comparison.incoming)):
            st.markdown(f"**{label}**")
            if not side.players:
                st.caption("(nobody)")
                continue
            for p in side.players:
                cols = st.columns([2, 1, 2, 4])
                cols[0].markdown(f"**{_esc(p.name)}** · {_esc(p.position)}")
                cols[1].markdown(f"{p.ppg_to_date:.1f} ppg" if p.ppg_to_date is not None else "—")
                cols[2].markdown(divergence_bar(p.divergence), unsafe_allow_html=True)
                cols[3].caption(" ".join(p.context))
            st.caption(f"Combined observed: {side.total_ppg_to_date:.1f} ppg (not a projection)")

        for note in comparison.context:
            st.caption(f"• {note}")

# ---------------------------------------------------------------- draft

with tab_draft:
    st.subheader("Who's left, and where is the market using the wrong window?")

    try:
        from edge_engine.draft.board import build_board, late_season_divergence
        from edge_engine.draft.market import ManualMarketPriceSource

        prices = ManualMarketPriceSource().get_market_prices()
        merged, usage_season, _lw, season_note = load_usage()
        divergence, div_gaps = late_season_divergence(merged, merged["points"], season=usage_season)
        board, board_gaps = build_board(prices, divergence_by_id=divergence)

        _gaps_block(([season_note] if season_note else []) + div_gaps + board_gaps)
        st.caption(
            "Priced at ADP. This board makes no claim to beat the market — its value is knowing "
            "what's left in each tier. The tags flag where last season's late usage disagrees."
        )

        board_positions = sorted({b.position for b in board})
        picked = st.multiselect("Positions", board_positions, default=board_positions, key="draft_pos")
        limit = st.slider("Rows", 10, 200, 50, key="draft_limit")

        shown = [b for b in board if b.position in picked][:limit]
        st.dataframe(
            pd.DataFrame([
                {
                    "ADP": b.adp, "Player": b.name, "Rank": f"{b.position}{b.position_rank}",
                    "Tm": b.team, "Tier": f"T{b.tier}",
                    "Note": " · ".join(b.tags) if b.tags else ("priced at ADP only" if b.low_confidence else ""),
                }
                for b in shown
            ]),
            hide_index=True, use_container_width=True,
        )
    except RuntimeError as e:
        st.info(
            f"{e}\n\nThe draft board needs an ADP export. Copy the bundled example and replace "
            "its numbers with a real one:\n\n`cp data/draft/adp.example.csv data/draft/adp.csv`"
        )

st.divider()
st.caption(
    "Opportunity score: XGBoost mean-regression on trailing usage trends, trained 2018–2024 and "
    "validated on the held-out 2025 season — 5.17 MAE vs a 5.66 baseline, 67.6% single-week hit "
    "rate (80.3% over the following 3 weeks). Fit score reflects position scarcity against your "
    "league's starting lineup only; it is not a FAAB bid recommendation. Bye-week, injury and "
    "roster-status flags are surfaced for your judgment, never folded into a score."
)
