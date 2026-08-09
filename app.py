"""Streamlit dashboard — the front end for the whole tool.

    streamlit run app.py

**Copy rule for this file:** everything a user reads must sound like a
fantasy football site, not a stats package. No MAE, no z-scores, no
"divergence", no "baseline", no model names. Real fantasy jargon (ADP,
FAAB, waiver wire, bye, flier, tier, buy low, sell high, workhorse,
snap share, target share) is fine and preferred — that's the language
managers already think in.

The underlying maths is unchanged. Only the words are different, plus a
"How does this work?" expander for anyone who wants the honest detail.
Note in particular that "buy low / sell high" IS the usage-vs-production
comparison — the fantasy concept and the statistic are the same thing,
so the tool should use the name people already know.
"""

from __future__ import annotations

import html as html_lib
from datetime import date

import pandas as pd
import streamlit as st

import theme
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
    """st.markdown() treats 4+ leading spaces as a code block even with
    unsafe_allow_html=True, which renders raw HTML as literal text."""
    flat = "\n".join(line.strip() for line in fragment.strip().splitlines())
    st.markdown(flat, unsafe_allow_html=True)


_html(theme.inject_css())


def _esc(text: str) -> str:
    return html_lib.escape(str(text))


@st.cache_data(show_spinner="Sizing up the waiver wire...")
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


@st.cache_data(show_spinner="Pulling up the season so far...")
def load_usage():
    player_week = pd.read_parquet(PLAYER_WEEK_PATH)
    league_config = get_default_source().get_league_config()

    season = league_config.season
    note = None
    if player_week[player_week["season"] == season].empty:
        season = int(player_week["season"].max())
        note = (
            f"The {league_config.season} season hasn't kicked off yet, so this is showing "
            f"{season} — last season's numbers."
        )

    available = player_week[player_week["season"] == season]
    points = compute_points_for_seasons([season], league_config.scoring)
    merged = available.merge(points, on=["season", "week", "player_id"], how="inner")
    latest_week = int(available["week"].max())
    return merged, season, latest_week, note


def _heads_up(notes: list[str]) -> None:
    """What the tool couldn't check, in plain language, ABOVE the results
    so nobody reads a partial answer thinking it's complete."""
    if not notes:
        return
    items = "".join(f"<li>{_esc(n)}</li>" for n in notes)
    _html(
        f'<div class="edge-panel edge-panel-alert"><strong>Heads up</strong>'
        f'<ul style="margin:6px 0 0 18px">{items}</ul></div>'
    )


# Raw gap strings are written for a developer; translate the ones users
# will actually hit into something a manager understands.
def _friendly(note: str) -> str:
    if "pool too small" in note:
        return "A few players at rare positions don't have enough comparable players to judge — they're skipped."
    if "did not resolve" in note or "no usage data" in note.lower():
        return note.replace("usage data", "stats").replace("divergence", "buy-low/sell-high read")
    if "league-wide distribution" in note:
        return ("Depth is judged against everyone at that position in the NFL, not against what "
                "your specific leaguemates are holding.")
    if "does not score" in note:
        return "Kickers and defenses aren't ranked — there's no useful workload data for them."
    if "no scores for this week at all" in note:
        return "It's too early in the season to have enough games to judge anyone yet."
    return note


def _fit_label(multiplier: float, have_roster: bool) -> str:
    """Roster fit as words, not a multiplier. Nobody thinks in ×3.33.

    With an empty roster every position scores as a gaping need, so the
    column would read "Big need" on all 375 rows — technically true and
    completely useless. Say there's no roster to compare against instead
    of rendering a column of meaningless green badges."""
    if not have_roster:
        return '<span class="edge-explanation">no roster yet</span>'
    if multiplier >= 2.0:
        return '<span class="edge-tier edge-tier-high">Big need</span>'
    if multiplier > 1.0:
        return '<span class="edge-tier edge-tier-medium">Helps you</span>'
    if multiplier == 1.0:
        return '<span class="edge-tier edge-tier-low">Neutral</span>'
    return '<span class="edge-tier edge-tier-low">You\'re deep here</span>'


_TIER_BLURB = {
    "High": "Worth a real FAAB bid",
    "Medium": "Worth a look",
    "Low": "$1 flier at most",
}


def render_ranking_table(filtered, player_week, trailing_window, have_roster: bool) -> str:
    rows_html = []
    for i, r in enumerate(filtered, start=1):
        c = r.candidate
        trend = get_usage_trend(player_week, c.player_id, c.season, c.week, trailing_window)
        spark = theme.sparkline_svg(trend.values) if trend.values else ""

        tier_class = {"High": "edge-tier-high", "Medium": "edge-tier-medium"}.get(
            c.confidence_tier, "edge-tier-low"
        )
        warn = f'<span class="edge-warn-note">{_esc(r.bye_week_collision)}</span>' if r.bye_week_collision else ""

        rows_html.append(f"""
        <tr data-rank="{i}">
          <td class="edge-rank">{i}</td>
          <td>
            <div class="edge-player-name">{_esc(c.name)}</div>
            <div class="edge-player-meta"><span class="edge-pos-pill">{_esc(c.position)}</span>{_esc(c.team)}</div>
          </td>
          <td><div class="edge-trend-cell">{spark}<span class="edge-explanation">{_esc(c.explanation)}</span></div></td>
          <td><div class="edge-score-trail"><span class="final">{c.predicted_score:.1f}</span></div></td>
          <td>{_fit_label(r.scarcity_multiplier, have_roster)}</td>
          <td><span class="edge-tier {tier_class}" title="{_TIER_BLURB.get(c.confidence_tier, '')}">
              {_esc(_TIER_BLURB.get(c.confidence_tier, c.confidence_tier))}</span></td>
          <td class="edge-explanation">{warn or "&mdash;"}</td>
        </tr>
        """)

    return f"""
    <div class="edge-table-scroll">
      <table class="edge-table">
        <thead>
          <tr>
            <th>#</th><th>Player</th><th>What changed</th>
            <th>Proj. pts</th><th>Fills a hole?</th><th>How much to spend</th><th>Watch out</th>
          </tr>
        </thead>
        <tbody>{''.join(rows_html)}</tbody>
      </table>
    </div>
    """


def buy_sell_bar(value: float | None) -> str:
    """Left = he's outscoring his workload (sell high). Right = his
    workload says more is coming (buy low). A bar, not a number, because
    the direction is the point."""
    if value is None:
        return "&mdash;"
    pct = min(abs(value) / 3.0, 1.0) * 50
    if value < 0:
        bar = f'<div style="width:{pct}%;background:#e0a13a;height:10px;margin-left:{50 - pct}%"></div>'
    else:
        bar = f'<div style="width:{pct}%;background:#2a8f4e;height:10px;margin-left:50%"></div>'
    return (
        f'<div style="position:relative;background:#1a1a1a;height:10px;width:150px">'
        f'<div style="position:absolute;left:50%;width:1px;height:10px;background:#666"></div>{bar}</div>'
    )


_VERDICT = {
    "undervalued": ("BUY LOW", "#2a8f4e"),
    "overvalued": ("SELL HIGH", "#e0a13a"),
}

# ---------------------------------------------------------------- header

st.title("Edge Engine")
st.caption("The waiver wire, your roster, trades and your draft — all read off who's actually getting the ball.")

col_refresh, _ = st.columns([1, 5])
if col_refresh.button("↻ Refresh", help="Pull the latest numbers"):
    load_rankings.clear()
    load_usage.clear()

results, unresolved, rostered, league_config, meta, player_week, trailing_window = load_rankings()

staleness = (date.today() - meta.as_of_date).days
status_cols = st.columns([2, 2, 2, 2])
status_cols[0].metric("Roster as of", f"Week {meta.as_of_week}")
status_cols[1].metric("League scoring", league_config.scoring.ppr_type.replace("_", " ").upper())
status_cols[2].metric(f"{league_config.waivers.system} left", f"${meta.remaining_faab:.0f}")
status_cols[3].metric("Info is", "Fresh" if staleness <= 7 else f"{staleness} days old")

if staleness > 7:
    st.warning(
        f"Your roster info is {staleness} days old. Update it before trusting anything here."
    )

with st.expander("How does this work? (30 seconds)"):
    st.markdown(
        """
**The idea:** most rankings sort by who *scored* last week. But scoring is the
last thing to change. A guy's snap share, targets and red-zone looks all move
**first** — usually a week or two before the box score catches up. That gap is
the whole point: grab him while he's still cheap.

**What the numbers mean**

- **Proj. pts** — what he's expected to score next week in your league's scoring.
- **Fills a hole?** — whether you actually *need* that position, based on your
  starting lineup. A great TE means less if you already have a great TE.
- **How much to spend** — a rough guide, not a bid amount. "Worth a real FAAB
  bid" is the short, high-confidence list; "$1 flier" is a dart throw.
- **Buy low / Sell high** — is he getting more work than his stat line shows
  (buy), or scoring more than his workload can keep up (sell)?

**How much to trust it:** it's meaningfully better than going off recent points,
and about two out of three players it highlights do beat their recent average.
It is *not* a crystal ball, and it won't tell you who to start over a stud.
Bye weeks, injuries and suspensions are shown as notes — they never quietly
change a player's score.
        """
    )

tab_waivers, tab_roster, tab_trade, tab_draft = st.tabs(
    ["Waiver Wire", "My Team", "Trade Check", "Draft Board"]
)

# ---------------------------------------------------------------- waivers

with tab_waivers:
    st.subheader("Who should I pick up?")

    worth_it = sum(1 for r in results if r.candidate.confidence_tier in ("High", "Medium"))
    collisions = sum(1 for r in results if r.bye_week_collision)

    tiles = st.columns(4)
    tiles[0].metric("Players available", len(results))
    tiles[1].metric("Actually worth a bid", worth_it)
    tiles[2].metric("Bye week clashes", collisions)
    tiles[3].metric("Names I couldn't find", len(unresolved))

    positions = sorted({r.candidate.position for r in results})
    picked = st.multiselect("Positions", positions, default=positions)
    only_good = st.checkbox(
        "Hide the dart throws", value=True,
        help="Most available players are $1 fliers. This hides them so the names actually "
             "worth bidding on aren't buried.",
    )

    filtered = [r for r in results if r.candidate.position in picked]
    if only_good:
        filtered = [r for r in filtered if r.candidate.confidence_tier in ("High", "Medium")]

    if not rostered:
        st.info(
            "You don't have a roster loaded yet, so these aren't filtered for what *you* need — "
            "it's just who's trending up league-wide. Normal before your draft."
        )

    if not filtered:
        _html('<p class="edge-empty-note">Nobody matches those filters right now.</p>')
    else:
        _html(render_ranking_table(filtered, player_week, trailing_window, bool(rostered)))

    if unresolved:
        st.subheader("Couldn't look these up")
        st.caption("Usually a spelling or team mismatch. They're skipped, not judged.")
        rows = "".join(
            f'<div class="edge-unresolved-row"><span class="name">{_esc(p.name)}</span> '
            f'({_esc(p.position)}, {_esc(p.team)})</div>'
            for p in unresolved
        )
        _html(f'<div class="edge-panel edge-panel-alert">{rows}</div>')

# ---------------------------------------------------------------- my team

with tab_roster:
    st.subheader("What's wrong with my team?")

    if not rostered:
        st.info("No players on your roster yet — nothing to look at. Normal before your draft.")
    else:
        merged, usage_season, latest_week, season_note = load_usage()
        c1, c2 = st.columns(2)
        window = c1.slider("Look back how many games?", 2, 6, 3)
        week = c2.slider("Through week", 1, latest_week, latest_week)

        config = load_model_config()
        report = build_report(
            rostered=rostered, player_week=merged, points=merged["points"],
            scored=score_as_of_week(usage_season, week + 1, config),
            lineup_slots=league_config.lineup_slots,
            injuries=load_injury_reports([usage_season]),
            season=usage_season, week=week, window=window,
        )
        _heads_up(([season_note] if season_note else []) + [_friendly(g) for g in report.data_gaps])

        st.markdown("**Where your roster breaks** — the week you'll be scrambling for a starter")
        st.dataframe(
            pd.DataFrame([
                {
                    "Position": s.position,
                    "Startable guys": s.startable_now,
                    "You must start": s.required,
                    "Spare right now": s.surplus_now,
                    "Worst week": s.worst_week if s.worst_week is not None else "—",
                    "Spare that week": s.worst_week_surplus if s.worst_week_surplus is not None else "—",
                }
                for s in sorted(report.scarcity, key=lambda x: x.position)
            ]),
            hide_index=True, use_container_width=True,
        )

        st.markdown("**Buy low / sell high**")
        st.caption(
            "Green means his workload says more points are coming. Amber means he's been "
            "scoring above what his workload can hold up."
        )
        movers = [p for p in report.players if p.signal in ("undervalued", "overvalued")]
        quiet = [p for p in report.players if p.signal == "neutral"]

        if not movers:
            _html('<p class="edge-empty-note">Nobody on your roster is out of line with '
                  'his workload right now. That\'s a real answer, not a blank screen.</p>')
        for p in movers:
            label, colour = _VERDICT[p.signal]
            cols = st.columns([2, 2, 2, 4])
            cols[0].markdown(f"**{_esc(p.name)}** · {_esc(p.position)}")
            cols[1].markdown(
                f'<span style="color:{colour};font-weight:700">{label}</span>', unsafe_allow_html=True
            )
            cols[2].markdown(buy_sell_bar(p.divergence), unsafe_allow_html=True)
            cols[3].caption(" ".join(p.context))

        if quiet:
            with st.expander(f"{len(quiet)} guys look about right"):
                for p in quiet:
                    st.caption(f"{p.name} ({p.position})")

        if report.bye_collisions:
            st.markdown("**Bye week clashes**")
            for c in report.bye_collisions:
                st.caption(f"Week {c.week}: your {c.position}s — {', '.join(c.players)}")
        for note in report.exposure_notes:
            st.caption(f"• {note}")

# ---------------------------------------------------------------- trade

with tab_trade:
    st.subheader("Should I make this trade?")
    st.caption(
        "This won't tell you yes or no — anyone who does is guessing. It shows you what each "
        "guy is actually doing on the field so you can decide."
    )

    merged, usage_season, latest_week, season_note = load_usage()
    if season_note:
        _heads_up([season_note])

    names = sorted({n for n in merged["player_name"].dropna().unique()})
    c1, c2 = st.columns(2)
    giving = c1.multiselect("You give up", names, key="trade_out")
    getting = c2.multiselect("You get back", names, key="trade_in")

    if not giving and not getting:
        _html('<p class="edge-empty-note">Pick at least one player on either side.</p>')
    else:
        comparison = compare_trade(
            giving, getting, merged, merged["points"],
            season=usage_season, week=latest_week, window=3,
        )
        _heads_up([_friendly(g) for g in comparison.data_gaps])

        for label, side in (("You give up", comparison.outgoing), ("You get back", comparison.incoming)):
            st.markdown(f"**{label}**")
            if not side.players:
                st.caption("Nobody")
                continue
            for p in side.players:
                cols = st.columns([2, 1, 2, 4])
                cols[0].markdown(f"**{_esc(p.name)}** · {_esc(p.position)}")
                cols[1].markdown(f"{p.ppg_to_date:.1f} pts/gm" if p.ppg_to_date is not None else "—")
                cols[2].markdown(buy_sell_bar(p.divergence), unsafe_allow_html=True)
                cols[3].caption(" ".join(p.context))
            st.caption(f"Combined so far this season: {side.total_ppg_to_date:.1f} pts/gm")

        st.caption(
            "These are points they've **already** scored, not a forecast of the rest of the "
            "season. Nobody can forecast that reliably, so this tool doesn't pretend to."
        )

# ---------------------------------------------------------------- draft

with tab_draft:
    st.subheader("Draft board")

    try:
        from edge_engine.draft.board import build_board, late_season_divergence
        from edge_engine.draft.market import ManualMarketPriceSource

        prices = ManualMarketPriceSource().get_market_prices()
        merged, usage_season, _lw, season_note = load_usage()
        late, late_notes = late_season_divergence(merged, merged["points"], season=usage_season)
        board, board_notes = build_board(prices, divergence_by_id=late)

        _heads_up(([season_note] if season_note else [])
                  + [_friendly(n) for n in late_notes + board_notes])
        st.caption(
            "Sorted by ADP — where players are actually going in drafts. The notes flag guys "
            "whose workload down the stretch last year doesn't match where they're being drafted."
        )

        board_positions = sorted({b.position for b in board})
        picked = st.multiselect("Positions", board_positions, default=board_positions, key="draft_pos")
        limit = st.slider("How many to show", 10, 200, 50, key="draft_limit")

        shown = [b for b in board if b.position in picked][:limit]
        st.dataframe(
            pd.DataFrame([
                {
                    "ADP": b.adp, "Player": b.name, "Rank": f"{b.position}{b.position_rank}",
                    "Team": b.team, "Tier": f"Tier {b.tier}",
                    "Note": " · ".join(b.tags) if b.tags else ("no history to go on" if b.low_confidence else ""),
                }
                for b in shown
            ]),
            hide_index=True, use_container_width=True,
        )
    except RuntimeError:
        st.info(
            "The draft board needs a list of where players are being drafted (ADP). "
            "Grab an export from FantasyPros, Sleeper or Underdog, save it as "
            "`data/draft/adp.csv`, and this page will fill in.\n\n"
            "To try it with sample data first: `cp data/draft/adp.example.csv data/draft/adp.csv`"
        )

st.divider()
st.caption(
    "Rankings are built from what players are actually doing on the field — snaps, targets, "
    "carries near the goal line — not from what they scored last week. Bye weeks, injuries and "
    "suspensions are shown as notes so you can weigh them yourself; they never quietly change "
    "a player's projection."
)
