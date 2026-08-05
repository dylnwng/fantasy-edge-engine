"""P1: Streamlit dashboard for the free-agent rankings (requirements 4/5),
with position filters and an on-demand refresh — the interactive
alternative to `python -m edge_engine.ranking.roster_fit`'s CLI table.

Usage:
    streamlit run app.py
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from edge_engine.model.config import load_model_config
from edge_engine.ranking.output import build_free_agent_rankings
from edge_engine.ranking.roster_fit import apply_roster_fit
from edge_engine.roster.interface import get_default_source

st.set_page_config(page_title="Edge Engine — Free Agent Rankings", layout="wide")


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

    return results, unresolved, rostered, league_config, meta


st.title("Edge Engine")
st.caption("Usage moves before points do — free agent opportunity rankings for your league.")

col_refresh, _ = st.columns([1, 5])
if col_refresh.button("↻ Refresh", help="Re-run the model against the current roster_state/ files"):
    load_rankings.clear()

results, unresolved, rostered, league_config, meta = load_rankings()

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
    st.info("No candidates match this filter.")
else:
    table = pd.DataFrame(
        [
            {
                "Rank": i + 1,
                "Player": r.candidate.name,
                "Pos": r.candidate.position,
                "Team": r.candidate.team,
                "Opportunity Score": round(r.candidate.predicted_score, 1),
                "Fit ×": round(r.scarcity_multiplier, 2),
                "Final Score": round(r.roster_fit_score, 1),
                "Confidence": r.candidate.confidence_tier,
                "Explanation": r.final_explanation,
            }
            for i, r in enumerate(filtered)
        ]
    )
    st.dataframe(
        table,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Opportunity Score": st.column_config.NumberColumn(format="%.1f"),
            "Fit ×": st.column_config.NumberColumn(format="%.2f"),
            "Final Score": st.column_config.NumberColumn(format="%.1f"),
        },
    )

col_roster, col_unresolved = st.columns(2)

with col_roster:
    st.subheader("Your Roster")
    if rostered:
        st.dataframe(
            pd.DataFrame(
                [{"Player": p.name, "Pos": p.position, "Team": p.team, "Bye": p.bye_week} for p in rostered]
            ),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.caption("No rostered players on file.")

with col_unresolved:
    st.subheader("Needs Attention")
    if unresolved:
        for p in unresolved:
            st.warning(f"**{p.name}** ({p.position}, {p.team}) — {p.match_note}")
    else:
        st.caption("Everything on the roster and free-agent files resolved cleanly.")

st.divider()
st.caption(
    "Opportunity score: XGBoost mean-regression on trailing usage trends, trained on 2018–2023, "
    "validated on the held-out 2024 season — beats a rolling-average baseline by ~10% MAE with a "
    "72% single-week hit rate (82% over the following 3 weeks). Fit score reflects position scarcity "
    "against your league's starting lineup only; it is not a FAAB bid recommendation. Bye-week and "
    "injury-context flags are surfaced for your judgment, not resolved automatically."
)
