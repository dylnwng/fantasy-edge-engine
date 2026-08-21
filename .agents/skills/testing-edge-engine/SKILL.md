---
name: testing-edge-engine
description: How to run and end-to-end test the Edge Engine Streamlit dashboard and the draft-night HTTP screen locally.
---

# Testing Edge Engine (dashboard + draft-night screen)

## Prereqs
- venv at `.venv` (created by `scripts/install.sh`), data ingested to `data/player_week.parquet`, models trained (see blueprint / CLAUDE.md).
- Draft board needs `data/draft/adp.csv` — if absent, copy `data/draft/adp.example.csv` to `adp.csv` (synthetic ADP, fine for testing).
- No credentials needed unless testing live ESPN features (`ESPN_*` vars, see `.env.example`).

## Starting the apps
- Dashboard: `cd <repo> && .venv/bin/streamlit run app.py --server.headless true --server.port 8501`.
- Draft screen: `.venv/bin/python -m edge_engine.draft --serve --port 8765` → http://127.0.0.1:8765.
- Gotcha: when backgrounding with `nohup ... &`, `source .venv/bin/activate` may not carry into the subshell — invoke `.venv/bin/python` / `.venv/bin/streamlit` directly.
- Dashboard takes ~15s to first render (model scoring on load). Streamlit hot-reloads on file save, so temporary app.py edits show up on browser refresh without a restart.

## Draft screen testing tips
- State endpoint for assertions: `curl -s http://127.0.0.1:8765/state` — fields: `picks_made`, `best_available`, `my_roster`, `unfilled`, `seconds_since_poll`, `stale`.
- Manual picks: type a player name in the fixed bottom input + Enter; prefix with `*` to mark as your pick. Matching is case-insensitive against the full board.
- Staleness indicator turns red past 10s with no pick/poll; a manual pick resets it only when no `--espn` poll source is active (by design).
- Known behavior: re-entering an already-drafted name is accepted again (supports two real players sharing a name); it increments `picks_made` without erroring.
- When typing multiple picks via automation, pause ~1s between submits — the input clears asynchronously after the POST resolves, so back-to-back typing races with the clear and mangles names.

## Dashboard testing tips
- With the default 2026 manual roster and 2025 data, My Team/Trade/Draft tabs show a "last season's numbers" heads-up — expected, not a bug.
- To exercise the week-1-only branch (Through-week slider skipped), temporarily set `latest_week = 1` in `load_usage()` in app.py, refresh, verify "Only week 1 is in the books so far.", then revert.

## Devin Secrets Needed
- None for manual-source testing. `ESPN_*` cookies only for live ESPN roster/draft sync.
