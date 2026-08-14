# Integrating Edge Engine into another team

This is for standing up your **own, independent** copy of Edge Engine against a
**different** ESPN league than the one it was originally built for — a friend's
team, not a hosted service. Each team runs its own checkout against its own
data; nothing here is shared between leagues.

If you just want to run this for yourself as the original author would,
[`SETUP.md`](SETUP.md) already covers that. This document exists because that
guide's step order has a real trap for a *second* league (see §2), and because
handing this to someone else raises questions — "is my data safe in this
repo?", "did I actually verify this works?" — that a same-author setup guide
doesn't need to answer.

~30–40 minutes, Python 3.12+, no cost, no API keys.

---

## 0. Will this actually fit your league?

Check this before investing the time — the tool is built to refuse outright
rather than quietly mis-score a league it doesn't support.

| Your league | Status |
|---|---|
| Platform is ESPN | ✅ Supported |
| Sleeper, Yahoo, or NFL.com | ❌ Not supported |
| Standard, half-PPR, or full-PPR scoring | ✅ Supported |
| Any other per-reception value (e.g. 1.5 PPR) | ❌ **Refuses to run** |
| Snake draft | ✅ Draft board works |
| Auction draft | ❌ Not supported |
| Keeper / dynasty | ❌ No keeper valuation |
| QB / RB / WR / TE | ✅ Modeled |
| Kickers & defenses | ❌ Not ranked — no usage data exists for them |

If your league's reception scoring isn't exactly 0, 0.5, or 1.0 points per
catch, the live ESPN connector raises this error verbatim:

> `League's reception scoring is N pts/catch, which isn't standard (0), half
> (0.5), or full (1.0) PPR. Refusing to guess.`

That's deliberate — guessing would silently mis-score every prediction.

---

## 1. Get your own copy

The repository is public, so cloning it needs no invitation or access
request.

```bash
git clone https://github.com/dylnwng/fantasy-edge-engine.git
cd fantasy-edge-engine
python3.12 -m venv .venv && source .venv/bin/activate
bash scripts/install.sh
```

> **Python version matters more than usual.** The project's floor is
> **3.12**, not 3.11 — confirmed directly: `numpy==2.5.1` and `xgboost` both
> refuse to install under 3.11 with `ResolutionImpossible`. Run
> `python3.12 --version` first if you're not sure which your machine
> defaults to.

Confirm the install worked:

```bash
python -m pytest -q
# expect: 440 passed, no network, no credentials
```

> **Your data doesn't belong in the original repo.** You'll shortly edit
> files with your own roster and, if you use the live connector, your own
> ESPN cookies. **Don't commit or push those back** — this clone is yours to
> run locally. If you want your own edits under version control, fork the
> repo into your own account first rather than pushing to the original.

---

## 2. Configure your league — before you download data

This step has to come **before** you ingest data or train, not after, which
is the order it's easy to assume from a plain read of `SETUP.md`. Training
computes each player's fantasy points using whatever scoring your active
roster source reports, and bakes those numbers into the model as its
prediction target. Train first and reconfigure later, and the model you end
up trusting learned the wrong game.

**Path A — manual CSV/YAML (recommended first).** No credentials; proves the
whole pipeline before you touch auth.

- Edit `data/roster_state/league_config.yaml` — your scoring type and lineup
  slots.
- Edit `my_roster.csv` and `free_agents.csv`.
- Edit `roster_meta.yaml` — as-of week, remaining FAAB.

**Path B — live ESPN connector.** Pulls your roster, free agents, and
scoring automatically. Required for the matchup simulator.

- `cp .env.example .env`
- Fill in `ESPN_LEAGUE_ID`, `ESPN_TEAM_ID`, `ESPN_YEAR`.
- Add `ESPN_S2` / `ESPN_SWID` cookies from your browser (DevTools →
  Application/Storage → Cookies → `fantasy.espn.com`).
- Set `EDGE_ENGINE_ROSTER_SOURCE=espn`.

Switching between them later is one line — `EDGE_ENGINE_ROSTER_SOURCE` is the
entire migration, by design; nothing downstream cares which source is
active.

> **Cookies are credentials — treat them like a password.** `ESPN_S2` and
> `SWID` can act on your ESPN account. Never paste them into a chat window,
> an AI assistant, a screenshot, or an email. Type them straight into
> `.env`, which is already gitignored. They also expire every few weeks;
> you'll refresh them periodically, not just once.

> **Running this setup inside a cloud dev environment, not your own
> machine?** A `.env` file may not survive between sessions — it's
> gitignored on purpose, so it was never saved anywhere durable. Set the
> same six variables as real environment variables on the environment
> itself instead, so they persist across container restarts without ever
> being typed into a chat transcript.

---

## 3. Download data & train

Now that your scoring is correct, pull NFL usage data and train against it.

```bash
python -m edge_engine.ingestion.pipeline --seasons 2018 2019 2020 2021 2022 2023 2024 2025
python -m edge_engine.model.train
python -m edge_engine.model.train_qb
```

Expect roughly **5.2 MAE** against a **5.7** naive baseline on the main
model, and a similar margin on the QB model — both re-measured on *your*
scoring, so don't be surprised if your numbers differ slightly from the
project's published ones. Run `python scripts/show_metrics.py` any time to
see your own current figures rather than trusting a number written in a doc.

You'll likely see a warning that the current season's stats are being
rebuilt from raw plays rather than fetched pre-aggregated — nflverse
publishes that summary table on a lag. **This is normal**, validated by the
project against a season where both sources exist (0.9989 correlation on
points).

This step is not weekly. `train` writes a static file; only re-run it if you
change your league's scoring or want to fold in a newly finished season.

---

## 4. Prove it end-to-end before you trust it

Passing tests proves the code is correct. It doesn't prove *your*
integration works — that your league's scoring loaded, your roster
resolved, and the dashboard actually renders against your data. Run through
this once, for real, before the first Tuesday you plan to rely on it.

- [ ] `python -m pytest -q` shows every test passing
- [ ] `streamlit run app.py` loads with no traceback
- [ ] All four tabs open cleanly — Waiver Wire, My Team, Trade Check, Draft
      Board
- [ ] Your real players appear on the Waiver Wire or My Team tab — check the
      "Couldn't look these up" panel for typos or team mismatches
- [ ] `python -m edge_engine.weekly` completes without a traceback

**What's actually been verified, and by whom:** the install path on a clean
3.12 venv, the full test suite, all four dashboard tabs rendering against
real historical usage and injury data, and the manual roster source
correctly separating resolved players from an intentional typo have all been
verified live, end-to-end, from a fresh checkout. The tool also degraded
gracefully rather than crashing when a schedule fetch failed — a real bug
this project's own test suite caught and fixed.

**Not independently re-verified for this document:** the live ESPN connector
against a real account (no test league was available to exercise it with),
and next-season play-by-play reconstruction beyond what the project's own
written evaluation already measured (`EVALUATION.md`). Both are traced
through the code and covered by existing tests, just not re-run against live
external services here — budget a few extra minutes the first time you flip
`EDGE_ENGINE_ROSTER_SOURCE=espn` in case something about your specific
league surfaces an edge case.

---

## 5. Run it on a weekly cadence

Usage data for a given week isn't available until that week's games finish,
which makes this inherently a midweek tool. Run it Tuesday or Wednesday
morning, before your league's waivers process:

```bash
python -m edge_engine.weekly   # refresh + rank + matchup, in one command
```

If you're on the manual roster source, this is also your weekly touch-up
moment: update `roster_meta.yaml`'s `as_of_week` and re-paste the current
free-agent list — about thirty seconds.

---

## Know the edges — these are decisions, not bugs

If one of these looks missing, it was left out on purpose. Filing it as a
bug will just point back here:

- No FAAB dollar amounts — confidence tiers only
- No trade scanner or opponent-roster mining
- No platforms besides ESPN
- No kickers or defenses — there's no usage data to model them from
- No start/sit logic inside the waiver ranking — that's a separate matchup
  simulator
- The draft board prices everything at market ADP and claims no edge over
  it — it's bookkeeping under a draft clock, not a projection engine

---

## Troubleshooting

| Message | What it means / fix |
|---|---|
| `ResolutionImpossible` during install | You're on Python 3.11 or older. Rebuild the venv with 3.12+. |
| ESPN rejected the cookies | They expired. Re-copy `espn_s2` / `SWID` from DevTools. |
| `No team with team_id=N` | `ESPN_TEAM_ID` is wrong — check *your* team's URL, not the league's. |
| `League's reception scoring is N pts/catch…` | Unsupported PPR value. Refuses on purpose rather than mis-scoring. |
| `0 rostered players` | Normal before your draft. Not an error state. |
| Names under "Needs attention" | Spelling or team mismatch in your CSV — surfaced, never silently dropped. |
| `ModuleNotFoundError: edge_engine` | You skipped `bash scripts/install.sh`, or its `pip install -e .` step failed. |

Any other failure: check the **Data gaps** block at the top of the dashboard
output first — whatever couldn't be computed is listed there, deliberately,
before the results rather than buried under them.
