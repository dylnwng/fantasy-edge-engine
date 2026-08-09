# Setting Edge Engine up for your league

A start-to-finish guide for someone who didn't build this. Roughly 20–30
minutes, most of it waiting on a one-time data download.

Read [§4, "Will this actually work for my league?"](#4-will-this-actually-work-for-my-league)
**before** you invest the time — there are real constraints, and it's better
to find out now than after the download.

---

## 0. What you need

- **Python 3.11+**
- **An ESPN fantasy football league.** ESPN only. Sleeper, Yahoo and NFL.com
  are not supported and adding them is not a small change.
- ~2 GB of disk for the cached NFL data.
- Optionally, your ESPN login (for the live connector). You can skip this
  entirely and type your roster into a CSV instead.

---

## 1. Install

```bash
git clone https://github.com/dylnwng/fantasy-edge-engine.git
cd fantasy-edge-engine
python3 -m venv .venv && source .venv/bin/activate
bash scripts/install.sh
```

`install.sh` exists because `nfl_data_py` needs a two-step install — don't
replace it with a plain `pip install -r requirements.txt`.

Check it worked:

```bash
python -m pytest -q
```

You should see ~235 tests pass. They use synthetic data only — no network,
no credentials.

---

## 2. Download the NFL data (one-time, slow)

```bash
python -m edge_engine.ingestion.pipeline --seasons 2018 2019 2020 2021 2022 2023 2024 2025
python -m edge_engine.model.train
python -m edge_engine.model.train_qb     # quarterbacks (separate model, see §4)
```

**Expect the first command to take several minutes** — it's pulling every
play of eight NFL seasons. It caches to `data/raw/`, so you only pay this
once. `train` takes about 20 seconds.

A few things you'll see and shouldn't worry about:

- `WARNING nflverse has no pre-aggregated weekly stats for 2025 yet —
  reconstructing them from play-by-play instead.` **This is normal.**
  nflverse publishes its summary tables on a lag; the tool rebuilds them
  from raw plays and validates that reconstruction against a season where
  both exist (0.999 correlation). See `EVALUATION.md`.
- `train` prints its validation numbers. Roughly 5.2 MAE vs a 5.7 baseline
  is expected.

You do **not** need to re-run `train` weekly. It writes a static model;
weekly runs only need fresh usage data.

---

## 3. Connect your league

Two options. **Option B (manual) is the safer place to start** — it needs no
credentials and proves everything works before you add auth.

### Option A — live ESPN connector

Pulls your roster, the free-agent pool and your league's scoring settings
automatically. Required for the matchup simulator (it needs your opponent's
lineup, which no CSV can give it).

```bash
cp .env.example .env
```

Then fill in `.env`:

| Variable | Where to find it |
|---|---|
| `ESPN_LEAGUE_ID` | In your league URL: `.../league?leagueId=**123456**` |
| `ESPN_YEAR` | The season, e.g. `2026` |
| `ESPN_TEAM_ID` | In *your team's* URL: `...&teamId=**4**` |
| `ESPN_S2` | A browser cookie — see below |
| `ESPN_SWID` | A browser cookie — see below (keep the `{}` braces) |

To get the two cookies: log into `fantasy.espn.com` in a normal browser,
open DevTools (F12) → **Application** tab (Chrome/Edge) or **Storage**
(Firefox) → Cookies → `https://fantasy.espn.com`, and copy the `Value` of
the rows named `espn_s2` and `SWID`.

> **These cookies are credentials for your ESPN account.** Treat them like
> a password. `.env` is gitignored so you won't commit them by accident —
> don't paste them into chat, don't email them, don't put them in a
> screenshot. They also **expire every few weeks**; when they do you'll get
> a clear message telling you to refresh them, not a cryptic crash.

Finally set `EDGE_ENGINE_ROSTER_SOURCE=espn` in `.env`.

### Option B — manual CSV/YAML

No credentials. Edit four files in `data/roster_state/` (the repo ships
working examples — replace their contents with yours):

- **`league_config.yaml`** — once per season: scoring type, your starting
  lineup slots, waiver settings.
- **`my_roster.csv`** — `name,position,team`, one row per player.
- **`free_agents.csv`** — same three columns. Paste ESPN's "Available
  Players" list for QB/RB/WR/TE; you don't need every player in the league.
- **`roster_meta.yaml`** — `as_of_week`, `as_of_date`, `remaining_faab`.
  About a 30-second weekly update.

Bye weeks are looked up automatically — don't enter them.

Full format reference: [`data/roster_state/README.md`](data/roster_state/README.md).

Then run with `EDGE_ENGINE_ROSTER_SOURCE=manual` (the default).

---

## 4. Will this actually work for my league?

Be honest with yourself here before spending the time.

| Your league | Status |
|---|---|
| ESPN | ✅ Supported |
| Sleeper / Yahoo / NFL.com | ❌ Not supported |
| Standard, half-PPR or full-PPR | ✅ Supported |
| Any other per-reception value (e.g. 1.5 PPR) | ❌ **Refuses to run** rather than mis-score everything |
| Per-TD / per-turnover bonuses | ✅ Supported |
| Yardage-threshold bonuses (e.g. +3 at 100 rush yds) | ⚠️ Not computed — warns, doesn't crash |
| Snake draft | ✅ Draft board works |
| Auction draft | ❌ Not supported |
| Keeper / dynasty | ❌ No keeper valuation |

**Positions covered: QB, RB, WR, TE.** Kickers and defenses are **not**
ranked — nflverse carries essentially no player-level usage data for them,
so there's nothing to model.

Quarterbacks are covered by a **separate model** with its own features
(pass attempts, share of team pass volume, rush attempts) because the main
model's receiving metrics are structurally undefined for a passer. It's
validated the same way and on two independent seasons beat a
trailing-points baseline by ~10–12% MAE with the confidence interval clear
of zero. It needs its own training step:

```bash
python -m edge_engine.model.train_qb
```

Skip that and everything still works — QBs are simply absent from the
rankings, exactly as before they were supported.

**What the edge actually is:** roughly 10% better next-week point error than
"just look at recent points," and a 68–80% hit rate on the players it flags,
measured on a held-out season. Real, modest, and honestly tested — not a
league-winning oracle. `EVALUATION.md` documents four ideas that were tried
and rejected for failing to beat that bar.

---

## 5. Run it

```bash
streamlit run app.py        # the dashboard — start here
```

Four tabs: **Waiver Wire**, **My Roster**, **Trade**, **Draft Board**.

Or from the terminal:

```bash
python -m edge_engine.weekly              # the weekly routine: refresh + rank + matchup
python -m edge_engine.insights            # what's wrong with my roster
python -m edge_engine.trade --out "Player A" --in "Player B"
python -m edge_engine.draft               # ADP draft board
python -m edge_engine.draft --serve       # draft-night screen in a browser
```

**Weekly cadence:** run `weekly` on Tuesday or Wednesday morning, before
your league's waivers process. Usage data for a given week isn't available
until that week's games finish, so this is inherently a
midweek tool.

### Draft board setup (optional)

The board needs an ADP export you provide:

```bash
cp data/draft/adp.example.csv data/draft/adp.csv
```

⚠️ **The bundled example has fake ADP numbers** — real player names, but
ordered by last season's points. Replace them with a real export
(FantasyPros, Sleeper, Underdog — any source, four columns:
`name,position,team,adp`) before drafting off it. Details in
[`data/draft/README.md`](data/draft/README.md).

---

## 6. When something goes wrong

The tool is built to fail with an explanation rather than a stack trace.
The common ones:

| Message | Fix |
|---|---|
| `ESPN rejected the ESPN_S2/ESPN_SWID cookies` | Cookies expired. Re-copy them from DevTools (§3). |
| `No team with team_id=N in league M` | `ESPN_TEAM_ID` is wrong — check *your team's* URL, not the league's. |
| `League's reception scoring is 1.5 pts/catch` | Unsupported scoring. It refuses rather than silently mis-scoring. |
| `nflverse has no play-by-play for 2026 either` | That season hasn't been played yet. Normal in the offseason. |
| `data/draft/adp.csv doesn't exist` | Draft board needs an ADP export (§5). |
| `0 rostered players` | Normal before your draft. The tool says so instead of showing zeros. |
| Names in "Needs Attention" | Spelling/team mismatch in your CSV. They're surfaced, never silently dropped. |

If a run looks wrong, check the **Data gaps** block at the top of the output
first — anything the tool couldn't compute is listed there, deliberately
above the results rather than buried under them.

---

## 7. Making it yours

- `model_config.yaml` — `flag_margin` controls how selective the flagged
  list is (higher = fewer, higher-confidence names). It was swept on two
  seasons and left at 3.0 because raising it didn't reliably buy precision;
  see `EVALUATION.md` before changing it.
- `simulation_config.yaml` — matchup simulator settings. Values are
  validated at load, so a typo gets rejected instead of silently producing
  a confident wrong answer.
- `CLAUDE.md` — orientation for an AI agent working in the codebase,
  including which ideas have already been tried and rejected.
