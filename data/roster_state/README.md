# Roster state — manual input format

This is the P0 "manual entry" implementation of the roster-state interface
(`edge_engine.roster.interface.RosterStateSource`). Nothing downstream reads
these files directly — everything goes through `get_rostered_players()`,
`get_free_agents()`, `get_league_config()`, `get_roster_meta()`. That's what
lets this get swapped for a live ESPN connector later without touching the
model or ranking code.

## One-time setup (`league_config.yaml`)

Fill in once per season, before Week 1:

- `season` — the NFL season year.
- `scoring.ppr_type` — `standard`, `half_ppr`, or `full_ppr`.
- `scoring.bonuses` — any custom bonus scoring on top of the base PPR type,
  as free-form `stat_name: points` pairs.
- `lineup_slots` — your league's starting lineup slot counts, e.g. `RB: 2`.
  Used later by requirement 5 to judge position scarcity against what your
  league actually starts, not just how many you happen to have rostered.
- `waivers` — system (`FAAB` or `priority`), season budget, minimum bid,
  the day waivers clear, and what happens to a player nobody claims.

## Weekly update (~5 minutes)

**`roster_meta.yaml`** — three fields: `as_of_week`, `as_of_date`,
`remaining_faab`. This is intentionally separate from `league_config.yaml`
so a weekly touch-up doesn't mean re-entering your whole league setup.

**`my_roster.csv`** — your current roster. Three columns:

```csv
name,position,team
Josh Jacobs,RB,GB
Kyren Williams,RB,LA
```

- `name` — as it appears on ESPN; doesn't need to be exact, just
  recognizable (see matching notes below).
- `position` — `QB`, `RB`, `WR`, `TE`, `K`, `DST`, etc.
- `team` — any common abbreviation works (`LAR` or `LA`, `WAS` or `WSH`,
  etc.) — team aliases are resolved automatically.

**`free_agents.csv`** — same three columns, listing the free agents you
actually want tracked. In practice: paste ESPN's "Available Players" list
for the positions you care about (QB/RB/WR/TE) rather than every rostered
long-snapper in the league — the model only needs the players worth
watching.

Bye weeks are **not** entered manually — they're looked up automatically
from the published NFL schedule.

## Name matching

Each row is resolved to an nflverse player ID by normalized name +
position (and team, if the name alone is ambiguous — e.g. two players
named "Josh Allen"). If a row doesn't resolve, it still comes back from
`get_rostered_players()` / `get_free_agents()` with `match_status` set to
`"unmatched"` or `"ambiguous"` and a `match_note` explaining why, rather
than being silently dropped — check those before trusting a weekly run.
