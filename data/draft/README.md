# Draft inputs

## `adp.csv`

Average draft position, exported once in August from wherever you like
(FantasyPros, Sleeper, Underdog). The board only needs four columns:

```csv
name,position,team,adp
Ja'Marr Chase,WR,CIN,1.4
Bijan Robinson,RB,ATL,2.1
```

- `name` / `position` / `team` are resolved to nflverse ids through the same
  `PlayerLookup` the roster sources use, so ADP and usage history join correctly.
  Unmatched rows still appear on the board, priced at ADP, marked `?`.
- `adp` is a positive number. Lower drafts earlier.

No free public ADP API was found while building this (Sleeper's public endpoint
carries `gsis_id` and depth charts but not aggregate ADP; FantasyPros and Underdog
gate theirs). Since ADP is an annual input rather than a live feed, a hand-exported
CSV is the appropriate primary source, not a degraded fallback. A live source can be
added behind `MarketPriceSource` later without touching the board.

## `adp.example.csv`

A runnable example so the board works out of the box. **Its ADP numbers are
synthetic** — real player names, positions and teams, but ordered by last
season's total fantasy points as a stand-in for market price. That is *not*
ADP: it is exactly the backward-looking full-season average the divergence
tags exist to argue with, so a board built on it will under-tag.

Copy it to `adp.csv` and replace the numbers with a real export before
trusting anything on the board:

```bash
cp data/draft/adp.example.csv data/draft/adp.csv
```
