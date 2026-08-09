"""Draft-night screen: a local HTTP server plus one static page.

    python -m edge_engine.draft --serve          # manual pick entry
    python -m edge_engine.draft --serve --espn   # poll ESPN's draft detail

**Framework: Python's stdlib `http.server`, not FastAPI.** The PRD calls
for FastAPI, and its reasoning is right -- Streamlit re-runs the whole
script on every state change and cannot hit a sub-second render against a
2-second poll. But FastAPI isn't installed here, and the PRD's own
constraint is "no framework, no build step, no dependency that can break
in August." Adding a dependency for the one event with zero recovery time
argues against itself. stdlib satisfies every requirement with nothing to
install, so that's the deviation and this is the reason.

**Everything is precomputed.** The board, tiers and ADP deltas are built
once at startup. During the draft the only work per pick is a set diff
and a sort over a shrinking list -- no model inference, no network calls
except the poll.

Design rules from §4.5, all enforced in the page below: one screen, no
scrolling, no tabs, no modals, no animation, keyboard-only, a persistent
staleness indicator that turns red past 10 seconds, high contrast, large
type.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer

from edge_engine.draft.board import BoardPlayer
from edge_engine.draft.tracker import DraftState, Pick, unfilled_slots

POLL_INTERVAL_SECONDS = 2.5  # never below live.MIN_POLL_SECONDS
STALE_AFTER_SECONDS = 10


@dataclass
class ServerState:
    """Mutable shared state. Guarded by a lock because the poll thread and
    the request handler both touch it."""
    draft: DraftState
    lineup_slots: dict[str, int]
    my_player_names: set[str] = field(default_factory=set)
    last_poll_ok: float = field(default_factory=time.time)
    last_error: str | None = None
    pick_source: object | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot(self) -> dict:
        with self.lock:
            draft = self.draft
            my_names = set(self.my_player_names)
            age = time.time() - self.last_poll_ok
            error = self.last_error

        mine = [b for b in draft.board if b.name in my_names]
        tiers = draft.tiers_remaining()

        return {
            "seconds_since_poll": round(age, 1),
            "stale": age > STALE_AFTER_SECONDS,
            "error": error,
            "picks_made": len(draft.picks),
            "tier_cliffs": [
                {"position": t.position, "tier": t.tier, "remaining": t.remaining,
                 "names": list(t.names[:3])}
                for t in tiers if t.remaining <= 3
            ][:8],
            "best_available": [
                {"name": b.name, "position": b.position, "adp": b.adp, "tier": b.tier,
                 "low_confidence": b.low_confidence,
                 # Value as a BAR LENGTH, never a decimal (§0.6 rule 6):
                 # how far the market's price sits from the top of the pool.
                 "bar": _value_bar(b, draft.board)}
                for b in draft.best_available(8)
            ],
            "run": draft.positional_run(),
            "my_roster": [{"name": b.name, "position": b.position} for b in mine],
            "unfilled": unfilled_slots(mine, self.lineup_slots),
        }


def _value_bar(player: BoardPlayer, board: list[BoardPlayer]) -> float:
    """0-1 bar length: how early this player goes relative to the board.
    A length, not a number, because humans parse relative length instantly
    and decimals slowly -- which is the whole constraint under a clock."""
    if not board:
        return 0.0
    worst = max(b.adp for b in board)
    best = min(b.adp for b in board)
    if worst == best:
        return 1.0
    return round(1.0 - (player.adp - best) / (worst - best), 3)


PAGE = """<!doctype html>
<meta charset="utf-8"><title>Draft</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background:#000; color:#fff; font:16px/1.35 -apple-system,system-ui,sans-serif;
         height:100vh; overflow:hidden; padding:14px; }
  .grid { display:grid; grid-template-columns: 1.25fr 1fr; gap:14px; height:100%; }
  h2 { font-size:13px; letter-spacing:.14em; text-transform:uppercase; color:#8a8a8a;
       margin-bottom:6px; }
  .cliff { font-size:26px; font-weight:700; padding:3px 0; }
  .cliff .n { color:#ff4444; }
  .avail { display:flex; align-items:center; gap:8px; font-size:22px; padding:3px 0; }
  .bar { height:11px; background:#2a6fff; border-radius:2px; min-width:2px; }
  .lc { color:#ffb020; }
  .stale { position:fixed; top:10px; right:14px; font-size:15px; color:#666; }
  .stale.bad { color:#fff; background:#c00; padding:3px 9px; border-radius:3px; }
  .err { background:#c00; padding:7px; margin-bottom:8px; font-size:15px; }
  .run { font-size:19px; color:#ffb020; margin-top:8px; }
  .need { font-size:19px; margin-top:8px; }
  li { list-style:none; font-size:17px; }
</style>
<div class="stale" id="stale">—</div>
<div id="err"></div>
<div class="grid">
  <div>
    <h2>Tier cliffs</h2><div id="cliffs"></div>
    <div class="run" id="run"></div>
  </div>
  <div>
    <h2>Best available</h2><div id="avail"></div>
    <div class="need" id="need"></div>
    <ul id="roster"></ul>
  </div>
</div>
<script>
// No animation anywhere: a transition is a period during which the screen
// is lying about board state.
function esc(s){ return String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }
async function tick(){
  let d;
  try { d = await (await fetch('/state')).json(); }
  catch (e) { document.getElementById('stale').textContent = 'NO CONNECTION';
              document.getElementById('stale').className = 'stale bad'; return; }

  const st = document.getElementById('stale');
  st.textContent = d.seconds_since_poll + 's';
  st.className = d.stale ? 'stale bad' : 'stale';

  document.getElementById('err').innerHTML = d.error
    ? '<div class="err">' + esc(d.error) + '</div>' : '';

  document.getElementById('cliffs').innerHTML = d.tier_cliffs.length
    ? d.tier_cliffs.map(t => '<div class="cliff">' + esc(t.position) + ' T' + t.tier +
        ': <span class="n">' + t.remaining + ' left</span></div>').join('')
    : '<div class="cliff" style="color:#666">none thin yet</div>';

  document.getElementById('avail').innerHTML = d.best_available.map(p =>
    '<div class="avail"><span class="bar" style="width:' + (p.bar*90) + 'px"></span>' +
    '<span class="' + (p.low_confidence ? 'lc' : '') + '">' + esc(p.name) +
    (p.low_confidence ? ' ?' : '') + '</span>' +
    '<span style="color:#777">' + esc(p.position) + ' T' + p.tier + '</span></div>').join('');

  document.getElementById('run').textContent = d.run || '';
  const need = Object.entries(d.unfilled).map(([k,v]) => k + ' x' + v).join(', ');
  document.getElementById('need').textContent = need ? 'Still need: ' + need : 'Starters filled';
  document.getElementById('roster').innerHTML = d.my_roster.map(p =>
    '<li>' + esc(p.position) + '  ' + esc(p.name) + '</li>').join('');
}
tick(); setInterval(tick, 2000);
</script>
"""


def _make_handler(state: ServerState):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path.startswith("/state"):
                body = json.dumps(state.snapshot()).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
            else:
                body = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass  # a request log per 2s poll would bury real output

    return Handler


def poll_loop(state: ServerState) -> None:
    """Background ESPN poll. Every failure is captured into state.error and
    rendered as a banner rather than killing the thread -- a cookie
    expiring mid-draft must degrade to a visible warning plus manual
    entry, not a dead screen."""
    while True:
        try:
            picks = state.pick_source.fetch_picks()  # type: ignore[union-attr]
            with state.lock:
                state.draft = state.draft.apply_all(picks)
                state.last_poll_ok = time.time()
                state.last_error = None
        except Exception as e:  # noqa: BLE001 - draft night: never die
            with state.lock:
                state.last_error = f"{type(e).__name__}: {e}. Board may be stale — enter picks manually."
        time.sleep(POLL_INTERVAL_SECONDS)


def serve(state: ServerState, port: int = 8765) -> None:
    if state.pick_source is not None:
        threading.Thread(target=poll_loop, args=(state,), daemon=True).start()
    httpd = HTTPServer(("127.0.0.1", port), _make_handler(state))
    print(f"Draft screen: http://127.0.0.1:{port}  (Ctrl-C to stop)")
    httpd.serve_forever()


def manual_pick(state: ServerState, name: str, mine: bool = False) -> bool:
    """Record a pick typed by hand. The fallback path when the poll fails,
    and the only path when not using --espn."""
    match = next((b for b in state.draft.board if b.name.lower() == name.lower()), None)
    if match is None:
        return False
    with state.lock:
        n = len(state.draft.picks) + 1
        state.draft = state.draft.apply(Pick((n - 1) // 12 + 1, (n - 1) % 12 + 1, match.name))
        if mine:
            state.my_player_names.add(match.name)
        state.last_poll_ok = time.time()
    return True
