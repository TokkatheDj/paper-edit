"""Has the editor actually been used, and did anything go wrong?

Answers the question you have after handing the editor to someone else: did it
get opened, did it do what they wanted, and did anything fail quietly. It reads
the same database the app uses and changes nothing.

It reports use, not people -- sign-ins carry no identity, so it cannot tell one
person's browser from another's.
"""
from __future__ import annotations

import datetime
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DB = ROOT / "projects" / "paperedit.db"


def when(t: float | None) -> str:
    if not t:
        return "never"
    stamp = datetime.datetime.fromtimestamp(t).strftime("%a %d %b %H:%M")
    gap = time.time() - t
    if gap < 3600:
        rel = f"{gap / 60:.0f} min ago"
    elif gap < 86400:
        rel = f"{gap / 3600:.0f} hours ago"
    else:
        rel = f"{gap / 86400:.0f} days ago"
    return f"{stamp}  ({rel})"


def main() -> int:
    if not DB.exists():
        print("No database yet -- nothing has been uploaded.")
        return 0
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row

    print("")
    print("  PAPER EDIT -- activity")
    print("  " + "-" * 52)

    try:
        rows = list(c.execute(
            "SELECT created, last_seen, label, address FROM sessions"
            " ORDER BY last_seen DESC, created DESC"))
        print(f"  Signed-in devices: {len(rows)}")
        for r in rows:
            label = r["label"] or "Unknown device"
            addr = f" from {r['address']}" if r["address"] else ""
            print(f"      {label}{addr}")
            print(f"          signed in {when(r['created'])}")
            print(f"          last used {when(r['last_seen'])}")
        if not rows:
            print("      nobody is signed in right now")
    except sqlite3.OperationalError:
        print("  Sign-in is not set up yet, or predates device labels.")
    print("")

    for p in c.execute("SELECT * FROM projects ORDER BY created"):
        total = c.execute("SELECT COUNT(*) FROM words WHERE project_id=?",
                          (p["id"],)).fetchone()[0]
        cut = c.execute("SELECT COUNT(*) FROM words WHERE project_id=? AND deleted=1",
                        (p["id"],)).fetchone()[0]
        print(f"  {p['name']}   [{p['status']}]")
        print(f"      {(p['duration'] or 0) / 60:.1f} min, {total} words")
        if cut:
            print(f"      *** {cut} words cut ({100 * cut / max(total, 1):.0f}% of the transcript)")
        else:
            print("      nothing cut yet")
        # Shown as plain settings, not as "switched on": Studio Sound sits on
        # 'auto' out of the box, and reporting a default as a choice someone
        # made is how you end up reading intent into nothing.
        print("      dead air {} | studio sound {} | captions {}".format(
            "on" if p["silence_on"] else "off",
            p["sound_preset"] or "auto", p["caption_style"] or "off"))
    print("")

    jobs = list(c.execute(
        "SELECT kind,status,message,created FROM jobs ORDER BY created DESC LIMIT 8"))
    print("  Recent jobs:")
    for j in jobs:
        mark = "FAILED  " if j["status"] == "failed" else "        "
        msg = (j["message"] or "").replace("\n", " ")[:44]
        print(f"      {mark}{j['kind']:9} {when(j['created']):32} {msg}")
    if not jobs:
        print("      none yet")

    # Only failures from the last day are worth raising. Older ones are
    # history, and showing them as live problems is just noise.
    day = time.time() - 86400
    fresh = [j for j in jobs if j['status'] == 'failed' and j['created'] > day]
    stale = [j for j in jobs if j['status'] == 'failed' and j['created'] <= day]
    print('')
    if fresh:
        print(f'  {len(fresh)} job(s) FAILED in the last 24 hours:')
        print('      ' + (fresh[0]['message'] or '').replace(chr(10), ' ')[:200])
    else:
        print('  Nothing has failed in the last 24 hours.')
        if stale:
            print(f'  ({len(stale)} older failure(s) listed above -- already fixed.)')
    print("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
