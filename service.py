"""Start the editor with no console window, and keep a log.

Task Scheduler runs this with `pythonw.exe` at logon so nothing flashes up on
the desktop -- but that also means stdout has nowhere to go, and a server that
died on startup would fail silently. Someone else depends on this being up, so
everything it prints goes to `paper-edit.log` instead. That file is the first
place to look when the editor is not answering.

Run `Start Paper Edit.cmd` instead when you want to watch it in a window.
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "paper-edit.log"
MAX_LOG = 2_000_000


def main() -> None:
    # Roll the log rather than truncating it: if this is crash-looping, the
    # evidence is in the run that just failed, not the one about to start.
    if LOG.exists() and LOG.stat().st_size > MAX_LOG:
        LOG.replace(ROOT / "paper-edit.log.1")

    fh = open(LOG, "a", buffering=1, encoding="utf-8", errors="replace")
    sys.stdout = sys.stderr = fh

    sys.path.insert(0, str(ROOT))
    import uvicorn

    import server
    from paperedit import store

    print("")
    print("=== started " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " ===")
    store.init()
    for ip in server.local_addresses():
        print("reachable at http://" + ip + ":" + str(server.PORT))

    try:
        uvicorn.run(server.app, host="0.0.0.0", port=server.PORT, log_level="warning")
    except Exception as e:
        # Task Scheduler will restart us; make sure the reason is on disk first.
        print("CRASHED: " + type(e).__name__ + ": " + str(e))
        raise


if __name__ == "__main__":
    main()
