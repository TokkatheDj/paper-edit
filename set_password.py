"""Set or reset the editor's password from the desktop.

Run this when nobody can remember it, or to change it. Setting a new password
signs every device out, so everyone signs in again with the new one.
"""
from __future__ import annotations

import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paperedit import auth, store


def main() -> int:
    store.init()
    if auth.has_password():
        print("A password is already set. Continuing will replace it and sign")
        print("every device out.")
    else:
        print("No password is set yet.")
    print("")

    first = getpass.getpass("New password: ")
    if len(first) < 4:
        print("Too short -- use at least 4 characters. Nothing changed.")
        return 1
    if first != getpass.getpass("Type it again: "):
        print("Those did not match. Nothing changed.")
        return 1

    auth.set_password(first)
    print("")
    print("Done. Everyone signs in again with the new password.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
