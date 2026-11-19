"""Allow ``python -m taksh`` to run the Taksh CLI.

Delegates to ``taksh.__init__:main``. Keeps the package version at 1.0.0.
"""

from __future__ import annotations

import sys

from . import main

if __name__ == "__main__":
    sys.exit(main() or 0)
