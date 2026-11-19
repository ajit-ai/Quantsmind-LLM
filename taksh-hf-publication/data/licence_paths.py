"""Path helpers shared by the licence verifier and the corpus builder.

Kept in its own module so neither has to import the other:
``verify_licenses`` needs the transport and config loader from the builder,
while the builder needs these path predicates to enforce exclusion decisions.
"""

from __future__ import annotations


def normalise_prefix(prefix: str) -> str:
    """Path prefixes are compared with a trailing slash so ``src`` != ``src2``."""
    return prefix if prefix.endswith("/") else prefix + "/"


def path_within(path: str, prefixes) -> bool:
    """True if ``path`` is inside any of the given subtrees.

    Descendants of an excluded subtree are excluded too: excluding
    ``src/blib2to3`` must also exclude ``src/blib2to3/pgen2/driver.py``. A path
    equal to the prefix is itself inside it, with or without a trailing slash.
    """
    if not prefixes:
        return False
    candidate = path.rstrip("/")
    for prefix in prefixes:
        p = normalise_prefix(prefix)
        if candidate == prefix.rstrip("/") or path.startswith(p):
            return True
    return False