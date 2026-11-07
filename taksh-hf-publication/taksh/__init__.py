"""Taksh release metadata.

The repository name and compatibility identifiers remain ``Quantsmind-LLM`` /
``quantsmind``; the product-facing name is Taksh.
"""

import argparse

__title__ = "Taksh"
__version__ = "1.0.0"
__release_status__ = "final"


def main(argv=None):
    parser = argparse.ArgumentParser(prog="taksh", description="Taksh release utilities")
    parser.add_argument(
        "--version",
        action="version",
        version=f"{__title__} {__version__} ({__release_status__})",
    )
    parser.parse_args(argv)
