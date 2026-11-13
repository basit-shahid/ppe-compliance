"""Command-line entry point; model execution is the next milestone."""

import argparse

from ppe_compliance import __version__


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PPE compliance prototype: project scaffold."
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.parse_args()
    parser.print_help()


if __name__ == "__main__":
    main()
