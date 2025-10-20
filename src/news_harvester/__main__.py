"""Permite ejecutar `python -m news_harvester`."""

from .cli import main


def run() -> None:
    main()


if __name__ == "__main__":  # pragma: no cover
    run()
