"""Allow running the package as `python -m glt …` after a git clone (no
`pip install` required)."""
from .cli import main

if __name__ == "__main__":
    main()
