"""Filesystem path utilities and project directory resolver."""

from pathlib import Path


def get_project_root() -> Path:
    """Return the repository root directory by walking up from the current module."""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return Path.cwd()


def resolve_path(relative_or_absolute: str | Path) -> Path:
    """Resolve a path against the project root if it is relative."""
    p = Path(relative_or_absolute)
    if p.is_absolute():
        return p
    return get_project_root() / p


def ensure_dir(path: str | Path) -> Path:
    """Ensure directory exists, creating parent folders if necessary."""
    p = resolve_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
