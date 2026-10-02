from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_required_release_layout_exists() -> None:
    for directory in ("config", "model", "trdq", "utils", "test", "requirements"):
        assert (ROOT / directory).is_dir(), directory
    for filename in (
        "README.md",
        "README_zh-CN.md",
        "requirements.txt",
        "pyproject.toml",
        "CITATION.cff",
    ):
        assert (ROOT / filename).is_file(), filename

