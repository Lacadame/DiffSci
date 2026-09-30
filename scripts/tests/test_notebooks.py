"""Exercise notebook conversion without executing research code or loading data."""

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import time

import pytest

jupytext = pytest.importorskip("jupytext")
nbformat = pytest.importorskip("nbformat")

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("notebook_tools", ROOT / "scripts/notebooks.py")
notebook_tools = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(notebook_tools)


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    # Jupytext signs rebuilt notebooks; keep its trust database inside the test.
    monkeypatch.setenv("JUPYTER_DATA_DIR", str(tmp_path / ".jupyter"))
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for name in (".gitignore", "paper_odds/jupytext.toml"):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    return tmp_path


def write_notebook(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.v4.new_notebook(
        metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
        cells=[
            nbformat.v4.new_markdown_cell("# Example\n\nProse survives conversion."),
            nbformat.v4.new_code_cell(
                "value = 42", execution_count=3, metadata={"tags": ["parameters"]},
                outputs=[nbformat.v4.new_output("stream", name="stdout", text="saved output\n")],
            ),
        ],
    )
    nbformat.write(notebook, path)
    return notebook


def test_build_all_repository_sources_in_fresh_checkout(checkout):
    sources = notebook_tools.discover(ROOT)
    assert len(sources) >= 4
    for source in sources:
        target = checkout / source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / source, target)

    assert notebook_tools.main(["build"], root=checkout) == 0
    for source in sources:
        expected = jupytext.read(checkout / source)
        notebook = nbformat.read(checkout / source.with_suffix(".ipynb"), as_version=4)
        nbformat.validate(notebook)
        assert [(c.cell_type, c.source) for c in notebook.cells] == [
            (c.cell_type, c.source) for c in expected.cells
        ]
        assert notebook.metadata.kernelspec == expected.metadata.kernelspec
        assert all(not c.get("outputs") and c.get("execution_count") is None for c in notebook.cells)
        assert (checkout / source).read_bytes() == (ROOT / source).read_bytes()


def test_build_preserves_existing_local_notebook(checkout):
    path = checkout / "paper_odds/example.ipynb"
    notebook = write_notebook(path)
    jupytext.write(notebook, path.with_suffix(".py"), fmt="py:percent")
    original = path.read_bytes()
    assert notebook_tools.main(["build"], root=checkout) == 0
    assert path.read_bytes() == original


def test_discovery_respects_ignored_workspaces_and_nested_folders(checkout):
    allowed = [
        "paper_odds/example.ipynb",
        "paper_odds/nested/example.ipynb",
        "paper_odds/space in name.ipynb",
    ]
    excluded = [
        "paper_odds/outputs/example.ipynb",
        "paper_odds/.ipynb_checkpoints/example.ipynb",
        "notebooks/tutorials/example.ipynb",
        "standalone.ipynb",
    ]
    for name in allowed + excluded:
        write_notebook(checkout / name)
    # Ordinary helper scripts are not notebook sources, even when they mention the header.
    (checkout / "paper_odds/helper.py").write_text('marker = "#   jupytext:"\n')
    subprocess.run(["git", "add", "-f", allowed[0]], cwd=checkout, check=True)
    assert notebook_tools.discover(checkout) == sorted(Path(name).with_suffix(".py") for name in allowed)


def test_sync_exports_new_notebook_and_preserves_outputs_both_directions(checkout):
    path = checkout / "paper_odds/example.ipynb"
    original = write_notebook(path)
    assert notebook_tools.main(["sync"], root=checkout) == 0
    source = path.with_suffix(".py")
    assert source.is_file()
    assert jupytext.read(source).cells[1].metadata.tags == ["parameters"]
    assert not jupytext.read(source).cells[1].outputs

    source.write_text(source.read_text().replace("value = 42", "value = 43"))
    now = time.time()
    os.utime(path, (now - 60, now - 60))
    os.utime(source, (now - 30, now - 30))
    assert notebook_tools.main(["sync"], root=checkout) == 0
    updated = nbformat.read(path, as_version=4)
    assert updated.cells[1].source == "value = 43"
    assert updated.cells[1].outputs == original.cells[1].outputs

    updated.cells[1].source = "value = 44"
    nbformat.write(updated, path)
    now = time.time()
    os.utime(source, (now - 60, now - 60))
    os.utime(path, (now - 30, now - 30))
    assert notebook_tools.main(["sync"], root=checkout) == 0
    assert jupytext.read(source).cells[1].source == "value = 44"
    assert nbformat.read(path, as_version=4).cells[1].outputs == original.cells[1].outputs


def test_sync_refuses_to_overwrite_ordinary_python_source(checkout):
    notebook = checkout / "paper_odds/example.ipynb"
    write_notebook(notebook)
    source = notebook.with_suffix(".py")
    source.write_text("# Existing handwritten code\nvalue = 99\n")
    before = source.read_bytes()
    with pytest.raises(SystemExit):
        notebook_tools.main(["sync"], root=checkout)
    assert source.read_bytes() == before
