"""
Test the `run_pylint_on_repo` function from the `linting` module.
"""
from types import SimpleNamespace  # standard library
import pandas as pd  # third-party
from src import linting  # type: ignore # local


def test_run_pylint_on_repo(tmp_path, monkeypatch):
    """
    Test run_pylint_on_repo by simulating a repo with a dummy Python file,
    mocking list_py_files to return it, and monkeypatching subprocess.
    """
    # Fake repo path with dummy .py file
    repo_path = tmp_path
    dummy_file = repo_path / "dummy.py"
    dummy_file.write_text("print('hello')")

    # Monkeypatch list_py_files to return our dummy
    monkeypatch.setattr("src.linting.list_py_files", lambda _: [str(dummy_file)])

    # Monkeypatch subprocess to simulate pylint JSON output
    def fake_run(*_, **kwargs):
        kwargs["stdout"].write(
            '[{"path": "dummy.py", "message-id": "E0001", "symbol": "error"}]'
        )
        return SimpleNamespace(stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)

    csv_path = linting.run_pylint_on_repo("DummyRepo", str(repo_path))
    assert csv_path and csv_path.endswith(".csv")
    df = pd.read_csv(csv_path)
    assert "message-id" in df.columns
