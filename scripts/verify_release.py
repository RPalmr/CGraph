"""Verify wheel installation, dependency-free core, and absence of private artifacts."""

import os
import pathlib
import subprocess
import sys
import tarfile
import tempfile
import venv
import zipfile


def main():
    """Validate built artifacts in an isolated environment and execute the wheel's example."""
    wheel = pathlib.Path("dist/cgraph_audit-0.3.0-py3-none-any.whl").resolve()
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert any(n.endswith("METADATA") for n in names)
        assert not any(n.startswith(("docs/", ".codex/", "private/", "artifacts/")) for n in names)
    with tarfile.open("dist/cgraph_audit-0.3.0.tar.gz") as archive:
        names = archive.getnames()
        assert not any("/docs/" in n or "/.codex/" in n or "/artifacts/" in n for n in names)
    with tempfile.TemporaryDirectory(prefix="cgraph-core-smoke-") as directory:
        location = pathlib.Path(directory)
        venv.create(location, with_pip=True, symlinks=True)
        python = location / "bin/python"
        subprocess.run(
            [str(python), "-m", "pip", "install", "--no-index", "--no-deps", str(wheel)],
            check=True,
            capture_output=True,
        )
        script = "import importlib.util; from cgraph import DAG,factorize,intervene,identify; assert importlib.util.find_spec('numpy') is None; assert identify(DAG(('t','y'),(('t','y'),)),treatment='t',outcome='y').status=='identified'; print('Dependency-free wheel import and identification passed')"
        subprocess.run([str(python), "-c", script], cwd=directory, check=True)
        subprocess.run([str(location / "bin/cgraph"), "--version"], cwd=directory, check=True)
    script = "import sys,runpy; sys.path.insert(0,sys.argv[1]); import cgraph; assert '.whl/' in cgraph.__file__; runpy.run_path('examples/discover_graphs.py',run_name='__main__')"
    subprocess.run(
        [sys.executable, "-c", script, str(wheel)],
        check=True,
        env={**os.environ, "PYTHONPATH": str(wheel)},
    )
    print("Wheel and source distribution contain no private planning/session artifacts")


if __name__ == "__main__":
    main()
