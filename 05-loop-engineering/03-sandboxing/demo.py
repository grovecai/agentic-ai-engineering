"""Subprocess limits for a fixed code-evaluation demonstration."""

import os
import subprocess
import sys


def run_snippet(code: str, timeout: float = 2.0) -> str:
    if len(code) > 2000:
        raise ValueError("Snippet too large")

    def limits() -> None:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (1, 1))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
        if sys.platform.startswith("linux"):
            resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))

    env = {"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
        env=env,
        cwd="/",
        preexec_fn=limits if os.name == "posix" else None,
    )
    return (
        f"exit={result.returncode} stdout={result.stdout[:1000]!r} stderr={result.stderr[:1000]!r}"
    )


if __name__ == "__main__":
    print(run_snippet("print(sum([2, 3, 5]))"))
