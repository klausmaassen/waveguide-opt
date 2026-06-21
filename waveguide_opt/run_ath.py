from __future__ import annotations

import subprocess
from pathlib import Path


def run_ath(
    ath_exe: str,
    cfg_path: str | Path,
    timeout_s: int = 600,
    workdir: str | Path | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [ath_exe, str(cfg_path)],
        check=False,
        text=True,
        capture_output=True,
        timeout=timeout_s,
        cwd=str(workdir) if workdir else None,
    )
