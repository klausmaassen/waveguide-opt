from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StartedAbec:
    pid: int | None


def _ps_quote(value: str | Path) -> str:
    text = str(value).replace("'", "''")
    return f"'{text}'"


def run_abec(
    abec_exe: str,
    project_path: str | Path,
    workdir: str | Path | None = None,
) -> StartedAbec:
    wd = workdir or Path(abec_exe).parent
    command = (
        "$p = Start-Process "
        f"-FilePath {_ps_quote(abec_exe)} "
        f"-ArgumentList @({_ps_quote(project_path)}) "
        f"-WorkingDirectory {_ps_quote(wd)} "
        "-WindowStyle Normal "
        "-PassThru; "
        "$p.Id"
    )
    startupinfo = None
    creationflags = 0
    if hasattr(subprocess, "STARTUPINFO"):
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 1
    if hasattr(subprocess, "CREATE_NEW_CONSOLE"):
        creationflags |= subprocess.CREATE_NEW_CONSOLE
    try:
        launcher = subprocess.Popen(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
            startupinfo=startupinfo,
            creationflags=creationflags,
        )
        return StartedAbec(launcher.pid)
    except OSError:
        return StartedAbec(None)
