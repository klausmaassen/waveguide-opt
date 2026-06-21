from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from .ath_config import write_cfg
from .evaluation import DEFAULT_OBJECTIVE_WEIGHTS, candidate_row, preview_from_row
from .geometry import candidate_preview
from .models import Bounds, Candidate, Settings
from .optimizer import optimize_cma, sample_candidates
from .run_abec import run_abec
from .run_ath import run_ath

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "web"


@dataclass
class JobState:
    running: bool = False
    phase: str = "idle"
    progress: int = 0
    total: int = 0
    run_dir: str | None = None
    best: dict[str, Any] | None = None
    candidates: list[dict[str, Any]] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)
    stop_requested: bool = False

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)


class AppState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.job = JobState(best=candidate_preview(Candidate(name="start_40mm")))
        self.thread: threading.Thread | None = None

    def log(self, message: str) -> None:
        with self.lock:
            stamp = time.strftime("%H:%M:%S")
            self.job.logs.append(f"{stamp} {message}")
            self.job.logs = self.job.logs[-100:]

    def update(self, **kwargs: Any) -> None:
        with self.lock:
            for key, value in kwargs.items():
                setattr(self.job, key, value)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return self.job.snapshot()


APP_STATE = AppState()


def _load_latest_completed_run() -> None:
    runs_root = ROOT / "runs" / "circsym"
    if not runs_root.exists():
        return
    files = sorted(runs_root.glob("web_*/web_candidates.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for file in files:
        try:
            rows = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not rows:
            continue
        best_row = min(rows, key=lambda row: row.get("pre_score", float("inf")))
        APP_STATE.update(
            phase="loaded",
            run_dir=str(file.parent),
            total=len(rows),
            progress=len(rows),
            best=preview_from_row(best_row),
            candidates=_visible_rows(rows, best_row),
        )
        APP_STATE.log(f"loaded latest run {file.parent.name}")
        return


def _visible_rows(rows: list[dict[str, Any]], best_row: dict[str, Any] | None, limit: int = 40) -> list[dict[str, Any]]:
    visible = rows[-limit:]
    if best_row and all(row.get("name") != best_row.get("name") for row in visible):
        visible = [best_row] + visible[-(limit - 1) :]
    return visible


def _load_settings() -> Settings:
    local = ROOT / "settings.local.json"
    if local.exists():
        return Settings.load(local)
    return Settings()


def _bounds_from_payload(payload: dict[str, Any]) -> Bounds:
    defaults = Bounds()
    values: dict[str, tuple[float, float]] = {}
    raw_bounds = payload.get("bounds") or {}
    for field_name in defaults.__dataclass_fields__:
        default_lo, default_hi = getattr(defaults, field_name)
        raw_pair = raw_bounds.get(field_name, [default_lo, default_hi])
        try:
            lo = float(raw_pair[0])
            hi = float(raw_pair[1])
        except (TypeError, ValueError, IndexError):
            lo, hi = default_lo, default_hi
        if hi < lo:
            lo, hi = hi, lo
        if hi == lo:
            hi = lo + 1e-6
        values[field_name] = (lo, hi)
    return Bounds(**values)


def _weights_from_payload(payload: dict[str, Any]) -> dict[str, float]:
    raw_weights = payload.get("objectiveWeights") or {}
    weights: dict[str, float] = {}
    for key, default in DEFAULT_OBJECTIVE_WEIGHTS.items():
        try:
            value = float(raw_weights.get(key, default))
        except (TypeError, ValueError):
            value = default
        weights[key] = max(0.0, value)
    if sum(weights.values()) <= 0:
        return DEFAULT_OBJECTIVE_WEIGHTS.copy()
    return weights


def _run_job(count: int, seed: int, run_ath_enabled: bool, bounds: Bounds, weights: dict[str, float]) -> None:
    run_name = time.strftime("web_%Y%m%d_%H%M%S")
    run_dir = ROOT / "runs" / "circsym" / run_name
    cfg_dir = run_dir / "configs"
    settings = _load_settings()
    ath_output_root = str(Path(settings.ath_output_root) / run_name)
    APP_STATE.update(running=True, phase="sampling", progress=0, total=count, run_dir=str(run_dir), candidates=[], stop_requested=False)
    APP_STATE.log(f"started run {run_name} with {count} candidates")
    APP_STATE.log("objective weights: " + ", ".join(f"{key}={value:.2f}" for key, value in weights.items()))
    try:
        candidates = sample_candidates(count, seed, bounds)
        rows: list[dict[str, Any]] = []
        best_row: dict[str, Any] | None = None
        best_score = float("inf")
        for idx, candidate in enumerate(candidates, start=1):
            if APP_STATE.snapshot().get("stop_requested"):
                APP_STATE.log("stop requested")
                break
            cfg_path = cfg_dir / f"{candidate.name}.cfg"
            write_cfg(candidate, cfg_path, output_root=ath_output_root)
            row = candidate_row(candidate, weights)
            score = row["pre_score"]
            row["cfg_path"] = str(cfg_path)
            row["ath_output_root"] = ath_output_root
            rows.append(row)
            if score < best_score:
                best_score = score
                best_row = row
                preview = preview_from_row(row)
                APP_STATE.update(best=preview)
            if idx % max(1, count // 200) == 0 or row is best_row:
                APP_STATE.update(phase="fast BEM", progress=idx, candidates=_visible_rows(rows, best_row))

        if run_ath_enabled and settings.ath_exe:
            APP_STATE.update(phase="running ATH", total=len(rows), progress=0)
            for idx, row in enumerate(rows, start=1):
                if APP_STATE.snapshot().get("stop_requested"):
                    break
                cfg_path = row["cfg_path"]
                result = run_ath(settings.ath_exe, cfg_path, timeout_s=900, workdir=settings.ath_workdir)
                row["ath_returncode"] = result.returncode
                APP_STATE.update(progress=idx, candidates=_visible_rows(rows, best_row))
                APP_STATE.log(f"ATH {Path(cfg_path).name}: return code {result.returncode}")
        elif run_ath_enabled:
            APP_STATE.log("ATH requested but settings.local.json has no ath_exe")

        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "web_candidates.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        if best_row:
            APP_STATE.log(f"best preliminary candidate: {best_row['name']} score {best_row['pre_score']:.3f}")
        APP_STATE.update(running=False, phase="complete", progress=len(rows), total=len(rows), candidates=_visible_rows(rows, best_row))
    except Exception as exc:
        APP_STATE.log(f"error: {exc}")
        APP_STATE.update(running=False, phase="error")


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(STATIC), **kwargs)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/status":
            self._json(APP_STATE.snapshot())
            return
        if parsed.path == "/api/best":
            self._json(APP_STATE.snapshot().get("best") or {})
            return
        if parsed.path.startswith("/api/cfg/"):
            name = unquote(parsed.path.removeprefix("/api/cfg/"))
            self._send_cfg(name)
            return
        super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/start":
            payload = self._read_json()
            self._start(payload)
            return
        if parsed.path == "/api/stop":
            APP_STATE.update(stop_requested=True)
            APP_STATE.log("stop flag set")
            self._json({"ok": True})
            return
        if parsed.path == "/api/open-abec-best":
            self._open_abec_best()
            return
        if parsed.path == "/api/open-best-folder":
            self._open_best_folder()
            return
        self.send_error(404)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw)

    def _json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _start(self, payload: dict[str, Any]) -> None:
        snapshot = APP_STATE.snapshot()
        if snapshot["running"]:
            self._json({"ok": False, "error": "job already running"}, 409)
            return
        seed = int(payload.get("seed", 1))
        bounds = _bounds_from_payload(payload)
        weights = _weights_from_payload(payload)
        mode = str(payload.get("mode", "sample"))
        if mode == "cma":
            max_evals = max(50, min(int(payload.get("count", 500)), 2000))
            thread = threading.Thread(
                target=_run_job_cma, args=(max_evals, seed, bounds, weights), daemon=True
            )
        else:
            run_ath_enabled = bool(payload.get("runAth", False))
            max_count = 500 if run_ath_enabled else 5000
            count = max(1, min(int(payload.get("count", 50)), max_count))
            thread = threading.Thread(
                target=_run_job, args=(count, seed, run_ath_enabled, bounds, weights), daemon=True
            )
        APP_STATE.thread = thread
        thread.start()
        self._json({"ok": True})

    def _send_cfg(self, name: str) -> None:
        snapshot = APP_STATE.snapshot()
        for row in snapshot.get("candidates", []):
            if row.get("name") == name and row.get("cfg_path"):
                path = Path(row["cfg_path"])
                if path.exists() and ROOT in path.resolve().parents:
                    body = path.read_text(encoding="utf-8").encode("utf-8")
                    self.send_response(200)
                    self.send_header("content-type", "text/plain; charset=utf-8")
                    self.send_header("content-length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
        self.send_error(404)

    def _open_abec_best(self) -> None:
        settings = _load_settings()
        if not settings.abec_exe:
            self._json({"ok": False, "error": "abec_exe missing"}, 400)
            return
        best = APP_STATE.snapshot().get("best") or {}
        cfg_path = best.get("cfg_path")
        if not cfg_path:
            self._json({"ok": False, "error": "no best ATH config path yet"}, 400)
            return
        candidate_name = (best.get("candidate") or {}).get("name")
        ath_output_root = best.get("ath_output_root") or settings.ath_output_root
        project = Path(ath_output_root) / str(candidate_name) / "ABEC_InfiniteBaffle" / "Project.abec"
        if not project.exists():
            self._json({"ok": False, "error": f"ABEC project not found: {project}"}, 404)
            return
        process = run_abec(settings.abec_exe, project, settings.abec_workdir)
        APP_STATE.log(f"opened ABEC project for {candidate_name}, pid {process.pid}")
        self._json({"ok": True, "pid": process.pid, "project": str(project)})

    def _best_project_path(self) -> Path | None:
        settings = _load_settings()
        best = APP_STATE.snapshot().get("best") or {}
        candidate_name = (best.get("candidate") or {}).get("name")
        if not candidate_name:
            return None
        ath_output_root = best.get("ath_output_root") or settings.ath_output_root
        project = Path(ath_output_root) / str(candidate_name) / "ABEC_InfiniteBaffle" / "Project.abec"
        return project if project.exists() else None

    def _open_best_folder(self) -> None:
        import subprocess

        project = self._best_project_path()
        if not project:
            self._json({"ok": False, "error": "ABEC project not found"}, 404)
            return
        subprocess.Popen(["explorer.exe", f"/select,{project}"])
        APP_STATE.log(f"opened folder for {project.name}")
        self._json({"ok": True, "project": str(project)})


def _run_job_cma(max_evals: int, seed: int, bounds: Bounds, weights: dict[str, float]) -> None:
    from dataclasses import replace as _dc_replace

    run_name = time.strftime("web_cma_%Y%m%d_%H%M%S")
    run_dir = ROOT / "runs" / "circsym" / run_name
    cfg_dir = run_dir / "configs"
    settings = _load_settings()
    ath_output_root = str(Path(settings.ath_output_root) / run_name)

    APP_STATE.update(
        running=True, phase="CMA-ES",
        progress=0, total=max_evals,
        run_dir=str(run_dir), candidates=[], stop_requested=False,
    )
    APP_STATE.log(f"CMA-ES gestartet: {run_name}  max_evals={max_evals}  seed={seed}")
    APP_STATE.log("Gewichte: " + ", ".join(f"{k}={v:.2f}" for k, v in weights.items()))

    try:
        partial: list[tuple[float, Any]] = []
        best_score = [float("inf")]

        class _Stop(Exception):
            pass

        def _progress(n: int, candidate: Any, score: float) -> None:
            partial.append((score, candidate))
            APP_STATE.update(progress=n)
            if APP_STATE.snapshot().get("stop_requested"):
                raise _Stop()
            if score < best_score[0]:
                best_score[0] = score
                row = candidate_row(candidate, weights)
                run_dir.mkdir(parents=True, exist_ok=True)
                cfg_path = cfg_dir / f"{candidate.name}.cfg"
                write_cfg(candidate, cfg_path, output_root=ath_output_root)
                row["cfg_path"] = str(cfg_path)
                row["ath_output_root"] = ath_output_root
                APP_STATE.update(best=preview_from_row(row))
                APP_STATE.log(f"neues Beste: {candidate.name}  score={score:.5f}")

        try:
            candidates = optimize_cma(
                bounds=bounds, max_evals=max_evals,
                seed=seed, weights=weights,
                progress_callback=_progress,
            )
        except _Stop:
            APP_STATE.log("Stop empfangen – speichere Zwischenstand")
            mouth_lo, mouth_hi = bounds.target_mouth_diameter
            partial.sort(key=lambda t: t[0])
            seen: set[str] = set()
            candidates = []
            for _, c in partial:
                if not (mouth_lo <= estimated_mouth_diameter(c) <= mouth_hi):
                    continue
                key = f"{c.length:.1f}_{c.coverage_angle:.1f}_{c.os_k:.3f}"
                if key not in seen:
                    seen.add(key)
                    candidates.append(c)

        rows: list[dict[str, Any]] = []
        best_row: dict[str, Any] | None = None
        for rank, c in enumerate(candidates[:50], 1):
            renamed = _dc_replace(c, name=f"cand_{rank:04d}")
            cfg_path = cfg_dir / f"{renamed.name}.cfg"
            run_dir.mkdir(parents=True, exist_ok=True)
            write_cfg(renamed, cfg_path, output_root=ath_output_root)
            row = candidate_row(renamed, weights)
            row["cfg_path"] = str(cfg_path)
            row["ath_output_root"] = ath_output_root
            rows.append(row)

        if rows:
            best_row = rows[0]
            APP_STATE.update(best=preview_from_row(best_row))
            APP_STATE.log(
                f"Fertig: {best_row['name']}  score={best_row['pre_score']:.5f}"
                f"  Mund={best_row.get('estimated_mouth_diameter', 0):.1f} mm"
            )

        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "web_candidates.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        APP_STATE.update(
            running=False, phase="complete",
            total=max_evals, progress=len(partial),
            candidates=_visible_rows(rows, best_row),
        )
    except Exception as exc:
        APP_STATE.log(f"Fehler: {exc}")
        APP_STATE.update(running=False, phase="error")


def run(host: str = "127.0.0.1", port: int = 8765) -> None:
    _load_latest_completed_run()
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Waveguide Opt UI: http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run()
