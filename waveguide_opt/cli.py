from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .ath_config import write_cfg
from .io_abec import read_polar_csv
from .models import Candidate, Settings
from .optimizer import write_batch
from .piston import target_table
from .ptt8 import export_csv
from .run_abec import run_abec
from .run_ath import run_ath
from .scoring import score_polar


def _candidate_from_json(path: str | None, name: str) -> Candidate:
    if not path:
        return Candidate(name=name)
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data.setdefault("name", name)
    return Candidate(**data)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="waveguide_opt")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("target-table")
    p.add_argument("--output")

    p = sub.add_parser("init-data")
    p.add_argument("--ptt8-zip", required=True)
    p.add_argument("--output-dir", default="data/ptt8")

    p = sub.add_parser("generate")
    p.add_argument("--name", default="start_40mm")
    p.add_argument("--candidate-json")
    p.add_argument("--output", default="configs/generated/start_40mm.cfg")
    p.add_argument("--settings")
    p.add_argument("--output-root")
    p.add_argument("--no-source-contours", action="store_true")

    p = sub.add_parser("batch")
    p.add_argument("--count", type=int, default=20)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--output-dir", default="runs/circsym/batch_001")
    p.add_argument("--settings")

    p = sub.add_parser("score")
    p.add_argument("--polar-csv", required=True)
    p.add_argument("--candidate-json")
    p.add_argument("--name", default="scored_candidate")
    p.add_argument("--output")

    p = sub.add_parser("run-ath")
    p.add_argument("--settings", required=True)
    p.add_argument("--cfg", required=True)
    p.add_argument("--timeout-s", type=int, default=600)

    p = sub.add_parser("run-abec")
    p.add_argument("--settings", required=True)
    p.add_argument("--project", required=True)

    p = sub.add_parser("ui")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)

    args = parser.parse_args(argv)
    if args.cmd == "target-table":
        rows = target_table()
        if args.output:
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        else:
            for row in rows:
                bw = row["beamwidth_6db_deg"]
                bw_text = ">180" if bw is None else f"{bw:.1f}"
                print(f"{row['freq_hz']:7.0f} Hz  -6 dB BW {bw_text:>6} deg  30/45/60 {row['db_30']:6.2f} {row['db_45']:6.2f} {row['db_60']:6.2f} dB")
    elif args.cmd == "init-data":
        spl, z = export_csv(args.ptt8_zip, args.output_dir)
        print(f"wrote {spl}")
        print(f"wrote {z}")
    elif args.cmd == "generate":
        candidate = _candidate_from_json(args.candidate_json, args.name)
        settings = Settings.load(args.settings) if args.settings else Settings()
        output_root = args.output_root or settings.ath_output_root
        path = write_cfg(candidate, args.output, include_source_contours=not args.no_source_contours, output_root=output_root)
        print(f"wrote {path}")
    elif args.cmd == "batch":
        settings = Settings.load(args.settings) if args.settings else Settings()
        path = write_batch(args.count, args.seed, args.output_dir, settings.ath_output_root)
        print(f"wrote {path}")
    elif args.cmd == "score":
        candidate = _candidate_from_json(args.candidate_json, args.name)
        score = score_polar(candidate, read_polar_csv(args.polar_csv))
        if args.output:
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(score.as_dict(), indent=2), encoding="utf-8")
            print(f"wrote {out}")
        else:
            print(json.dumps(score.as_dict(), indent=2))
    elif args.cmd == "run-ath":
        settings = Settings.load(args.settings)
        if not settings.ath_exe:
            raise SystemExit("ath_exe missing in settings")
        result = run_ath(settings.ath_exe, args.cfg, args.timeout_s, settings.ath_workdir)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        raise SystemExit(result.returncode)
    elif args.cmd == "run-abec":
        settings = Settings.load(args.settings)
        if not settings.abec_exe:
            raise SystemExit("abec_exe missing in settings")
        process = run_abec(settings.abec_exe, args.project, settings.abec_workdir)
        print(f"started ABEC pid {process.pid}")
    elif args.cmd == "ui":
        from .web_app import run

        run(args.host, args.port)
