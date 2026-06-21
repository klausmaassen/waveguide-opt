from __future__ import annotations

import argparse
import csv
import json
import time
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

    p = sub.add_parser("optimize", help="CMA-ES optimisation (requires: pip install cma)")
    p.add_argument("--max-evals", type=int, default=500)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--sigma0", type=float, default=0.25)
    p.add_argument("--output-dir", default=None, help="default: runs/circsym/web_cma_<timestamp>")
    p.add_argument("--settings")
    p.add_argument("--bounds-json", help="JSON file with bounds overrides, e.g. {\"coverage_angle\": [50, 65]}")
    p.add_argument("--top", type=int, default=20, help="write CFG files for the top N candidates")

    p = sub.add_parser("export-top", help="Select best candidates from a run dir and write ATH CFG files")
    p.add_argument("--run-dir", required=True, help="directory containing web_candidates.json")
    p.add_argument("--count", type=int, default=5)
    p.add_argument("--diverse", action="store_true", default=True, help="greedy farthest-point selection (default)")
    p.add_argument("--no-diverse", dest="diverse", action="store_false", help="pure top-N by score")
    p.add_argument("--top-fraction", type=float, default=0.20, help="fraction of candidates forming the diversity pool")
    p.add_argument("--output-dir", default=None, help="default: <run-dir>/export_top")
    p.add_argument("--settings")

    p = sub.add_parser("sensitivity", help="Finite-difference score sensitivity for each parameter")
    p.add_argument("--candidate-json", help="JSON file with candidate parameters (omit to use defaults)")
    p.add_argument("--name", default="start_40mm")
    p.add_argument("--delta", type=float, default=0.10, help="step size as fraction of each parameter's range")
    p.add_argument("--output", help="write results as JSON to this path")

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

    elif args.cmd == "optimize":
        from .evaluation import DEFAULT_OBJECTIVE_WEIGHTS, candidate_row
        from .models import Bounds
        from .optimizer import SHAPE_PARAMS, optimize_cma

        bounds = Bounds()
        if args.bounds_json:
            raw = json.loads(Path(args.bounds_json).read_text(encoding="utf-8"))
            overrides = {k: tuple(v) for k, v in raw.items() if hasattr(Bounds(), k)}
            bounds = Bounds(**{**{f: getattr(Bounds(), f) for f in Bounds().__dataclass_fields__}, **overrides})

        settings = Settings.load(args.settings) if args.settings else Settings()
        run_name = f"web_cma_{time.strftime('%Y%m%d_%H%M%S')}"
        output_dir = Path(args.output_dir) if args.output_dir else Path("runs/circsym") / run_name
        cfg_dir = output_dir / "configs"

        t0 = time.time()
        counts = [0]

        def _progress(n: int, _c: object, score: float) -> None:
            counts[0] = n
            if n % 50 == 0 or n == 1:
                print(f"  eval {n:4d}  score {score:.5f}  [{time.time() - t0:.0f}s]")

        print(f"CMA-ES: max_evals={args.max_evals}  sigma0={args.sigma0}  seed={args.seed}")
        candidates = optimize_cma(
            bounds=bounds,
            sigma0=args.sigma0,
            max_evals=args.max_evals,
            seed=args.seed,
            progress_callback=_progress,
        )
        print(f"Done - {counts[0]} evaluations, {len(candidates)} valid candidates")

        top = candidates[: args.top]
        weights = DEFAULT_OBJECTIVE_WEIGHTS
        rows = []
        for rank, candidate in enumerate(top, 1):
            candidate = candidate.__class__(name=f"cand_{rank:04d}", **{
                p: getattr(candidate, p) for p in SHAPE_PARAMS + [
                    "target_mouth_diameter", "mesh_length_segments",
                    "mesh_throat_resolution", "mesh_mouth_resolution",
                    "abec_f1", "abec_f2", "abec_num_frequencies",
                    "polar_max_angle", "polar_points",
                ]
            })
            cfg_path = cfg_dir / f"{candidate.name}.cfg"
            write_cfg(candidate, cfg_path, output_root=settings.ath_output_root)
            row = candidate_row(candidate, weights)
            row["cfg_path"] = str(cfg_path)
            rows.append(row)

        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "web_candidates.json"
        json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

        summary_fields = ["name"] + SHAPE_PARAMS + [
            "estimated_mouth_diameter", "acoustic_score",
            "constant_directivity", "ptt8_crossover_match",
            "off_axis_smoothness", "response_ripple", "resonance_penalty",
            "beamwidth_2000", "beamwidth_4000",
        ]
        summary_path = output_dir / "summary.csv"
        with summary_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=summary_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

        print(f"Output: {output_dir}")
        print(f"  CFG files : {cfg_dir}/ ({len(top)} files)")
        print(f"  Summary   : {summary_path}")
        print(f"  Web JSON  : {json_path}  (loadable in 'waveguide-opt ui')")
        if rows:
            best = rows[0]
            print(f"Best: {best['name']}  score={best['acoustic_score']:.5f}"
                  f"  mouth={best.get('estimated_mouth_diameter', 0):.1f} mm"
                  f"  BW@2kHz={best.get('beamwidth_2000') or 0:.1f}°")

    elif args.cmd == "export-top":
        from .ath_config import write_cfg as _write_cfg
        from .evaluation import candidate_from_row
        from .optimizer import SHAPE_PARAMS, select_diverse

        run_dir = Path(args.run_dir)
        json_path = run_dir / "web_candidates.json"
        if not json_path.exists():
            raise SystemExit(f"web_candidates.json not found in {run_dir}")

        rows = json.loads(json_path.read_text(encoding="utf-8"))
        if not rows:
            raise SystemExit("No candidates in run dir")

        if args.diverse:
            selected = select_diverse(rows, args.count, args.top_fraction)
        else:
            selected = sorted(rows, key=lambda r: float(r.get("pre_score", float("inf"))))[: args.count]

        settings = Settings.load(args.settings) if args.settings else Settings()
        output_dir = Path(args.output_dir) if args.output_dir else run_dir / "export_top"
        cfg_dir = output_dir / "configs"
        output_dir.mkdir(parents=True, exist_ok=True)

        for row in selected:
            candidate = candidate_from_row(row)
            _write_cfg(candidate, cfg_dir / f"{candidate.name}.cfg", output_root=settings.ath_output_root)

        summary_fields = ["name"] + SHAPE_PARAMS + [
            "estimated_mouth_diameter", "acoustic_score",
            "constant_directivity", "ptt8_crossover_match",
            "off_axis_smoothness", "response_ripple", "resonance_penalty",
            "beamwidth_2000", "beamwidth_4000",
        ]
        summary_path = output_dir / "summary.csv"
        with summary_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=summary_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(selected)

        readme_lines = [
            f"Export from : {run_dir}",
            f"Selection   : {'diverse top' if args.diverse else 'top'} {len(selected)} of {len(rows)} candidates",
            f"Pool        : top {args.top_fraction:.0%} by score",
            "",
            "Candidates (sorted by score):",
        ]
        for i, row in enumerate(selected, 1):
            score = row.get("acoustic_score", row.get("pre_score", "?"))
            mouth = row.get("estimated_mouth_diameter", 0)
            bw2 = row.get("beamwidth_2000") or 0
            bw4 = row.get("beamwidth_4000") or 0
            score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
            readme_lines.append(
                f"  {i}. {row['name']:16s}  score={score_str}  mouth={mouth:.1f}mm"
                f"  BW@2kHz={bw2:.1f}°  BW@4kHz={bw4:.1f}°"
            )
        readme_lines += [
            "",
            "Next steps:",
            "  1. Run ATH with each .cfg file in configs/",
            "  2. Open the generated ABEC project",
            "  3. Run ABEC simulation",
            "  4. Compare ABEC polar results against summary.csv scores",
        ]
        (output_dir / "README.txt").write_text("\n".join(readme_lines), encoding="utf-8")

        print(f"Exported {len(selected)} candidates to {output_dir}")
        print(f"  CFG files : {cfg_dir}/")
        print(f"  Summary   : {summary_path}")
        print(f"  README    : {output_dir / 'README.txt'}")

    elif args.cmd == "sensitivity":
        from .optimizer import sensitivity_analysis

        candidate = _candidate_from_json(args.candidate_json, args.name)
        results = sensitivity_analysis(candidate, delta=args.delta)

        if args.output:
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"wrote {out}")
        else:
            print(f"Sensitivity for '{candidate.name}'  (delta={args.delta:.0%} of each parameter's range)\n")
            print(f"  {'Parameter':<22} {'Base':>9} {'Score-':>9} {'Score0':>9} {'Score+':>9} {'Sensitivity':>12}")
            print("  " + "-" * 74)
            for r in results:
                print(
                    f"  {r['parameter']:<22} {r['base_value']:>9.4f}"
                    f" {r['score_minus']:>9.5f} {r['score_base']:>9.5f} {r['score_plus']:>9.5f}"
                    f" {r['sensitivity']:>12.5f}"
                )
