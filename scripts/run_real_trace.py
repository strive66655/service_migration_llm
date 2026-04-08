from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mdp_migration.io import save_json
from mdp_migration.plotting import plot_real_trace_results
from mdp_migration.real_trace import RealTraceConfig, run_real_trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=str(ROOT / "traceRealCellLocations.mat"))
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--max-user-each-cloud", type=int, default=50)
    parser.add_argument("--num-cells-with-cloud", type=int, default=100)
    parser.add_argument("--avail-resource-trans-factor", type=float, default=1.5)
    parser.add_argument("--avail-resource-migration-factor", type=float, default=1.5)
    parser.add_argument("--num-states-2d", type=int, default=20)
    args = parser.parse_args()

    config = RealTraceConfig(
        data_path=args.data_path,
        gamma=args.gamma,
        max_user_each_cloud=args.max_user_each_cloud,
        num_cells_with_cloud=args.num_cells_with_cloud,
        avail_resource_trans_factor=args.avail_resource_trans_factor,
        avail_resource_migration_factor=args.avail_resource_migration_factor,
        num_states_2d=args.num_states_2d,
    )
    results = run_real_trace(config)
    print(
        json.dumps(
            {
                "summary": results["summary"],
                "std_summary": results["std_summary"],
                "gain_stats": results["gain_stats"],
                "first_migrate_stats": results["first_migrate_stats"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        save_json(Path(args.save_dir) / "real_trace_results.json", results)
    if args.plot:
        plot_real_trace_results(results, args.save_dir)


if __name__ == "__main__":
    main()
