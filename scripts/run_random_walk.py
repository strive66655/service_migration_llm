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
from mdp_migration.plotting import plot_random_walk_results
from mdp_migration.random_walk import RandomWalkConfig, run_random_walk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--seed-count", type=int, default=None)
    parser.add_argument("--use-1d", action="store_true")
    parser.add_argument("--gamma", type=float, nargs="*", default=None)
    parser.add_argument("--migrate-proportional", type=float, nargs="*", default=None)
    parser.add_argument("--num-states-left", type=int, default=None)
    parser.add_argument("--num-states-right", type=int, default=None)
    parser.add_argument("--num-states-2d", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    defaults = RandomWalkConfig()
    config = RandomWalkConfig(
        use_2d=not args.use_1d,
        gamma_vector=tuple(args.gamma) if args.gamma else defaults.gamma_vector,
        migrate_proportional_vector=tuple(args.migrate_proportional) if args.migrate_proportional else defaults.migrate_proportional_vector,
        sim_seed_vector=tuple(range(1, args.seed_count + 1)) if args.seed_count is not None else defaults.sim_seed_vector,
        num_workers=args.workers if args.workers is not None else defaults.num_workers,
        num_states_left=args.num_states_left if args.num_states_left is not None else defaults.num_states_left,
        num_states_right=args.num_states_right if args.num_states_right is not None else defaults.num_states_right,
        num_states_2d=args.num_states_2d if args.num_states_2d is not None else defaults.num_states_2d,
        cell_dist=defaults.cell_dist,
        center_coordinate=defaults.center_coordinate,
        power_factor=defaults.power_factor,
    )
    results = run_random_walk(config)
    preview = {
        "use_2d": config.use_2d,
        "gamma_vector": list(config.gamma_vector),
        "migrate_proportional_vector": list(config.migrate_proportional_vector),
        "time_th_policy": results["time_th_policy"],
        "value_th_policy": results["value_th_policy"],
        "value_policy": results["value_policy"],
        "value_never": results["value_never"],
        "value_always": results["value_always"],
        "value_myopic": results["value_myopic"],
        "value_error": results["value_error"],
    }
    print(json.dumps(preview, indent=2, ensure_ascii=False))
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        save_json(Path(args.save_dir) / "random_walk_results.json", results)
    if args.plot:
        plot_random_walk_results(results, args.save_dir)


if __name__ == "__main__":
    main()
