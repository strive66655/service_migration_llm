# Dynamic Service Migration in Mobile Edge Computing Based on Markov Decision Process

This is the simulation code for the paper S. Wang, R. Urgaonkar, M. Zafer, T. He, K. Chan, K. K. Leung, "Dynamic service migration in mobile edge computing based on Markov decision process," IEEE/ACM Transactions on Networking, vol. 27, no. 3, pp. 1272-1288, Jun. 2019. (arXiv link: [https://arxiv.org/abs/1506.05261](https://arxiv.org/abs/1506.05261))

The original code runs best on MATLAB. This repository now also contains a Python port that aims to cover the same experiment flow and major outputs.

To reproduce the original MATLAB random walk result (Fig. 6 of the paper), run `mainRandomWalk.m`.

To reproduce the original MATLAB result with real base station locations (Fig. 8 of the paper), run `mainRealCellLocation.m`.

There are certain parameters in `mainRandomWalk.m` and `mainRealCellLocation.m` that can be changed for different experiments. The MATLAB algorithms are implemented in `algorithms.m`.

Real user traces are obtained from [http://crawdad.org/epfl/mobility/20090224/](http://crawdad.org/epfl/mobility/20090224/) and the base station locations are obtained from [http://www.antennasearch.com/](http://www.antennasearch.com/). These are saved in `traceRealCellLocations.mat`.

## Python Port

The Python implementation lives under `src/mdp_migration/` and the entry scripts live under `scripts/`.

### Random Walk Experiment

Run the default 2D experiment:

```bash
python scripts/run_random_walk.py
```

Run a faster smoke test:

```bash
python scripts/run_random_walk.py --seed-count 1 --gamma 0.9 --migrate-proportional 0 1
```

Run the 1D version:

```bash
python scripts/run_random_walk.py --use-1d --seed-count 1 --gamma 0.9 --migrate-proportional 0
```

Useful options:

- `--plot` together with `--save-dir` to write figures and JSON output
- `--num-states-left` and `--num-states-right` for 1D random walk
- `--num-states-2d` for 2D ring depth

### Real Trace Experiment

Run the default real-trace experiment:

```bash
python scripts/run_real_trace.py
```

Example with explicit parameters:

```bash
python scripts/run_real_trace.py --gamma 0.9 --max-user-each-cloud 50 --num-cells-with-cloud 100
```

### Result Comparison

If you export MATLAB results to JSON with matching field names, you can compare MATLAB and Python outputs with:

```bash
python scripts/compare_results.py --reference matlab_results.json --candidate python_results.json
```

This prints the largest absolute and relative differences first, which makes it easier to do final parity tuning.

### MATLAB JSON Export

To export MATLAB random-walk results to JSON, run:

```matlab
exportRandomWalkResults
```

This writes `matlab_random_walk_results.json`.

To export MATLAB real-trace results to JSON, run:

```matlab
exportRealTraceResults
```

This writes `matlab_real_trace_results.json`.

### Tests

Run the Python verification suite:

```bash
python -m unittest tests.test_core
```

### Coverage Notes

The Python port covers:

- 1D and 2D random walk experiment flows
- value iteration, policy iteration, modified policy iteration, myopic, never migrate, and always migrate policies
- real trace loading from `traceRealCellLocations.mat`
- MATLAB-style aggregate metrics for both experiment flows
- JSON export and result-diff tooling for MATLAB/Python parity checks

The original MATLAB files remain in the repository as the reference implementation.
