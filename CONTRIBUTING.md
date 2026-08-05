# Contributing

Create a focused branch, keep changes scoped, and add tests alongside behavior. Before opening a
pull request, run:

```bash
python -m pip install -e '.[dev]'
make lint
make test
edge-opt demo --output-dir /tmp/edge-opt-demo
```

Hardware profiles must identify whether their numbers are measured, derived, or illustrative. New
sparse formats must include metadata overhead. New optimization paths must retain the fail-closed
quality gate; do not substitute quantization MSE for task accuracy unless that is explicitly the task
metric.

Keep framework-specific dependencies optional. Tests that require an optional framework should skip
cleanly when it is absent and run in a dedicated CI job when it is present.

