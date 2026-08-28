.PHONY: advanced-demo demo lint test

advanced-demo:
	PYTHONPATH=src python examples/advanced_optimization.py --output-dir reports/advanced

demo:
	PYTHONPATH=src python -m edge_opt demo --output-dir reports/demo

lint:
	ruff check src tests examples scripts

test:
	PYTHONPATH=src python -m unittest discover -s tests -v
