.PHONY: demo lint test

demo:
	PYTHONPATH=src python -m edge_opt demo --output-dir reports/demo

lint:
	ruff check src tests examples

test:
	PYTHONPATH=src python -m unittest discover -s tests -v

