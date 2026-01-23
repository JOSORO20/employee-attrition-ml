.PHONY: install lint test train evaluate serve

install:
	python -m pip install -r requirements.txt

lint:
	ruff check src tests
	black --check src tests

test:
	pytest -q

train:
	python -m cc_approval.cli train

evaluate:
	python -m cc_approval.cli evaluate

serve:
	uvicorn cc_approval.app:app --reload --port 8000
