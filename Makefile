.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y liquor_app || :
	@pip install -e .

run_preprocess:
	python -c 'from liquor_app.interface.main import preprocess; preprocess()'

run_train:
	python -c 'from liquor_app.interface.main import train; train()'

run_pred:
	python -c 'from liquor_app.interface.main import pred; pred()'

run_evaluate:
	python -c 'from liquor_app.interface.main import evaluate; evaluate()'

run_all: run_preprocess run_train run_pred run_evaluate

run_api:
	uvicorn liquor_app.api.fast:app --reload