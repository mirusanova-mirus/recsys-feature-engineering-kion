PYTHON ?= python3
CONFIG ?= configs/main.json
.NOTPARALLEL:

.PHONY: reproduce prepare-reference download features train catalog benchmark additional-plan additional replication independent report pdf test validate bundle
reproduce: prepare-reference download features train catalog benchmark
	$(PYTHON) -m recsys.additional --seal-only --renew-plan
	$(MAKE) additional replication independent test validate report bundle PYTHON=$(PYTHON)

prepare-reference:
	$(PYTHON) tools/prepare_reference_results.py

download:
	$(PYTHON) -u -m recsys.data --users 30000 --seed 42
features:
	$(PYTHON) -u -m recsys.build --config $(CONFIG)
train:
	$(PYTHON) -u -m recsys.experiment --config $(CONFIG)
catalog:
	$(PYTHON) -u -m recsys.catalog --config $(CONFIG)
benchmark:
	$(PYTHON) -u -m recsys.benchmark --config $(CONFIG)
additional-plan:
	$(PYTHON) -m recsys.additional --seal-only
additional: additional-plan
	$(PYTHON) -u -m recsys.additional
replication:
	$(PYTHON) -u -m recsys.build --config configs/temporal_replication.json
	$(PYTHON) -u -m recsys.experiment --config configs/temporal_replication.json
	$(PYTHON) -u -m recsys.catalog --config configs/temporal_replication.json
	$(PYTHON) -m recsys.validate --config configs/temporal_replication.json
independent:
	$(PYTHON) -u -m recsys.independent --config configs/user_holdout.json
report:
	$(PYTHON) tools/run_generated.py --output README.md --output REPORT.md --output ADDITIONAL.md -- $(PYTHON) -u -m recsys.report --config $(CONFIG)
pdf:
	$(MAKE) latex
test:
	$(PYTHON) -m unittest discover -s tests -v
validate:
	$(PYTHON) -m recsys.validate --config $(CONFIG)
	$(PYTHON) -m recsys.additional_validate
	$(PYTHON) -m recsys.independent --audit-only
check:
	$(PYTHON) tools/check_submission.py
bundle: check
	$(PYTHON) tools/package_submission.py --output submission/recsys_kion.zip

.PHONY: check

.PHONY: latex
latex:
	$(MAKE) -C report all
