
.PHONY: build
build:
	python -m build

.PHONY: dist
dist: build
	python -m twine upload dist/*

.PHONY: clean
clean:
	rm -rf build/ dist/ *.egg-info/

