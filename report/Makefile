LATEXMK ?= latexmk
.PHONY: all clean
all:
	mkdir -p build
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build main.tex
	cp build/main.pdf report.pdf
clean:
	$(LATEXMK) -c -outdir=build main.tex
