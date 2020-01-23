.PHONY: all clean gitclean

all: draft.pdf manuscript.pdf

clean:
	latexmk -CA

gitclean:
	git clean -fx

%.pdf: %.tex
	latexmk -pdf -pdflatex="pdflatex -interaction=nonstopmode" -use-make $^

manuscript.tex: draft.tex
	sed 's/^\\\paragraph/% \\\paragraph/g' $^ | sed 's/backref//g' > $@

draft.tex: stats.tex examples.tex result.tex
	touch draft.tex
