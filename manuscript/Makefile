.PHONY: all clean gitclean
arxiv_dir := arxiv

all: draft.pdf manuscript.pdf

clean:
	latexmk -CA && rm -rf ${arxiv_dir} && rm -f arxiv.zip

gitclean:
	git clean -fx

%.pdf: %.tex
	latexmk -pdf -pdflatex="pdflatex -interaction=nonstopmode" -use-make $^

manuscript.tex: draft.tex
	sed 's/^\\\paragraph/% \\\paragraph/g' $^ | sed 's/backref//g' > $@

draft.tex: stats.tex examples.tex result.tex
	touch draft.tex

arxiv.zip: manuscript.pdf
	mkdir -p ${arxiv_dir} ${arxiv_dir}/swarm_plot ${arxiv_dir}/confusion_matrix ${arxiv_dir}/images/examples ${arxiv_dir}/images/median
	cp *.tex *.bbl *.bst *.cls ${arxiv_dir}
	cp swarm_plot/plot-eps-converted-to.pdf ${arxiv_dir}/swarm_plot/plot.pdf
	cp confusion_matrix/cm-eps-converted-to.pdf ${arxiv_dir}/confusion_matrix/cm.pdf
	sed -i -e "s/\.eps}/\.pdf}/g" -e "s/\.png}/\.jpg}/g" ${arxiv_dir}/result.tex
	sed -i -e "s/\.eps}/\.pdf}/g" -e "s/\.png}/\.jpg}/g" ${arxiv_dir}/examples.tex
	cp --parents images/examples/*.png ${arxiv_dir}
	cp --parents images/median/*/*.png ${arxiv_dir}
	mogrify -format jpg ${arxiv_dir}/images/examples/*.png && rm ${arxiv_dir}/images/examples/*.png
	mogrify -resize 256x256 -format jpg ${arxiv_dir}/images/median/*/*.png && rm ${arxiv_dir}/images/median/*/*.png
	mv ${arxiv_dir}/manuscript.tex ${arxiv_dir}/ms.tex && mv ${arxiv_dir}/manuscript.bbl ${arxiv_dir}/ms.bbl
	cd ${arxiv_dir} && zip -r ../$@ *

arxiv.pdf: arxiv.zip
	cd arxiv && latexmk -pdf -pdflatex="pdflatex -interaction=nonstopmode" -use-make ms.tex && mv ms.pdf ../arxiv.pdf
