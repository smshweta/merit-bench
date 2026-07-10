#!/bin/sh
# Package the arXiv submission: LaTeX source + PDF figures only
# (arXiv compiles the source itself; no .pdf of the paper, no aux files).
set -eu
cd "$(dirname "$0")/../paper"
tar czf ../merit-arxiv.tar.gz main.tex figures/*.pdf
cd ..
echo "wrote merit-arxiv.tar.gz:"
tar tzf merit-arxiv.tar.gz
