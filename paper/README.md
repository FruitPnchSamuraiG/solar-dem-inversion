# Paper draft

Two-column draft in the CVPR 2027 author-kit layout (`cvpr.sty`, `ieeenat_fullname.bst`). Not a submission.

    cd paper
    ../.venv/bin/python make_figures.py      # needs output/meeting_followup/*.npz and ~/dem_preview
    latexmk -pdf main.tex

The folder compiles on its own, so it can be uploaded to Overleaf (compiler pdfLaTeX). The PDF is not tracked. `figures/`
holds copies of the three figures that `make_figures.py` writes to `results/plots/24_paper_20261031/`; copy them over after
regenerating. Check results live in that same results folder. `refs_verification.md` records how each citation was confirmed.
`refs.bib` is the verified bibliography plus two later additions.
