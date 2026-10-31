# Paper draft

Two-column draft in the CVPR 2027 author-kit layout (`cvpr.sty`, `ieeenat_fullname.bst`). Not a submission.

    cd paper
    ../.venv/bin/python make_figures.py      # needs output/meeting_followup/*.npz and ~/dem_preview
    latexmk -pdf main.tex

Figures and the new check results live in `results/plots/24_paper_20261031/`. `refs_verification.md` records how each
citation was confirmed. `refs.bib` is the verified bibliography plus two later additions.
