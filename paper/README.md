# Paper: EEG-LeJEPA manuscript

LaTeX source for the IEEE TNSRE submission. Built from `PAPER_OUTLINE.md` at the
repository root.

## Compile

```bash
cd paper/
latexmk -pdf main.tex          # full build with bibtex
# or for incremental:
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

Output: `main.pdf`.

If you don't have `latexmk`, install via TeX Live (`tlmgr install latexmk`) or
use any IEEE-compatible LaTeX environment (Overleaf works out of the box —
upload the `paper/` directory as a project; pick `main.tex` as the main file).

## Layout

```
paper/
├── main.tex              # document setup, \input{} the sections
├── refs.bib              # all citations
├── sections/
│   ├── abstract.tex      # full abstract (paragraph)
│   ├── introduction.tex  # full intro with contributions list
│   ├── related.tex       # full related-work section incl. Laya positioning
│   ├── method.tex        # SKELETON: architecture, SIGReg, probe protocol
│   ├── experiments.tex   # SKELETON: setup, hardware, training defaults
│   ├── results.tex       # SKELETON: 7 result subsections (scaling, λ, capacity,
│   │                     #            transfer, per-fold, supervised, headline)
│   ├── discussion.tex    # SKELETON: predictor mechanism, deployment, limitations
│   └── conclusion.tex    # SKELETON: tight closing paragraph
├── figures/
│   ├── lambda_sweep.png       # ablation U-curve (Session 7)
│   └── per_fold_figure.png    # 20-subject LOSO bar chart (Session 12.1)
└── README.md             # this file
```

## Status

| Section | State | Source for content |
|---------|-------|--------------------|
| Abstract | Drafted | PAPER_OUTLINE.md elevator pitch |
| Introduction | Drafted | PAPER_OUTLINE.md §1 |
| Related work | Drafted | PAPER_OUTLINE.md §2 + DECISIONS Laya map |
| Method | Skeleton with TODOs | PAPER_OUTLINE.md §3 |
| Experiments | Skeleton with TODOs | PAPER_OUTLINE.md §4.1 |
| Results | Skeleton with TODOs | PAPER_OUTLINE.md §4.2–4.6 + DECISIONS |
| Discussion | Skeleton with TODOs | PAPER_OUTLINE.md §5 |
| Conclusion | Skeleton with TODOs | PAPER_OUTLINE.md §7 |
| Bibliography | Drafted (15 entries) | from related work + datasets |

Every `\todo{...}` macro in the source is currently visible in red in
the PDF (draft mode). Before submission: set `\draftmodefalse` in `main.tex`
to suppress them.

## Open items before submission

1. **Verify Laya's task identities.** Their Table 1 row "LH vs RH MI" — which
   corpus? If EEGMMIDB, our head-to-head claim is direct; if MOABB-derived,
   add a "comparable task type" caveat. Read Appendix D.1.1 of the Laya PDF.
2. **Pull Laya's Appendix B** for exact hyperparameters (λ value, optimizer,
   batch size, etc.) to fill the method-comparison table.
3. **Add an architecture diagram** (`figures/architecture.pdf`). Either
   hand-drawn or generated from a tikz figure or compiled from a draw.io export.
4. **Fill all `\todo` markers** in the skeleton sections.
5. **Author info, affiliation, funding acknowledgement** — currently
   placeholder.
6. **Format figures to 2-column width** if they're currently single-panel —
   may want subfigures.
