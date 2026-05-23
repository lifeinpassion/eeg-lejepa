# Cover Letter

*Draft — target journal: **Biomedical Signal Processing and Control** (Elsevier;
Q1, free to publish via the subscription route). The same letter adapts with
one-line edits to the on-scope fallback **Cognitive Neurodynamics** (Springer,
Q2) or the safety-net **Journal of Neuroscience Methods** (Elsevier, Q3). The
accuracy/AUC numbers below are final (leakage-free rerun, `encoder_mean`
source); only the `[bracketed]` author/date/repository fields remain to fill.*

---

[Date]

To the Editor-in-Chief
Biomedical Signal Processing and Control
Elsevier

Dear Editor,

Please consider our manuscript, **"Edge-Deployable EEG Foundation Models:
Scaling Laws and Cross-Dataset Transfer for Compact SIGReg-Pretrained
Encoders,"** for publication as an original research article in *Biomedical
Signal Processing and Control*.

Self-supervised EEG foundation models have advanced rapidly, but the leading
systems use 5–10 M+ parameters and multi-GPU training budgets aimed at
cross-dataset generalization. The complementary regime that most laboratories
and on-device applications actually operate in — focused, single-corpus
pretraining for in-distribution tasks under a tight compute budget — remains
largely uncharacterized. Our work addresses that gap directly. We present
EEG-LeJEPA, a 2.86 M-parameter Joint-Embedding Predictive Architecture trained
with Sketched Isotropic Gaussian Regularization, whose entire
pretraining-plus-evaluation pipeline runs on a single consumer GPU in under 15
minutes at a compute cost below one US dollar.

We believe the manuscript is well matched to *Biomedical Signal Processing and
Control*, which publishes the development and rigorous empirical evaluation of
signal-processing and machine-learning methods for biomedical problems,
including EEG and brain–computer interfaces. Beyond reporting a compact model,
the paper contributes findings of methodological interest to that readership:

- A scaling-law characterization across six (data, compute) operating points:
  best-source AUC rises then plateaus near 0.76, and a 2.4× larger model is
  modestly but consistently better at matched data (seed-stable across three
  seeds) — the plateau reflects a capacity limit, not a data limit. An
  isolation run further shows the SIGReg projection count must scale with
  embedding dimension; ignoring this manufactures a spurious "larger model
  regresses" result, a previously unreported practical pitfall when applying
  isotropy regularizers across model widths.
- A SIGReg-weight ablation revealing a sharp optimum and **complete
  representational collapse** when the regularizer is removed, giving a clean
  empirical validation of the anti-collapse mechanism.
- A counterintuitive cross-dataset transfer result: channel-padded transfer
  from a full-montage encoder **outperforms** channel-matched re-pretraining.
- A leakage-clean evaluation protocol: we show that allowing evaluation
  subjects into pretraining inflates predictor-derived feature sources, and
  under a strictly held-out split the simple encoder mean-pool is both the
  strongest and the most trustworthy source.
- A controlled comparison showing the self-supervised, frozen-encoder probe
  **ties** end-to-end supervised training on left-versus-right MI (−0.2 pp,
  n.s.) and **significantly exceeds** it on rest-versus-activity (+8.4 pp,
  p<0.001), with one frozen encoder serving both tasks.

On our two downstream EEGMMIDB tasks the model reaches 68.6% accuracy /
0.751 AUC on rest-versus-activity and 62.1% / 0.690 AUC on left-versus-right
motor imagery (encoder mean-pool features) via linear probe, under strict
leave-one-subject-out evaluation with the 20 evaluation subjects held out of
pretraining (both gains over a random-initialized baseline significant,
q<0.002). All headline
differences are assessed with paired significance tests (exact sign-flip
permutation tests, bootstrap confidence intervals, and Benjamini–Hochberg
correction), and the complete pipeline — code, configurations, trained
checkpoints, and per-fold outputs — is released for full reproducibility at
https://github.com/lifeinpassion/eeg-lejepa.

This manuscript is original, has not been published previously, and is not under
consideration elsewhere. All data are from public repositories (PhysioNet
EEGMMIDB and BCI Competition IV Dataset 2a); no new human-subjects data were
collected. The authors declare no competing interests. [Adjust funding /
acknowledgements as applicable.]

We note for transparency that an independent and contemporaneous study (Laya;
Panchavati et al.) applies the same family of methods to EEG at foundation
scale; our contribution is complementary, addressing the compact, single-corpus,
edge-deployment regime and the scaling, ablation, and transfer questions that
work does not examine.

Thank you for your consideration. We look forward to your response.

Sincerely,

Ping Liang and Anton Louise P. De Ocampo
Department of Electronics Engineering, Batangas State University, Batangas City 4200, Philippines
Corresponding author: Anton Louise P. De Ocampo — antonlouise.deocampo@ieee.org

---

### Suggested reviewers (optional — most journals allow 3–5)

- [Name, affiliation, email] — expertise in EEG self-supervised learning
- [Name, affiliation, email] — expertise in motor-imagery BCI
- [Name, affiliation, email] — expertise in representation learning / JEPA

### Remaining fields to fill before sending

All accuracy/AUC/significance numbers above are final (leakage-free rerun,
`encoder_mean` source; `runs/perfold_probe_clean.csv`,
`runs/significance_tests.csv`). Authors, affiliation, corresponding email, and
the repository URL are filled. Only these remain:

| Field | Note |
|-------|------|
| [Date] | set on the day you submit |
| Suggested reviewers | optional; BSPC allows recommending reviewers |
