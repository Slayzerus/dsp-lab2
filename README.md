# DSP Lab 2 – Windowing in spectral analysis

Digital Signal Processing (prof. dr hab. Vasyl Martsenyuk), UBB — Radosław Skrzypczyński, Informatyka II st., Gr. 2.

**Task:** a signal of three close sinusoids (A₂ = 0.1·A₁, A₃ = 0.01·A₁) is analysed with several window functions
(rectangular, Hann, Hamming, Blackman, Blackman–Harris, flat-top): normalised DFT level of the windowed signals
(like Fig. 6 of the instructions), window DTFTs normalised to the main-lobe maximum (like Fig. 7), and an
interpretation of detectability and separation. All 15 variants of Table 1 are computed and compared.

| File | Content |
|---|---|
| `Lab2_Windowing.ipynb` | executed notebook (validation, variants, windows, spectra of all variants, metrics, interpretation) |
| `Lab2_Windowing.pdf` | PDF export of the notebook |
| `Report_Lab2.pdf` / `.docx` | report based on the course template |
| `src/Lab2_Windowing.py` | the notebook as a plain Python script (jupytext percent format) |
| `figures/` | all figures of the notebook |
| `scripts/` | PDF export of the notebook and report generator |

Reproduce (Python 3.14, numpy, scipy, matplotlib, pandas, jupytext, nbconvert):

```
py -3.14 -m jupytext --to ipynb --output Lab2_Windowing.ipynb src/Lab2_Windowing.py
py -3.14 -m jupyter nbconvert --to notebook --execute --inplace Lab2_Windowing.ipynb
py -3.14 -X utf8 scripts/export_notebook_pdf.py Lab2_Windowing.ipynb Lab2_Windowing.pdf
py -3.14 -X utf8 scripts/make_report.py path/to/DSP_TemplateReport.docx
```
