"""Build Report_Lab2.docx / Report_Lab2.pdf from the course template.

Usage:  py -3.14 -X utf8 scripts/make_report.py [path/to/DSP_TemplateReport.docx]
The numbers quoted here are the results printed by the executed notebook Lab2_Windowing.ipynb;
the figures are taken from figures/.
"""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from report_builder import Report  # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
TEMPLATE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "DSP_TemplateReport.docx"
REPO = "https://github.com/Slayzerus/dsp-lab2"

rep = Report(TEMPLATE, lab=2, date=date.today().strftime("%d.%m.%Y"),
             topic="Windowing: influence of window functions on the detection and separation of close sinusoids",
             variant="wszystkie (1–15)")

# 1 ---------------------------------------------------------------------------------------------------------------
rep.heading("1. Problem statement")
rep.para("Generate a discrete-time signal consisting of three sinusoidal components with close frequencies "
         "f₁, f₂, f₃ sampled with the sampling frequency fₛ:")
rep.para("x[k] = A₁·sin(2π·f₁/fₛ·k) + A₂·sin(2π·f₂/fₛ·k) + "
         "A₃·sin(2π·f₃/fₛ·k),   0 ≤ k < N,", align=WD_ALIGN_PARAGRAPH.CENTER)
rep.para("with A₁ = |x[k]|max, A₂ = 0.1·A₁ (−20 dB) and A₃ = 0.01·A₁ (−40 dB). "
         "Apply several window functions before computing the DFT and plot (1) the normalized DFT level of the "
         "windowed signals (like Fig. 6 of the instructions) and (2) the DTFT of the windows normalized to their "
         "main-lobe maximum (like Fig. 7). Interpret which window gives the best detection of weak tones close to a "
         "strong tone, which gives the best separation of close frequencies, how the sidelobe level and the "
         "main-lobe width influence the spectra, and why f₁, f₂ and f₃ may look different despite their "
         "small spacing. **All 15 variants of Table 1 were computed and compared.**")

# 2 ---------------------------------------------------------------------------------------------------------------
rep.heading("2. Input data")
T1 = [(1, 300, 300.25, 299.75, 2, 400, 2000, "C1"), (2, 400, 400.25, 399.75, 2, 600, 3000, "C2"),
      (3, 500, 500.25, 499.75, 2, 800, 1800, "C3"), (4, 600, 600.25, 599.75, 2, 500, 2000, "C4"),
      (5, 300, 300.25, 299.75, 2, 400, 2000, "C1"), (6, 600, 600.25, 599.75, 3, 800, 2000, "C5"),
      (7, 400, 400.25, 399.75, 3, 600, 3000, "C2"), (8, 500, 500.25, 499.75, 3, 800, 1800, "C3"),
      (9, 600, 600.25, 599.75, 3, 500, 2000, "C4"), (10, 300, 300.25, 299.75, 3, 400, 2000, "C1"),
      (11, 200, 200.25, 199.75, 4, 400, 2000, "C6"), (12, 400, 400.25, 399.75, 4, 600, 3000, "C2"),
      (13, 500, 500.25, 499.75, 4, 800, 1800, "C3"), (14, 600, 600.25, 599.75, 4, 500, 2000, "C4"),
      (15, 500, 500.25, 499.75, 4, 800, 2000, "C7")]
rep.table(["No", "f₁ [Hz]", "f₂ [Hz]", "f₃ [Hz]", "|x|max = A₁", "fₛ [Hz]", "N",
           "configuration"], T1, caption="Variants (Table 1 of the instructions) and the configuration they belong to",
          size=9)
rep.para("Variants with the same (f₁, f₂, f₃, fₛ, N) give identical spectra in dB re A₁ (A₁ only "
         "shifts the absolute level), so the 15 variants form **7 configurations C1–C7**. Every variant violates "
         "the sampling theorem (f > fₛ/2), so the tones appear at aliased frequencies; the decisive parameter is "
         "the observation time T = N/fₛ, which sets the tone spacing of 0.25 Hz to 0.5625–1.25 DFT bins.")
CFG = [("C1", "1, 5, 10", 5.0, 0.2, "100.00 / 99.75 / 100.25", "1500 / 1501.25 / 1498.75", 1.25),
       ("C2", "2, 7, 12", 5.0, 0.2, "200.00 / 199.75 / 200.25", "2000 / 2001.25 / 1998.75", 1.25),
       ("C3", "3, 8, 13", 2.25, 0.444, "300.00 / 299.75 / 300.25", "1125 / 1125.56 / 1124.44", 0.5625),
       ("C4", "4, 9, 14", 4.0, 0.25, "100.00 / 100.25 / 99.75", "2400 / 2401 / 2399", 1.0),
       ("C5", "6", 2.5, 0.4, "200.00 / 199.75 / 200.25", "1500 / 1500.62 / 1499.38", 0.625),
       ("C6", "11", 5.0, 0.2, "200.00 / 199.75 / 199.75", "1000 / 1001.25 / 998.75", 1.25),
       ("C7", "15", 2.5, 0.4, "300.00 / 299.75 / 300.25", "1250 / 1250.62 / 1249.38", 0.625)]
rep.table(["config.", "variants", "T [s]", "Δf [Hz]", "apparent f₁ / f₂ / f₃ [Hz]",
           "periods in N (f₁ / f₂ / f₃)", "spacing [bins]"], CFG,
          caption="Derived quantities of the configurations", size=8.5)
rep.para("Variant 11 (C6) is degenerate: f₁ = fₛ/2 is sampled at its zero crossings (A₁·sin(πk) = 0) and "
         "f₂ aliases onto f₃, so the samples contain a single sine of amplitude 0.09·A₁. "
         "Windows compared (periodic, `sym=False`): rectangular, Hann, Hamming, Blackman (named in the task), "
         "Blackman–Harris (named in Sec. 1 of the instructions) and flat-top (used in the instructions' code).")
rep.para("Environment: Python 3.14.2, NumPy 2.4.1, SciPy 1.18.1, Matplotlib 3.10.8, pandas 3.0.6, Jupyter.", size=11)

# 3 ---------------------------------------------------------------------------------------------------------------
rep.heading("3. Commands used")
rep.subheading("a) Source code (condensed excerpt of the notebook Lab2_Windowing.ipynb)")
rep.code('''
WINDOWS = {"Rectangular": lambda N: np.ones(N),
           "Hann": lambda N: sw.hann(N, sym=False),
           "Hamming": lambda N: sw.hamming(N, sym=False),
           "Blackman": lambda N: sw.blackman(N, sym=False),
           "Blackman-Harris": lambda N: sw.blackmanharris(N, sym=False),
           "Flat-top": lambda N: sw.flattop(N, sym=False)}

def components(v, N=None):                 # the three tones of variant v
    k = np.arange(N or int(v.N))
    A = np.array([1.0, 0.1, 0.01]) * v.Amax                  # A1, A2 = 0.1 A1, A3 = 0.01 A1
    f = np.array([v.f1, v.f2, v.f3])
    return A, f, A[:, None] * np.sin(2*np.pi*f[:, None]/v.fs*k)

def fft2db(X):                             # Sec. 5.1.1: level normalised to sine amplitudes
    N = X.size
    Xtmp = 2/N * X
    Xtmp[0] *= 1/2                         # 0 Hz bin exists only once
    if N % 2 == 0:
        Xtmp[N//2] /= 2                    # fs/2 bin exists only once
    return 20*np.log10(np.maximum(np.abs(Xtmp), 1e-15))

def winDTFTdB(w):                          # Sec. 5.2.1: quasi-DTFT by 100x zero-padding
    Nz = 100 * w.size
    W = np.abs(fftshift(fft(w, Nz)))
    return 2*np.pi/Nz*np.arange(Nz) - np.pi, 20*np.log10(W / W.max())

def level_re_A1(x, w, A1):                 # Fig. 6-type level in dB re A1, CG-corrected
    return fft2db(fft(x * w)) - 20*np.log10(w.mean()) - 20*np.log10(A1)
''')
rep.para("Three metrics compare the windows objectively for each weak tone: **M1 – DFT visibility** (the tone "
         "dominates one of its two neighbouring DFT bins by ≥ 6 dB; what the requested plot shows), "
         "**M2 – grid-independent detectability** (the tone exceeds the worst-case leakage of the other tones "
         "within ±0.5 bin, read from the window DTFT) and **M3 – separation** (the zero-padded spectrum shows the "
         "tone as its own peak for all 16 relative phases).")
rep.subheading("b) Screenshots")
rep.figure(FIG / "fig01_fig6_reproduction.png",
           "Validation: the helper functions reproduce Fig. 6 of the instructions (f = 200 / 200.25 Hz, "
           "fₛ = 800 Hz, N = 1600): best-case peaks 0 / −6.02 / −13.33 dB, worst case −3.92 dB (rect.)",
           12.5)
rep.figure(FIG / "fig04_windows_dtft_zoom.png",
           "Window DTFT normalized to the main-lobe maximum (analogue of Fig. 7): (a) main lobes and first "
           "sidelobes, (b) close-up of the region of the weak tones (0.56–1.25 bins) with the −20/−40 dB levels",
           16.0)
WP = [("Rectangular", "0.00", "1.000", "1.21", "1", "−13.3", "3.92"),
      ("Hann", "−6.02", "1.500", "2.00", "2", "−31.5", "1.42"),
      ("Hamming", "−5.35", "1.363", "1.81", "2", "−42.7", "1.75"),
      ("Blackman", "−7.54", "1.727", "2.30", "3", "−58.1", "1.10"),
      ("Blackman–Harris", "−8.90", "2.004", "2.67", "4", "−92.0", "0.83"),
      ("Flat-top", "−13.33", "3.770", "4.58", "5", "−93.0", "0.01")]
rep.table(["window", "coherent gain [dB]", "ENBW [bins]", "6-dB BW [bins]", "main-lobe half-width [bins]",
           "highest sidelobe [dB]", "scalloping loss [dB]"], WP,
          caption="Computed window parameters (agree with Harris 1978)", size=9)
rep.link("Link to the remote repository (GitHub): ", REPO)
rep.para("The repository contains the notebook (Lab2_Windowing.ipynb), its PDF export (Lab2_Windowing.pdf), the "
         "source in plain Python (src/), all figures (figures/) and this report.", size=11)

# 4 ---------------------------------------------------------------------------------------------------------------
rep.heading("4. Outcomes")
rep.para("The notebook contains one Fig. 6-type figure per configuration (all variants). Three representative "
         "cases are shown here: markers = DFT bins (level in dB re A₁ with coherent-gain correction), thin line = "
         "32× zero-padded DFT, vertical lines = apparent f₁, f₂, f₃, horizontal lines = −20 / −40 dB.")
rep.figure(FIG / "fig07_C1.png", "C1 – variants 1, 5, 10 (T = 5 s, spacing 1.25 bins): Hann and Hamming show both "
                                  "weak tones as shoulders on the DFT bins next to the coherent strong line", 16.0)
rep.figure(FIG / "fig10_C4.png", "C4 – variants 4, 9, 14 (T = 4 s, all tones coherent, 1 bin apart): the "
                                  "rectangular window gives three clean lines at 0, −20 and −40 dB", 16.0)
rep.figure(FIG / "fig09_C3.png", "C3 – variants 3, 8, 13 (T = 2.25 s, spacing 0.5625 bins): only the rectangular "
                                  "window shows f₂; f₃ is hidden for every window", 16.0)
rep.figure(FIG / "fig14_heatmap_M1.png", "Metric M1 (visibility on the DFT bins, SLR in dB) for every variant and "
                                          "window; detected if ≥ 6 dB", 14.5)
SUMMARY = [("1, 2, 5, 7, 10, 12 (C1, C2)", "5", "Rect., Hann, Hamming", "Hann, Hamming", "Hamming, Hann"),
           ("4, 9, 14 (C4)", "4", "Rectangular", "Rectangular", "Rectangular"),
           ("3, 8, 13 (C3)", "2.25", "Rectangular", "—", "Rectangular"),
           ("6, 15 (C5, C7)", "2.5", "Rectangular", "—", "Rectangular"),
           ("11 (C6)", "5", "n/a", "n/a", "n/a – f₁ not sampled, f₂ ≡ f₃")]
rep.table(["variants", "T [s]", "f₂ visible in DFT (M1)", "f₃ visible in DFT (M1)", "best window (DFT)"],
          SUMMARY, caption="Summary of the DFT view for all 15 variants", size=9)
rep.para("**Window-intrinsic view (M2, M3):** no window detects or separates f₂ or f₃ in any variant (best M2 "
         "value −9.6 dB, no separation). Every DFT detection above exists only because f₁ is coherently "
         "sampled, and it disappears for a frequency error of 0.1–0.25 bin. A general sweep (strong tone plus a weak "
         "tone at 0 … −80 dB and 0.25 … 40 bins) gives the minimum spacing needed by each window:")
rep.figure(FIG / "fig15_sweep_min_spacing.png", "Minimum spacing Δmin (in DFT bins) at which a weak tone is "
                                                 "separated, as a function of its level, for each window", 13.5)
TMIN = [("Rectangular", "8.625", "∞", "—", "—"), ("Hann", "2.750", "4.250", "17", "6800"),
        ("Hamming", "2.750", "9.625", "38.5", "15 400"), ("Blackman", "3.625", "3.750", "15", "6000"),
        ("Blackman–Harris", "4.250", "4.750", "19", "7600"), ("Flat-top", "6.250", "6.750", "27", "10 800")]
rep.table(["window", "Δmin f₂ (−20 dB) [bins]", "Δmin f₃ (−40 dB) [bins]", "Tmin for both [s]",
           "Nmin at fₛ = 400 Hz"], TMIN,
          caption="Observation time needed to separate both weak tones of the task (0.25 Hz spacing)", size=9)
rep.figure(FIG / "fig16_longer_record.png", "Check with a longer record: C1 with N = 8002 (T ≈ 20 s, spacing 5 bins, "
                                             "f₁ half a bin off): Hann, Blackman and Blackman–Harris separate both "
                                             "weak tones", 13.5)

# 5 ---------------------------------------------------------------------------------------------------------------
rep.heading("5. Conclusions")
rep.para("For the reasons given, we conclude that:", keep_next=True)
rep.bullets([
    "**Best detection of weak tones close to a strong one:** a low-sidelobe window chosen by the amplitude ratio "
    "– Hann/Hamming down to about −30 dB, Blackman for −40 … −50 dB, Blackman–Harris for −60 dB and "
    "below – provided the weak tone lies outside the main lobe of the strong one (here T ≥ 15 s). In the short "
    "records of Table 1 the DFT result depends on coherent sampling: Hamming/Hann for T = 5 s (both weak tones "
    "visible), the rectangular window for T = 4 s (perfect lines) and for T = 2.25–2.5 s (f₂ only).",
    "**Best separation of close frequencies:** the rectangular window (narrowest main lobe, 6-dB bandwidth 1.21 "
    "bins; separates equal tones from 1.6 bins), followed by Hamming and Hann; flat-top is the worst (5.25 bins). "
    "Its advantage vanishes for unequal amplitudes (−20 dB needs 8.6 bins, −40 dB fails).",
    "**Sidelobes vs. main lobe:** the main-lobe width decides how close two tones may be before they merge; the "
    "sidelobe level and decay set the floor that hides weaker tones farther away (−13 dB rectangular vs. "
    "−58 dB Blackman, −92 dB Blackman–Harris). Lower sidelobes always cost a wider main lobe and a lower "
    "coherent gain; a K-term window spreads a coherent tone over 2K − 1 bins.",
    "**Why f₁, f₂, f₃ look different:** f₁ completes an integer number of periods in N and falls on a bin, "
    "while f₂ and f₃ lie between bins (scalloping and leakage); all frequencies are aliased (mirrored aliases "
    "swap f₂ and f₃; in variant 11 f₁ = fₛ/2 vanishes); and the weak tones sit at different heights of the "
    "same leakage skirt.",
    "Separating both weak tones of the task needs a record about 3–7 times longer than in Table 1 (T ≥ 15 s "
    "with Blackman); zero-padding only interpolates the spectrum and does not improve resolution.",
], numbered=True)

rep.save(ROOT / "Report_Lab2.docx", ROOT / "Report_Lab2.pdf")
