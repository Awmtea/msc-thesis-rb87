# msc-thesis-rb87
Python code accompanying my MSc thesis on a single ⁸⁷Rb hyperfine qubit in an optical tweezer. Evaluates the Breit–Rabi clock transition and error budget analytically, and uses QuTiP to simulate Lindblad dynamics (T₁, T₂, T₂*, Hahn echo). Benchmarked against recent tweezer-array experiments.
| Script              | Thesis figures      | Method                    |
|---------------------|---------------------|---------------------------|
| fig_breit_rabi.py   | Fig. 2.1            | Breit–Rabi formula (exact)|
| fig_clock_shift.py  | Fig. 2.2            | Analytical + quadratic approx. |
| fig_rabi.py         | Figs. 3.1, 3.2      | Closed-form Rabi solution |
| fig_leakage.py      | Fig. 3.3            | First-order perturbation  |
| fig_decoherence.py  | Figs. 4.1–4.3       | QuTiP Lindblad simulation |
| fig_budget.py       | Figs. 5.1–5.3       | Analytical error budget + RB |
