# mast_wrf: the unresolved part of the GEP mast signal, and the WRF–mast residual

`config.py`, `stats.py` and `noise_characterisation.py` come from the separate "SPDE solvers" session
on the user's computer, which wrote them to `D:\NS2D_Python_Improved\spde_measured_noise\` on 2026-10-07.
They are recovered from that session's transcript, with its two later patches applied (numpy-1/2
compatible `trapezoid`, the `CUP_EPS` order, and the WRF cell field in the summary). The paths in
`config.py` point to the user's drives: the GEP 1-min logger on E:, and WRF U10/V10 on F:.

- `stats.py`: one reduction for real and virtual masts, plus the exact MSE split
  `bias² + (s_m − s_o)² + 2 s_m s_o (1 − r)`.
- `noise_characterisation.py`: mast spectra, integral times and increments, and the mast − WRF
  residual. The residual is split, out of sample, into a systematic part and an unpredictable
  part ε, the candidate unresolved term.
- `rectification_budget.py` (new): tests whether the WRF speed bias is produced by the
  unresolved part. Speed is convex in the wind components, so a zero-mean unresolved vector v′
  raises the mean measured speed by about σ²_cross / (2|V|). The script measures v′ at the mast
  (1-min deviations within each hour) and predicts the bias with no tuning. It then re-splits
  the MSE.

The other session also wrote `measured_forcing.py`, which adds Gaussian Ornstein–Uhlenbeck forcing
to ns2d. It is not ported here. Injecting an external Gaussian forcing contradicts the project
axiom, that the noise is the unresolved part of the system and is not injected. It also
contradicts the measured non-Gaussian distributions. The intrinsic alternatives are in `../`.

Run on the user's machine (data are local):

    python noise_characterisation.py
    python rectification_budget.py            # writes results/rectification/{budget.csv,summary.json}

Synthetic smoke test (a perfect resolved vector plus sub-hour fluctuations): the bias is 3.3% of
the MSE for raw WRF. After the out-of-sample rectification it is 0.04%.
