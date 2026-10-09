# Is it publishable? Honest assessment (2026-10-09)

Short answer: **most of the theory is known.** We re-derived it well, and every claim is checked
on code, but it is not new. **One narrow applied paper looks open**, and it stands or falls on a
test that has to run on the GEP mast and WRF data.

## What is already known (we rediscovered it)

| Our result | Prior art |
|---|---|
| Noise = eliminated degrees of freedom; Green–Kubo amplitude; memory | Mori–Zwanzig (Zwanzig 1973), homogenization (Papanicolaou, Majda–Timofeyev–Vanden-Eijnden 2001, Pavliotis–Stuart 2008), deterministic chaos → noise (Melbourne, Gottwald) |
| L96: a measured, state-dependent, non-Gaussian closure with memory beats Gaussian AR(1) | Wilks 2005; Arnold, Moroz & Palmer 2013; conditional Markov chains, Crommelin & Vanden-Eijnden 2008 ([pdf](https://homepages.cwi.nl/~dtc/pubs/L96_JAS_revised.pdf)); VARX ([Verheul & Crommelin](https://pure.uva.nl/ws/files/65173657/Stochastic_parametrization_with_VARX_processes.pdf)); [Crommelin & Edeling](https://arxiv.org/pdf/2004.01457) |
| Rectification: a model's resolved speed reads low by about σ²/(2V) | Gustiness, S² = V² + U_g² (Jabouille et al. 1996; [notes](https://www.inscc.utah.edu/~krueger/6220/gustiness.pdf)); a sub-grid wind PDF removes a ~1 m/s GCM ocean wind bias ([AGU 2015](https://agu.confex.com/agu/fm15/webprogram/Paper71229.html)) |
| Rice / Weibull from Gaussian wind components | Monahan 2006, 2007, 2018 ([NPG](https://npg.copernicus.org/articles/25/335/2018/)) |
| Gaussian wind vector in post-processing | Bivariate-normal EMOS for wind vectors, Schuhen, Thorarinsdottir & Gneiting 2012 ([arXiv](https://arxiv.org/pdf/1201.2612)) |
| Sub-filter kinetic energy from the resolved field via a test filter | LES similarity / dynamic procedure (Germano; Meneveau & Katz) |
| Tropical-cyclone wind probability at a point from track error | NHC Monte Carlo wind speed probabilities, DeMaria et al. 2009 ([pdf](https://www.Nhc.Noaa.Gov/pdf/2009waf_wsp.pdf)). Our Rice mixture is an analytic-quadrature version of it |
| Saddlepoint, Cornish–Fisher, tail asymptotics | Textbook (Daniels 1954; Lugannani–Rice 1980) |

**A caution on the user's thesis.** WRF 10-m winds over plains and valleys are usually biased
*high*: Jiménez & Dudhia attribute this to unresolved orographic drag
([topo_wind](https://www2.mmm.ucar.edu/wrf/users/physics/phys_refs/SURFACE_LAYER/topo_wind.pdf)).
Rectification can only make a model read *low*. So the missing unresolved part cannot be the
whole WRF bias. It is one signed term in a budget that also has drag, PBL and representativeness
terms.

## What looks open (no prior found in our searches; a check, not a review)

1. **Physics-structured probabilistic correction of a single deterministic WRF run at a mast.**
   The law is Rice(α|V_wrf|, σ), with σ² from the resolved gradients plus a measured floor. It
   needs no ensemble. We found no Rice-based EMOS, and none whose variance is derived from the
   resolved gradients. Standard practice is truncated normal, gamma or log-normal
   ([Baran & Lerch](https://arxiv.org/pdf/1511.02001)).
2. **A model-free budget of WRF − mast error** at GEP Ben Guerir. It splits the error into an
   intrinsic floor (a single-mast Taylor "dual reporter", including rectification) and error in
   the resolved state, with T* (the mast averaging window that minimises the error) as a
   measure of WRF's effective resolution. These are known ingredients, and region-specific data
   for a Moroccan site would be new.
3. (Weaker) **Transport equations for distribution-family parameters** with a closure table: which
   physics keeps Weibull, Weibull 3p, exp-Weibull or Champernowne exact, and how to close mixing.
   We found only static height profiles of k (Kelly et al. 2014, BLM). This could be a short
   methods note (NPG), but the reviewer will say "presumed PDF".

## The paper that is realistically publishable

*"How much of the WRF–mast wind-speed error is unresolved? A parameter-light decomposition and a
Rice-law correction at Ben Guerir"*. Target: Wind Energy Science, J. Appl. Meteor. Climatol.,
or Renewable Energy.

Required, all runnable on the user's machine (`spde/mast_wrf/`):

1. `intrinsic_floor.py`: the floor vs excess of WRF's MSE, and T*.
2. `rectification_budget.py`: how much of the speed bias the unresolved variance explains, out of
   sample, with its sign.
3. `rice_vs_emos.py`: Rice-EMOS and Rice-gradient against the standard truncated-normal EMOS
   (CRPS, PIT, calm-wind and tail scores).

The paper exists only if (3) shows a clear gain, at calm winds or in the tails, or with fewer
parameters, **and** the fitted σ agrees with the variance measured independently in (1). On
synthetic Gaussian data the two tie (CRPS 0.513 vs 0.512), as they should.
4. Controls a reviewer will ask for: stability and time-of-day stratification, nearest vs best
   grid point, sub-grid orography on/off, anemometer height/definition.

## What the rest is good for

The L96, Navier–Stokes, saddlepoint and singularity work is supporting material: methods,
supplement, or a thesis chapter. It is correct and documented, but not the headline. In
particular, the cyclone component reproduces DeMaria et al.'s idea analytically. It would matter
only if tied to real track errors and mast or station observations during actual storms.
