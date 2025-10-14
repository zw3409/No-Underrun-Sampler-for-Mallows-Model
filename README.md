# Mallows NURS

Implementation of the **No-Underrun Sampler (NURS)** for Mallows permutation models, along with a Hit-and-Run (HR) baseline for numerical comparison.

## Structure


- `nurs.py` – Main sampling algorithm with doubling strategy
- `distance.py` – Distance metric functions (L1, L2, Hamming, Ulam, inversion)
- `hit_run.py` – Hit-and-Run sampler for L2 distance
- `tests` –  Contraction property analysis,  Effective Sample Size measurements,  Visualization of sampling behavior

## License
MIT
