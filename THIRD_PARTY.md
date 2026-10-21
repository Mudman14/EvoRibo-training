# Third-Party Boundaries

The project-owned code is distributed under `LICENSE` (MIT). This does not
relicense PyTorch, NumPy, matplotlib, adabmDCA, or any upstream model/data.
Dependencies are installed separately, not vendored.

The optional native inverse runner uses the low-level Python API from
[adabmDCApy](https://github.com/spqb/adabmDCApy), whose original license and
attribution continue to apply. Its optional requirements pin the version tested
here; do not assume compatibility with arbitrary future low-level APIs.
The clean-room `mi3_potts_inverse.py` is project code, not mi3-GPU source.

RESM feature extraction, upstream pretrained weights, SPOT-RNA BPP generation,
and original training data are not shipped. Obtain them separately and follow
their respective terms. The project MIT license does not grant rights to those
weights or datasets.
