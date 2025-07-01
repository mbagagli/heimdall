# HEIMDALL

![heim_logo](logo/heim_logo_box_red.png)
version: _0.3.0_
authors: _Matteo Bagagli_, _Francesco Grigoli_, _Davide Bacciu_

--------------------------------------------------------------

This is the official repository of **HEIMDALL**:
a grap**H**-based s**EI**s**M**ic **D**etector **A**nd **L**ocator for seismicity.

This package implements a novel _Spatio-Temporal Graph Neural Network_
 that performs seismic phase picking, phase association,
 and event location in a single pass.
 By eliminating the need for separate sequential algorithms,
 it significantly reduces the time required for hyperparameter
 tuning and transfer learning, resulting in a smoother and more
 consistent user experience—especially for site-specific analyses.

## Install

We offer an env.yml file for the installation (it may take a while).
We recommend using `conda` for the installation:

```bash
git clone PROJECTHTML
cd HEIMDALL
conda env create  -f env.yml
conda activate heimdall
pip install .  # mind the dot
```

A proper packaging hosted on conda channels and PyPI repositories will
be delivered as soon as ready.

## Play

In the `bin` folder are stored all the executables that, once installed,
will be available in your Python path, so can be run directly on shell

Some IPython notebook will be distributed as well to illustrate the workflow

Documentation is already in place, and a read-the-docs will be deployed soon.

