# connPFM

[![Latest Version](https://img.shields.io/pypi/v/connPFM.svg)](https://pypi.python.org/pypi/connPFM/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/connPFM.svg)](https://pypi.python.org/pypi/connPFM/)
[![DOI](https://zenodo.org/badge/111111.svg)](https://zenodo.org/badge/latestdoi/111111)
[![License](https://img.shields.io/badge/License-LGPL%202.1-blue.svg)](https://opensource.org/licenses/LGPL-2.1)
[![CircleCI](https://circleci.com/gh/SPiN-Lab/connPFM.svg?style=shield)](https://circleci.com/gh/SPiN-Lab/connPFM)
[![Documentation Status](https://readthedocs.org/projects/connPFM/badge/?version=latest)](http://connPFM.readthedocs.io/en/latest/?badge=latest)
[![Codecov](https://codecov.io/gh/SPiN-Lab/connPFM/branch/main/graph/badge.svg)](https://codecov.io/gh/SPiN-Lab/connPFM)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![DOI](https://zenodo.org/badge/413858406.svg)](https://zenodo.org/badge/latestdoi/413858406)

## Methodology

The connPFM method comprises of 3 steps (Fig1): 
1. Deconvolution with stability-selection paradigm free mapping that computes the probability (area under the curve; AUC) that a neuronal-related event generates a BOLD event at each time; 
2. Selection of significant CF events based on the pairwise CF matrix computed from the AUC timecourses (CF-AUC); this selection step can be done in 2 ways: 
    Temporal: Thresholding of the root sum of squares (RSS) time-series calculated from the CF-AUC matrix, where the threshold is based on a null distribution of the RSS computed by repeating the same procedure with 100 phase-randomized surrogate datasets, and selecting all the events occurring at those exceeding times; 
    Spatio-temporal selection: Thresholding of the CF-AUC matrix based on the null distribution of all the elements of the 100 surrogate CF-AUC matrices (e.g., p<0.05).
3. Debiasing of the neuronal related activity associated with the selected deconvolved events, which by definition show a significant CF with any other region, through ordinary least-squares regression. 

![connPFM flowchart](https://github.com/vinferrer/connPFM/tree/readme/docs/connPFM_flowchart.jpg?raw=true)

## Installation
```
    git clone https://github.com/SPiN-Lab/connPFM.git
    cd connPFM
    pip3 install -e .[all]
```
