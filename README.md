# gPCS: Gamma-ray Photon-Counts Statistics
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8070852.svg)](https://doi.org/10.5281/zenodo.8070852)
[![Downloads](https://pepy.tech/badge/gPCS)](https://pepy.tech/project/gPCS)
[![](https://img.shields.io/pypi/v/gPCS.svg?maxAge=3600)](https://pypi.org/project/gPCS)

This repository contains the code for the paper [Deepening gamma-ray point-source catalogues with sub-threshold information](https://arxiv.org/abs/2306.16483). 

We provide our results in the form of a [precomputed FITS](examples/firing_pixels.fits), as well as python package which can be used to read the data as `numpy` arrays, as well as export a similar FITS table. 

# Installation
This package can easily be installed through pip:
```python
pip install gPCS
```

# Example usage
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurelio-amerio/gPCS/blob/main/examples/analysis.ipynb) 

This package can be used to obtain the firing pixels, either given a chosen TS_star or by fixing a quality factor (QF) and significance level (alpha). 

```python
import numpy as np
from gPCS import gPCS

# specify manually a TS_star
TS_star = 36
pixel_firing = gPCS.get_firing_pixels(TS_star, filter=False)
print(len(pixel_firing))

# Compute the TS_star from a chosen QF and alpha
QF = 0.5
alpha = 0.05
TS_star = gPCS.get_TS_from_QF(QF, alpha=alpha)
pixel_firing = gPCS.get_firing_pixels(TS_star, filter=False)
print(len(pixel_firing))
```

We can get the TS of the firing pixels:
```python
TS_star=36
firing_pixels = gPCS.get_firing_pixels(TS_star, filter=False) 
TS_ranking = gPCS.TS_map_Fermi[firing_pixels]
```
It is easy to obtain the galactic coordinates of the firing pixels using healpy:
```python
import healpy as hp
lon, lat = hp.pix2ang(NSIDE, firing_pixels, lonlat=True) # lon lat in degrees
```

And we can compute the QF and QF range of the firing pixels, given alpha:
```python
# obtain the QF using all the simulations
QF = gPCS.get_QF_from_TS(TS_ranking, alpha=alpha)

# compute the mean and std of the QF using batches of simulations
mean_QF, std_QF = gPCS.get_QF_ranges_from_TS(TS_ranking, alpha=alpha, 
                            batches=100, batch_size=3000)

# we can obtain the QF range for the firing pixels using the mean and std
QF_min = mean_QF - std_QF
QF_max = mean_QF + std_QF
```

If desired, we provide some simple functions to filter the firing pixels, for example by removing pixels that are firing for the simulated 4FGL $\mathcal{K}$ map (see paper):

```python
# filter out the firing pixels for the 4FGL K map
pixel_firing = gPCS.get_firing_pixels(TS_star, filter=True)
# If conservative is True, after filtering the pixels in K, 
# the routine will also filter the pixels in the 1 pixel neighborhood of the pixels in K.
pixel_firing = gPCS.get_firing_pixels(TS_star, filter=True, conservative=True)
# If deg is specified, the routine will also filter the pixels 
# in the disc of radius deg centered on the centroid of the 4FGL catalog sources.
pixel_firing = gPCS.get_firing_pixels(TS_star, filter=True, conservative=True, deg=0.5)
```

For ease of use, we provide the TS maps both for Fermi and the simualted 4FGL map ($\mathcal{K}$) as `numpy` attays. The maps are computed at `nside=512` and are stored in the following variables:
```python
gPCS.TS_map_Fermi
gPCS.TS_map_4FGL
```

In order to compute the 4FGL $\mathcal{K}$ map, we use employ the `gll_psc_v30` catalog, available from the Fermi LAT colalboration [website](https://fermi.gsfc.nasa.gov/ssc/data/access/lat/12yr_catalog/).

# Exporting the results
We provide a simple function to export the results in the form of a FITS table, which can be used to reproduce the results of the paper. 

```python
export_fits_table(filename, QF, alpha, overwrite=False, **kwargs)
```
`export_fits_table` accepts the following arguments:
- filename : name of the output FITS file
- QF : quality factor
- alpha : significance level (can be an array, and the supported values are 0.01, 0.05, 0.1)
- overwrite : if True, the routine will overwrite the output file if it already exists
- **kwargs : additional arguments to be passed to `get_firing_pixels`.

`export_fits_table` will create a FITS table with the following columns:
- pixel : pixel index
- TS : TS value
- QF_best : QF value obtained by considering all the simulations
- QF_min : lower bound of the QF range
- QF_max : upper bound of the QF range

In order to export the FITS table available in the examples folder, we can run the command:
```python
gPCS.export_fits_table(filename="firing_pixels.fits", QF=0.50, alpha=[0.01, 0.05, 0.1])
```
# List of functions
- `get_QF_from_TS(TS, alpha)`: computes the quality factor from a given TS and alpha

- `get_QF_ranges_from_TS(TS, alpha, batches=100, batch_size=3000)`: computes the mean and std of the QF from a given TS and alpha, using batches of simulations

- `get_TS_from_QF(QF, alpha)`: computes the TS from a given QF and alpha

- `get_firing_pixels(TS_lim, filter=False, conservative=False, deg=None)`: computes the firing pixels for a given TS_lim.

- `export_fits_table(filename, QF, alpha, overwrite=False, **kwargs)`: exports the results in the form of a FITS table.

For more information about what each function does, please refer to the [docstrings](src/gPCS/gPCS.py) and help of each function.

# License
This code is released under the Zlib license. See the [LICENSE](LICENSE) file for more information.
