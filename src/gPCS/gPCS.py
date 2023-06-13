# %%
import numpy as np
import healpy as hp
from astropy.io import fits
from numba import jit, njit, prange, vectorize
import numba
import pathlib

basepath = pathlib.Path(__file__).parent.resolve()

NSIDE = 512
TS_map_Fermi = np.load(f"{basepath}/data/TS_map_fermi.npz")["TS_map"]
TS_map_4FGL = np.load(f"{basepath}/data/TS_map_4FGL.npz")["TS_map"]
gll_psc_v30_path = f"{basepath}/data/fits/gll_psc_v30.fit"

TS_star_alpha_001 = np.load(f"{basepath}/data/TS_star_alpha_0.01.npz")["TS_star"]
TS_star_alpha_005 = np.load(f"{basepath}/data/TS_star_alpha_0.05.npz")["TS_star"]
TS_star_alpha_01 = np.load(f"{basepath}/data/TS_star_alpha_0.1.npz")["TS_star"]


def get_TS_ranking(TS_map, deg):
    """
    Returns the TS ranking of the pixels in the given TS_map, with a cut in galactic latitude of deg degrees.
    """
    filter = hp.query_strip(
        NSIDE, np.deg2rad(90 - deg), np.deg2rad(90 + deg), inclusive=True
    )
    galplane_mask = np.ones_like(TS_map, dtype=bool)
    galplane_mask[filter] = False

    pixels = np.arange(len(TS_map))
    pixels = pixels[galplane_mask]

    TS_sort = np.argsort(TS_map[galplane_mask])[::-1]
    TS_ranking = TS_map[galplane_mask][TS_sort]
    return TS_ranking, pixels[TS_sort]


TS_ranking_fermi, pixel_ranking_fermi = get_TS_ranking(TS_map_Fermi, 30)
TS_ranking_4FGL, pixel_ranking_4FGL = get_TS_ranking(TS_map_4FGL, 30)


@njit
def get_CDF(x_):
    """
    Returns the CDF of the input array x_ and the sorted array x.
    """
    x = np.sort(x_)
    lenx = len(x)
    y = np.arange(1, lenx + 1) / lenx
    return x, y


@njit(parallel=True)
def get_CDFs(x_, batches=100, batch_size=3000):
    nx = len(x_)
    assert (
        nx > batch_size
    ), "the number of values of `x_` must be greather than `batch_size`"
    x_matrix = np.zeros((batches, batch_size))  # matrix of TS
    y_arr = np.arange(1, batch_size + 1) / batch_size
    for i in np.arange(batches):
        idx_ = np.arange(nx)
        np.random.shuffle(idx_)
        x_matrix[i, :] = np.sort(x_[idx_[0:batch_size]])

    return x_matrix, y_arr


def _get_QF_from_TS(TS, x_TS_CDF, y_TS_CDF):
    idx = np.searchsorted(x_TS_CDF, TS)
    idx = np.clip(idx, 0, len(y_TS_CDF) - 1)
    return y_TS_CDF[idx]


def get_QF_from_TS(TS, alpha):
    """
    Returns the QF of the input TS and alpha, where TS can be an array.

    Parameters
    ----------
    TS : float or array
        Test statistic value. If array, it must be 1D.
    alpha : float
        Significance level. Must be 0.01, 0.05 or 0.1.

    Returns
    -------
    QF : float or array
        Quality factor value. If TS is an array, QF will be an array.

    Examples
    --------
    >>> from gPCS import gPCS
    >>> gPCS.get_QF_from_TS(36, 0.05)

    """
    if alpha == 0.01:
        TS_list = TS_star_alpha_001
    elif alpha == 0.05:
        TS_list = TS_star_alpha_005
    elif alpha == 0.1:
        TS_list = TS_star_alpha_01
    else:
        raise ValueError("alpha must be 0.01, 0.05 or 0.1")
    x, y = get_CDF(TS_list)

    return _get_QF_from_TS(TS, x, y)


def _get_QF_array(TS_matrix, TS_star):
    """
    Returns the QF array of the input TS_matrix and TS_star.
    """
    TS_star = np.atleast_1d(TS_star)
    batches, batch_size = TS_matrix.shape
    CDF = np.arange(1, batch_size + 1) / batch_size
    QF_arr = np.zeros((len(TS_star), batches))
    for i in np.arange(batches):
        QF_arr[:, i] = _get_QF_from_TS(TS_star, TS_matrix[i, :], CDF)
    return QF_arr


def _get_QF_ranges_from_TS(TS, TS_matrix):
    QF_arr = _get_QF_array(TS_matrix, TS)

    mean_arr = np.mean(QF_arr, axis=-1)
    std_arr = np.std(QF_arr, axis=-1)
    return mean_arr, std_arr


def get_QF_ranges_from_TS(TS, alpha, batches=100, batch_size=3000):
    """
    Returns the quality factor mean and std for the input values of TS and alpha.

    Parameters
    ----------
    TS : float or array
        Test statistic value. If array, it must be 1D.
    alpha : float
        Significance level. Must be 0.01, 0.05 or 0.1.
    batches : int, optional
        Number of batches to use when costructing the QF distribution. The default is 100.
    batch_size : int, optional
        Number of values to use in each batch. Must be < 4900, the default is 3000.

    Returns
    -------
    mean_arr : float or array
        Mean of the QF distribution. If TS is an array, mean_arr will be an array.
    std_arr : float or array
        Standard deviation of the QF distribution. If TS is an array, std_arr will be an array.

    Examples
    --------
    >>> from gPCS import gPCS
    >>> gPCS.get_QF_ranges_from_TS(36, 0.05)


    """
    if alpha == 0.01:
        TS_list = TS_star_alpha_001
    elif alpha == 0.05:
        TS_list = TS_star_alpha_005
    elif alpha == 0.1:
        TS_list = TS_star_alpha_01
    else:
        raise ValueError("alpha must be 0.01, 0.05 or 0.1")

    assert batch_size < 4900, "batch_size must be < 4900, optimal value is ~3000"

    TS_matrix, _ = get_CDFs(TS_list, batches=batches, batch_size=batch_size)
    return _get_QF_ranges_from_TS(TS, TS_matrix)


@njit
def _get_TS_from_QF(QF, x_TS_CDF, y_TS_CDF):
    idx = np.searchsorted(y_TS_CDF, QF, side="left")
    return x_TS_CDF[idx]


@njit
def get_TS_from_QF(QF, alpha):
    """
    Returns the TS corresponding to the input QF and alpha.

    Parameters
    ----------
    QF : float or array
        Quality factor value. If array, it must be 1D.
    alpha : float
        Significance level. Must be 0.01, 0.05 or 0.1.

    Returns
    -------
    TS : float or array
        Test statistic value. If QF is an array, TS will be an array.

    Examples
    --------
    >>> from gPCS import gPCS
    >>> gPCS.get_TS_from_QF(0.50, 0.05)

    """
    if alpha == 0.01:
        TS_list = TS_star_alpha_001
    elif alpha == 0.05:
        TS_list = TS_star_alpha_005
    elif alpha == 0.1:
        TS_list = TS_star_alpha_01
    else:
        raise ValueError("alpha must be 0.01, 0.05 or 0.1")

    x_CDF, y_CDF = get_CDF(TS_list)
    return _get_TS_from_QF(QF, x_CDF, y_CDF)


def get_4FGL_source_pixels(Smin=None):
    """
    Returns the pixels of the 4FGL catalog with fluxes greater than Smin.
    """
    deg = 30
    # read and process 4fgl catalog
    with fits.open(gll_psc_v30_path, mode="readonly") as hdul:
        glat_ = hdul[1].data["GLAT"]  # type: ignore
        glon_ = hdul[1].data["GLON"]  # type: ignore
        flux_band_4_5 = hdul[1].data["Flux_Band"][:, 3:5]  # type: ignore
    S_i_ = np.sum(flux_band_4_5, axis=0)[0]
    if Smin is None:
        St = 0.0
    else:
        St = Smin
    filter = np.logical_and(np.abs(glat_) >= deg, S_i_ >= St)
    glon = glon_[filter]
    glat = glat_[filter]

    pixels = [
        hp.ang2pix(NSIDE, np.radians(90 - lat), np.radians(lon))
        for lat, lon in zip(glat, glon)
    ]

    return np.unique(pixels)


def get_firing_pixels(TS_lim, filter=False, conservative=False, deg=None):
    """
    For a given TS_lim, return the pixels of the Fermi map with TS>=TS_lim.

    Parameters
    ----------
    TS_lim : float
        The TS threshold. The routine will return only pixels with TS>=TS_lim.
    filter : bool
        If True, the routine will return only pixels that are not in the 4FGL catalog simulation K (for TS>=TS_lim).
        Else it will return all the pixels in the Fermi map with TS>=TS_lim.
    conservative : bool
        If True, after filtering the pixels in K, the routine will also filter the pixels in the
        1 pixel neighborhood of the pixels in K.
    deg : float
        If not None, the routine will also filter the pixels in the disc of radius deg centered on the centroid of the 4FGL catalog sources.

    Returns
    -------
    firing_pixels : array
        The pixels of the Fermi map with TS>=TS_lim, potentially filtered.

    Examples
    --------
    >>> from gPCS import gPCS
    >>> TS_star = 36
    >>> gPCS.get_firing_pixels(TS_star)

    """
    filter_fermi = TS_ranking_fermi >= TS_lim
    filter_4FGL = TS_ranking_4FGL >= TS_lim

    pix_fermi = pixel_ranking_fermi[filter_fermi]
    if not filter:
        return pix_fermi

    pix_4FGL = pixel_ranking_4FGL[filter_4FGL]

    # compute complement of the intersection
    if conservative:
        pixlist = np.empty(
            0,
        )
        for pix in pix_4FGL:
            pixlist = np.append(pixlist, hp.get_all_neighbours(NSIDE, pix))
        if deg is not None:
            for pix in get_4FGL_source_pixels():
                pixlist = np.append(
                    pixlist,
                    hp.query_disc(NSIDE, hp.pix2vec(NSIDE, pix), np.deg2rad(deg)),
                )

        pixlist = np.sort(pixlist)
        pixlist = np.unique(pixlist)
        comp_inters = np.setdiff1d(pix_fermi, pixlist)
    else:
        comp_inters = np.setdiff1d(pix_fermi, pix_4FGL)
    return comp_inters


def export_fits_table(filename, QF, alpha, overwrite=False, **kwargs):
    """
    Export a FITS table with the firing pixels, given QF and alpha.

    Parameters
    ----------
    filename : str
        Name of the output file.
    QF : float
        Quality factor. Must be between 0 and 1. The FITS table will contain the firing pixels with QF_i >= QF.
    alpha : float
        Significance level. Must be 0.01, 0.05 or 0.1.
    overwrite : bool
        If True, overwrite the output file if it already exists.
    **kwargs : dict
        Additional arguments to pass to get_firing_pixels. In particular, filtering options can be specified here.

    Returns
    -------
    filename : str
        Name of the output file.

    Notes
    -----
    The FITS table will contain the following columns:
    - pixel : pixel index
    - TS : TS value
    - QF_best : QF value obtained by considering all the simulations
    - QF_min : lower bound of the QF range
    - QF_max : upper bound of the QF range

    Examples
    --------
    >>> from gPCS import gPCS
    >>> gPCS.export_fits_table("test.fits", QF=0.5, alpha=[0.01, 0.05, 0.1], overwrite=True)
    >>> # read the fits file
    >>> with fits.open("fiting_pixels.fits") as f:
            print(f.info())
            data_001 = f[1].data
            data_005 = f[2].data
            data_01  = f[3].data

    >>> print("data for alpha=0.01")
    >>> print(data_001)
    """
    alpha_arr = np.atleast_1d(alpha)

    hdus = [fits.PrimaryHDU()]
    for a in alpha_arr:
        TS_star = get_TS_from_QF(QF, alpha=a)
        firing_pixels = get_firing_pixels(TS_star, **kwargs)
        TS_ranking = TS_map_Fermi[firing_pixels]
        QF_best = get_QF_from_TS(TS_ranking, alpha=a)
        mean_QF, std_QF = get_QF_ranges_from_TS(
            TS_ranking, alpha=a, batches=100, batch_size=3000
        )
        QF_min = mean_QF - std_QF
        QF_max = mean_QF + std_QF

        c1 = fits.Column(name="pixel", array=firing_pixels, format="K")
        c2 = fits.Column(name="TS", array=TS_ranking, format="E")
        c3 = fits.Column(name="QF_best", array=QF_best, format="E")
        c4 = fits.Column(name="QF_min", array=QF_min, format="E")
        c5 = fits.Column(name="QF_max", array=QF_max, format="E")

        cols = fits.ColDefs([c1, c2, c3, c4, c5])
        hdu = fits.BinTableHDU.from_columns(cols, name=f"alpha={a}")
        hdus.append(hdu)  # type: ignore

    hdul = fits.HDUList(hdus)
    hdul.writeto(filename, overwrite=overwrite)
    return filename

