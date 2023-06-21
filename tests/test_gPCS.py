#%%
import numpy as np
import sys
sys.path.append("src")
from gPCS import gPCS
import matplotlib.pyplot as plt
from astropy.io import fits
import os
basepath = os.path.dirname(__file__)
#%%

def test_get_TS_ranking():
    TS_ranking_fermi, _ = gPCS.get_TS_ranking(gPCS.TS_map_Fermi, 30)
    TS_ranking_fermi_ref = np.load(f"{basepath}/files/TS_ranking_fermi.npz")["TS_ranking_fermi"]
    assert np.allclose(TS_ranking_fermi, TS_ranking_fermi_ref, rtol=1e-1)
    return

def test_get_CDF():
    x = np.array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
    y = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6 ,0.7, 0.8, 0.9, 1.0])
    x_cdf, y_cdf = gPCS.get_CDF(np.arange(1,11))
    assert np.allclose(x_cdf, x) and np.allclose(y_cdf, y)
    return

def test_get_QF_from_TS():
    QF_ref = 0.47013565499088883
    QF = gPCS.get_QF_from_TS(36, alpha=0.05)
    assert np.allclose(QF, QF_ref)
    return

def test_get_QF_ranges_from_TS():
    m_ref = 0.47
    s_ref = 0.0086
    m, s = gPCS.get_QF_ranges_from_TS(36, alpha=0.05, batches=1_000, batch_size=2000)
    assert np.allclose(m, m_ref, rtol=1e-1) and np.allclose(s, s_ref, rtol=1e-1)
    return

def test_get_TS_from_QF():
    TS_ref = 36.56653911691979
    TS = gPCS.get_TS_from_QF(0.5, alpha=0.05)
    assert np.allclose(TS, TS_ref)
    return

def test_get_4FGL_source_pixels():
    pix_4FGL_ref = np.sort(np.load(f"{basepath}/files/pix_4FGL.npz")["pix_4FGL"])
    pix_4FGL = np.sort(gPCS.get_4FGL_source_pixels())
    assert np.allclose(pix_4FGL, pix_4FGL_ref)
    return

def test_get_firing_pixels():
    data = np.load(f"{basepath}/files/pixel_firing.npz")

    pix_firing_ref = np.sort(data["pixel_firing"])

    pix_firing  = np.sort(gPCS.get_firing_pixels(36, filter=False))

    assert np.allclose(pix_firing, pix_firing_ref) 

    return

