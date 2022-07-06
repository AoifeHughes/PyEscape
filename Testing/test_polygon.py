from PyEscape.escape_polygonhelper import make_hull_and_scale
import numpy as np
import pytest


def test_make_hull_and_scale():
    pts = np.random.random((10, 3))
    hull, pts = make_hull_and_scale(pts, V_t=1)
    assert np.around(hull.volume, 2) == 1
