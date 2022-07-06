from itertools import product, combinations
import numpy as np
import plotly.graph_objects as go

from .escape_utility import sphere_vol_to_r, cube_vol_to_r
from .escape_points import random_points_on_ellipsoid, points_on_cuboid

import matplotlib.pyplot as plt


def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.
    Functions from @Mateen Ulhaq and @karlo

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)


def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])


def draw_sphere(v, ax):
    """Draws a sphere on an axis

    Sphere volume and axis to draw on need to be specified

    ax must be a 3D axis
    """
    r = sphere_vol_to_r(v)
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z,  rstride=4, cstride=4,
                    color='r', linewidth=0.1, alpha=0.1)
    set_axes_equal(ax)


def draw_cube(v, ax):
    """Draws a cube on axis

    Cube volume and axis to draw on need to be specified

    ax must be a 3D axis
    """
    r = cube_vol_to_r(v)
    r = [-r/2, r/2]
    for s, e in combinations(np.array(list(product(r, r, r))), 2):
        if np.sum(np.abs(s-e)) == r[1]-r[0]:
            ax.plot3D(*zip(s, e), color="r")
    set_axes_equal(ax)


def plot_escape_locations_ellipsoid(escp_locs, A=1,B=1,C=1, npts=1000):
    r = np.linalg.norm(escp_locs[0])
    vol = 4/3*np.pi*r**3
    dist = r/10
    XYZ = np.array(random_points_on_ellipsoid([A,B,C], vol=vol, npts=npts))
    x,y,z = XYZ[:,0],XYZ[:,1],XYZ[:,2]
    intens = []
    for x1,y1,z1 in zip(x,y,z):
        loc = np.array([x1,y1,z1])
        res = np.linalg.norm(escp_locs-loc, axis=1)
        N = len(res[res<dist])
        intens.append(N)
    intens=np.array(intens)
    fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z,
                                    intensity=intens,alphahull=0,opacity=1, cmin=0)])
    return fig



def plot_escape_locations_cuboid(escp_locs, scale, dist, A=1,B=1,C=1,pos=(0,0,0)):   
    XYZ = points_on_cuboid([A,B,C], vol=scale, npts=530)
    x,y,z = XYZ[:,0],XYZ[:,1],XYZ[:,2]

    intens = []
    for x1,y1,z1 in zip(x,y,z):
        loc = np.array([x1,y1,z1])
        res = np.linalg.norm(escp_locs-loc, axis=1)
        N = len(res[res<dist])
        intens.append(N)
    intens=np.array(intens)
    fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z,
                                    intensity=intens,alphahull=2, opacity=1, cmin=0)])
    border = np.max([np.max(x), np.max(y), np.max(z)])*1.1
    fig.update_layout(
    scene = dict(
        xaxis = dict(nticks=10, range=[-border,border],),
                     yaxis = dict(nticks=4, range=[-border,border]),
                     zaxis = dict(nticks=4, range=[-border,border])))
    return fig