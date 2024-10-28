import numpy as np
from scipy.spatial.distance import euclidean
from .escape_utility import sphere_vol_to_r, calculate_delta, vol_ellipsoid
from .escape_plan import travel
from .escape_detection import in_polygon, in_ellipsoid


def make_clusters(npointspercluster, nclusters=0, v=1, jitter=0.1, cluster_points=None):
    """
    Takes a number of points per cluster, option number of clusters and points and creates a random
    distribution of points on the surface of a cell

    returns a list of clusters and their associated points
    """

    if cluster_points is None:
        cluster_points = fibonacci_spheres(samples=nclusters, v=v)
    else:
        nclusters = len(cluster_points)
    clusters = []
    for ic in cluster_points:
        jit = np.random.normal(0, jitter, (3, npointspercluster))
        points = np.reshape(ic, (3, 1)) + jit
        points /= np.linalg.norm(points, axis=0)
        points *= sphere_vol_to_r(v)
        clusters.append(points)
    return clusters


def fibonacci_spheres(samples=1, v=1, randomize=True):
    """Produces pseudo-evenly distributed points on the surface
    of a sphere

    Optional arguments give the number of points to return, volume
    of the sphere and to randomize the points initial positions.

    returns 3D coordinates of specified number of points
    """
    radius = sphere_vol_to_r(v) 
    rnd = 1 if randomize is False else np.random.randint(10)
    points = []
    offset = 2./samples
    increment = np.pi * (3. - np.sqrt(5.))
    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2)
        r = np.sqrt(1 - pow(y, 2))
        phi = ((i + rnd) % samples) * increment
        x = np.cos(phi) * r
        z = np.sin(phi) * r
        z = z * radius
        y = y * radius
        x = x * radius
        points.append(np.array([x, y, z]))
    return np.array(points).reshape((-1,3))


def sample_in_hull(hull, npts=100):

    def sample(npts):
        xmin, xmax = hull.points[:, 0].min(), hull.points[:, 0].max()
        ymin, ymax = hull.points[:, 1].min(), hull.points[:, 1].max()
        zmin, zmax = hull.points[:, 2].min(), hull.points[:, 2].max()

        Xs = np.random.uniform(low=xmin, high=xmax, size=npts)
        Ys = np.random.uniform(low=ymin, high=ymax, size=npts)
        Zs = np.random.uniform(low=zmin, high=zmax, size=npts)

        stacked = np.zeros((npts, 3))
        stacked[:, 0] = Xs
        stacked[:, 1] = Ys
        stacked[:, 2] = Zs

        mask = np.ones(npts).astype('bool')
        for idx, p in enumerate(stacked):
            if not in_polygon(p, hull):
                mask[idx] = 0
        stacked = stacked[mask]
        return stacked

    stacked = sample(npts)
    while len(stacked) < npts:
        stacked = np.concatenate((stacked, sample(npts)))

    return stacked[:npts]


def random_points_on_hull(hull, npts=1, samples=100):
    pts = []
    delta = calculate_delta(400, 1e-6)
    sampled = []
    start_pos = sample_in_hull(hull, npts=samples)
    for i in range(samples):
        cur_pos = start_pos[i]
        while(in_polygon(cur_pos, hull)):
            cur_pos = travel(delta, cur_pos)
        sampled.append(cur_pos)

    NSections = 11
    p0 = hull.max_bound
    maxDist = euclidean(p0, hull.min_bound)
    ranges = np.linspace(0, maxDist, num=NSections)

    for pt in range(npts):
        ptsSelection = []
        np.random.shuffle(sampled)
        for mi, ma in zip(ranges[:-1], ranges[1:]):
            for p1 in sampled:
                dist = euclidean(p0, p1)
                if mi < dist < ma:
                    ptsSelection.append(p1)
                    break
        np.random.shuffle(ptsSelection)
        pts.append(ptsSelection[0])

    pts = np.array(pts).reshape((-1, 3))    
    return pts

def scale_cuboid(ABC, vol):
    volN = ABC[0] * ABC[1] * ABC[2]
    cbrt_diff = np.cbrt(vol/volN)
    ABC = np.array(ABC) * cbrt_diff
    s1 = ABC[0] * ABC[1] * 2
    s2 = ABC[0] * ABC[2] * 2
    s3 = ABC[1] * ABC[2] * 2 
    sas = [s1, s2, s3]
    ta = s1+s2+s3
    return ABC, sas, ta

def make_pts_cuboid_side(endsX, endsY, dist_p):
    endsX /=2 
    endsY /=2
    xy = np.around(np.array(np.meshgrid(np.arange(-endsX, endsX, dist_p), np.arange(-endsY, endsY, dist_p))).reshape(2,-1).T, 2)
    xy = np.array(xy) + np.array([dist_p, dist_p])

    xys = []
    for pt in xy:
        if abs(pt[0]) < endsX and abs(pt[1]) < endsY:
            xys.append(pt)
    return np.array(xys)


def conv_2D_cuboid_surface_to_3D_pts(ps, xy, i):
    if i ==0:
        ps[len(ps)//2:, 0:2] = xy
        ps[:len(ps)//2, 0:2] = xy
    elif i==1:
        ps[len(ps)//2:, 1:] = xy
        ps[:len(ps)//2, 1:] = xy
    else:
        ps[len(ps)//2:, 0::2] = xy
        ps[:len(ps)//2, 0::2] = xy
    return ps

def points_on_cuboid(ABC, vol=1, npts=1, faces_dist=np.array([1.,1.,1.])):
    ABC, _, ta = scale_cuboid(ABC, vol)

    # normalize faces
    faces_dist=faces_dist/faces_dist.max()

    # area per pore
    # this modification will mess up the numbers depending on cell wall area per surface
    # this coooould be an issue for wanting exact numbers, 
    # but hey, then make your own custom distribution
    app = ta/npts
    app_pw = app/faces_dist
    dist_ps = np.sqrt(app_pw)

    pts=[]
    xyzs = [(0,1), (1,2), (0,2)]
    for i, (xyz, dist_p) in enumerate(zip(xyzs, dist_ps) ):
        xy = make_pts_cuboid_side(ABC[xyz[0]], ABC[xyz[1]], dist_p)
        ps = np.ones((len(xy)*2, 3)) * (ABC/2)
        # half these pores will be on the opp side (to create mirroring effect)
        ps[len(ps)//2:] = -ps[len(ps)//2:] 
        ps = conv_2D_cuboid_surface_to_3D_pts(ps, xy, i)
        pts.append(ps)
    return np.concatenate(np.array(pts, dtype='object'))

def make_distrib_points_2d(num_pts, offsetx=0, offsety=0, ri=1):
    indices = np.arange(0, num_pts, dtype=float) + 0.5

    r = np.sqrt(indices/num_pts)
    theta = np.pi * (1 + 5**0.5) * indices

    pts = (r*np.cos(theta), r*np.sin(theta))

    return np.column_stack(np.array(pts)) * ri + np.array([offsetx, offsety])


def make_custer_pts_side(ppa, endsX, endsY, pts_per_cluster, ri=None):
    pts = make_pts_cuboid_side(ppa, endsX, endsY)
    dist_p = np.min([endsX/2/int(np.sqrt(ppa)), endsY/2/int(np.sqrt(ppa)) ])
    spirals = []
    for pt in pts:
        spirals.append(make_distrib_points_2d(pts_per_cluster, offsetx=pt[0], offsety=pt[1], ri=(dist_p/2 if ri is None else ri) ))
    return np.concatenate(spirals)


def make_clusters_on_cuboid(ABC, vol=1, npts=6, nclusters=6, ri=None):
    if npts<6 or nclusters<6:
        raise ValueError('Cuboids have 6 sides, need at least 1 pore per side and nclusters >= npts (npts >= 6, nclusters >= 6)')
    ABC, sas, ta = scale_cuboid(ABC, vol)
    pts = [ ]
    for i, xyz in enumerate([(0,1), (0,2), (1,2)]):
        pts_per_surface = nclusters/6 #(sas[i]/2 / ta) * npts
        sp_xy = make_custer_pts_side(pts_per_surface, ABC[xyz[0]], ABC[xyz[1]], npts/nclusters, ri=ri)
        #conv_2D_cuboid_surface_to_3D_pts(i, pts, ABC, sp_xy)
        ps = np.ones((len(sp_xy)*2, 3)) * (ABC[xyz[0]]/2)
        ps[len(ps)//2:] = -ps[len(ps)//2:] 
        ps = conv_2D_cuboid_surface_to_3D_pts(ps, sp_xy, i)
        pts.append(ps)
    return np.concatenate(np.array(pts))

def random_point_ellipsoid(a, b, c):
    """
    This function is taken from stackoverflow user Nikolay Frick
    https://stackoverflow.com/a/61786434

    It is used with minor modification
    """

    u = np.random.rand()
    v = np.random.rand()
    theta = u * 2.0 * np.pi
    phi = np.arccos(2.0 * v - 1.0)
    sinTheta = np.sin(theta)
    cosTheta = np.cos(theta)
    sinPhi = np.sin(phi)
    cosPhi = np.cos(phi)
    rx = a * sinPhi * cosTheta
    ry = b * sinPhi * sinTheta
    rz = c * cosPhi
    return rx, ry, rz


def random_points_on_ellipsoid(ABC, vol=1, npts=1):
    pts = []
    ABC = np.array(ABC).astype('float64')
    volN = vol_ellipsoid(*ABC)
    cbrt_diff = np.cbrt(vol/volN)
    a, b, c = np.array(ABC * cbrt_diff)
    for i in range(npts):
        pts.append(random_point_ellipsoid(a, b, c))
    return pts


def random_points_on_cube_surface(samples, r=1):
    """Gives a random distribution of points on a cube surface

    A number of samples and an optional cube radius can be given

    returns a series of points randomly distributed on surface of cube
    """
    points = []
    r = np.cbrt(r)
    for i in range(samples):
        p = np.random.random(3) * (r/2) * \
            np.random.choice([-1, 1], 3)
        dim = np.random.choice([0, 1, 2])
        p[dim] = (r/2) * np.random.choice([-1, 1])
        points.append(p)
    return points
