import numpy as np
from .escape_utility import sphere_vol_to_r, cube_vol_to_r, calculate_delta
from .escape_utility import calculate_opt_dt, vol_ellipsoid
from .escape_detection import in_sphere, in_cube, in_polygon, in_ellipsoid, in_cuboid
from .escape_detection import passthrough_pore
MAX_NEW_MOVEMENTS = 100



def pre_generate_steps(delta, p, max_steps, dtype=np.float32):
    target_shape = (p.shape[0], max_steps)
    xyz = np.random.random(target_shape)
    xyz_sum = np.sum(xyz, axis=0)
    xyz = np.sqrt(xyz / xyz_sum) * delta * \
        np.random.choice([-1, +1], target_shape).astype(np.int8)
    return xyz.T.astype(dtype)

def escape(D, vol, pore_size, pore_locs, dt=None, seed=None,
           shape='sphere', hull=None, ABC=None, max_steps=(int(1e7)), track=False,
            tol=1, inner_r=None, dtype=np.float16):
    """Wrapper function that can be called by a user - used to optimise code
    shared between escape methods

    Takes a diffusion coefficient, volume of container, escape pore size, escape
    pore(s) location. Optional arguments of step-size and a numpy random seed
    can be given, as well as specifying maximum number of steps that a walk can
    take. Switches to return the full path (or just the time required), to
    start the particle at the container centroid or random placement

    """
    if dt is None:
        dt = calculate_opt_dt(pore_size, D)

    delta = calculate_delta(D, dt)
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed()
    max_steps = (int(1/dt) if max_steps is None else max_steps)

    if shape == 'sphere':
        check_func = in_sphere
        r = sphere_vol_to_r(vol)
    elif shape == 'cube':
        check_func = in_cube
        r = cube_vol_to_r(vol)
    elif shape == 'cuboid':
        if ABC is None:
            raise NameError("Argument 'ABC' is undefined")
        ABC = np.array(ABC).astype('float64')
        volN = np.prod(ABC)
        cbrt_diff = np.cbrt(vol/volN)
        ABC = np.array(ABC * cbrt_diff) 
        def cuboid_wrapper(p,r):
            return in_cuboid(p, ABC)
        check_func = cuboid_wrapper
        r = 0
    elif shape == 'ellipsoid':
        if ABC is None:
            raise NameError("Argument 'ABC' is undefined")
        ABC = np.array(ABC).astype('float64')
        volN = vol_ellipsoid(*ABC)
        cbrt_diff = np.cbrt(vol/volN)
        a, b, c = np.array(ABC * cbrt_diff)
        def ellipsoid_wrapper(p, r):
            return in_ellipsoid(p[0], p[1], p[2], a, b, c)
        check_func = ellipsoid_wrapper
        r = 0

    elif shape == 'polygon':
        check_func = in_polygon
        if hull is None:
            raise NameError("Argument 'hull' is undefined")
        r = hull

    cur_pos = np.zeros(3)

    if inner_r is not None:
        cur_pos[0] = inner_r
        return escape_with_path_inner_sphere(r, delta, dt,
                            max_steps, pore_locs,
                            pore_size, check_func, cur_pos, tol, inner_r=inner_r)



    return escape_aio(r, delta, dt, max_steps, pore_locs, pore_size, check_func,
                        cur_pos, tol, dtype, track)


def escape_aio(r, delta, dt, max_steps,
                 pore_locs, pore_size, check_func, cur_pos, tol, dtype, track=False):
    """Simulates escape without tracking position through container

    Takes a radius of container, delta step size, dt difference of time, a
    shape (cube of sphere), a maximum number of steps to allow, location of
    escape pore(s), size of escape pore(s), a checking function for collision
    and the current position of the particle

    returns a number of steps taken to escape

    """
    pot_steps = pre_generate_steps(delta, cur_pos, max_steps, dtype)
    if track:
        steps = np.zeros(len(pot_steps), dtype=np.int32)
        step_iter = iter(range(len(steps)))
    idx = 0
    while idx < len(pot_steps):
        new_pos = cur_pos+pot_steps[idx]
        while (not check_func(new_pos, r)) and idx < max_steps:
            if passthrough_pore(new_pos, pore_locs, r=pore_size, tol=tol):
                    return ( pot_steps[steps[steps>0]], (idx+1)*dt ) if track else (idx+1)*dt
            idx += 1
            new_pos = cur_pos+pot_steps[idx]
        cur_pos = new_pos
        if track:
            steps[next(step_iter)] = idx
        idx+=1 
    return 0


def travel(delta,  pa):
    """Find a new position for a particle
    Takes a delta, movement size and a particle of N dimensions returns a new
    array of similar dimensions to pa.

    DEPRECATED! Kept only for hull functions and inner spheres
    """

    p = pa.copy()
    xyz = np.random.random(p.shape)
    xyz_sum = np.sum(xyz)
    xyz = np.sqrt(xyz / xyz_sum) * delta * \
        np.random.choice([-1, +1], p.shape)
    p += xyz
    return p

def escape_with_path_inner_sphere(r, delta, dt, max_steps,
                     pore_locs, pore_size, check_func, cur_pos, tol, inner_r):
    """Provides the full path of a particle as it escapes from a container

    Takes a radius of container, delta step size, dt difference of time, a
    shape (cube of sphere), a maximum number of steps to allow, location of
    escape pore(s), size of escape pore(s), a checking function for collision
    and the current position of the particle

    A inner sphere is included as an additional obstacle 

    returns a multi-dimensional array of a particle at each position as it
    escapes the container will return if max steps is reached or when escape is
    detected

    """
    path = np.zeros((max_steps, 3))
    path[0] = cur_pos
    steps = 0
    while steps < max_steps:
        new_pos = travel(delta, cur_pos)
        new_pos_steps = 0
        while (not (check_func(new_pos, r)) and new_pos_steps < MAX_NEW_MOVEMENTS or np.linalg.norm(new_pos) < inner_r ):
            new_pos = travel(delta, cur_pos)
            new_pos_steps += 1
            for pd_loc in pore_locs:
                if passthrough_pore(new_pos, pd_loc, r=pore_size, tol=tol):
                    path[steps] = new_pos
                    return (steps+1)*dt, path[:steps]
        cur_pos = new_pos
        steps = steps + 1
        path[steps] = cur_pos
    return (steps*dt, path) if steps < max_steps else (0, path)