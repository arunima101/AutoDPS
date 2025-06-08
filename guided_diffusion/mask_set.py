import torch
from torch.fft import fftshift, ifftshift, fftn, ifftn
Ft = lambda x : torch.fft.fftshift(torch.fft.fft2(x))
IFt = lambda x : torch.fft.ifftn(torch.fft.ifftshift(x))

import numpy as np
# import numba as nb
from numpy.lib.stride_tricks import as_strided

def normal_pdf(length, sensitivity):
    return np.exp(-sensitivity * (np.arange(length) - length / 2)**2)


def cartesian_mask(shape, acc, sample_n=10, centred=True):
    """
    Sampling density estimated from implementation of kt FOCUSS

    shape: tuple - of form (..., nx, ny)
    acc: float - doesn't have to be integer 4, 8, etc..

    """
    N, Nx, Ny = int(np.prod(shape[:-2])), shape[-2], shape[-1]
    pdf_x = normal_pdf(Nx, 0.5/(Nx/10.)**2)
    lmda = Nx/(2.*acc)
    n_lines = int(Nx / acc)

    # add uniform distribution
    pdf_x += lmda * 1./Nx

    if sample_n:
        pdf_x[Nx//2-sample_n//2:Nx//2+sample_n//2] = 0
        pdf_x /= np.sum(pdf_x)
        n_lines -= sample_n

    mask = np.zeros((N, Nx))
    for i in range(N):
        idx = np.random.choice(Nx, n_lines, False, pdf_x)
        mask[i, idx] = 1

    if sample_n:
        mask[:, Nx//2-sample_n//2:Nx//2+sample_n//2] = 1
#         print(mask.shape)
#         plt.imshow(abs(mask),cmap='gray')

    size = mask.itemsize
    mask = as_strided(mask, (N, Nx, Ny), (size * Nx, size, 0))

    mask = mask.reshape(shape)

    if not centred:
        mask = IFt(mask)
    
    mask_torch = torch.from_numpy(mask)
    return mask_torch

def poisson(
    img_shape,
    accel,
    calib=(0, 0),
    dtype=complex,
    crop_corner=True,
    return_density=False,
    seed=0,
    max_attempts=30,
    tol=0.1,
):

    if accel <= 1:
        raise ValueError(f"accel must be greater than 1, got {accel}")

    if seed is not None:
        rand_state = np.random.get_state()

    ny, nx = img_shape
    y, x = np.mgrid[:ny, :nx]
    x = np.maximum(abs(x - img_shape[-1] / 2) - calib[-1] / 2, 0)
    x /= x.max()
    y = np.maximum(abs(y - img_shape[-2] / 2) - calib[-2] / 2, 0)
    y /= y.max()
    r = np.sqrt(x**2 + y**2)

    slope_max = max(nx, ny)
    slope_min = 0
    while slope_min < slope_max:
        slope = (slope_max + slope_min) / 2
        radius_x = np.clip((1 + r * slope) * nx / max(nx, ny), 1, None)
        radius_y = np.clip((1 + r * slope) * ny / max(nx, ny), 1, None)
        mask = _poisson(
            img_shape[-1],
            img_shape[-2],
            max_attempts,
            radius_x,
            radius_y,
            calib,
            seed,
        )
        if crop_corner:
            mask *= r < 1

        actual_accel = img_shape[-1] * img_shape[-2] / np.sum(mask)

        if abs(actual_accel - accel) < tol:
            break
        if actual_accel < accel:
            slope_min = slope
        else:
            slope_max = slope

    if abs(actual_accel - accel) >= tol:
        raise ValueError(f"Cannot generate mask to satisfy accel={accel}.")

    mask = mask.reshape(img_shape).astype(dtype)

    if seed is not None:
        np.random.set_state(rand_state)

    return torch.from_numpy(mask)

# @nb.jit(nopython=True, cache=True)
def _poisson(nx, ny, max_attempts, radius_x, radius_y, calib, seed=None):
    mask = np.zeros((ny, nx))

    # Add calibration region
    mask[
        int(ny / 2 - calib[-2] / 2) : int(ny / 2 + calib[-2] / 2),
        int(nx / 2 - calib[-1] / 2) : int(nx / 2 + calib[-1] / 2),
    ] = 1

    if seed is not None:
        np.random.seed(int(seed))

    # initialize active list
    pxs = np.empty(nx * ny, np.int32)
    pys = np.empty(nx * ny, np.int32)
    pxs[0] = np.random.randint(0, nx)
    pys[0] = np.random.randint(0, ny)
    num_actives = 1
    while num_actives > 0:
        i = np.random.randint(0, num_actives)
        px = pxs[i]
        py = pys[i]
        rx = radius_x[py, px]
        ry = radius_y[py, px]

        # Attempt to generate point
        done = False
        k = 0
        while not done and k < max_attempts:
            # Generate point randomly from r and 2 * r
            v = (np.random.random() * 3 + 1) ** 0.5
            t = 2 * np.pi * np.random.random()
            qx = px + v * rx * np.cos(t)
            qy = py + v * ry * np.sin(t)

            # Reject if outside grid or close to other points
            if qx >= 0 and qx < nx and qy >= 0 and qy < ny:
                startx = max(int(qx - rx), 0)
                endx = min(int(qx + rx + 1), nx)
                starty = max(int(qy - ry), 0)
                endy = min(int(qy + ry + 1), ny)

                done = True
                for x in range(startx, endx):
                    for y in range(starty, endy):
                        if mask[y, x] == 1 and (
                            ((qx - x) / radius_x[y, x]) ** 2
                            + ((qy - y) / (radius_y[y, x])) ** 2
                            < 1
                        ):
                            done = False
                            break

            k += 1

        # Add point if done else remove from active list
        if done:
            pxs[num_actives] = qx
            pys[num_actives] = qy
            mask[int(qy), int(qx)] = 1
            num_actives += 1
        else:
            pxs[i] = pxs[num_actives - 1]
            pys[i] = pys[num_actives - 1]
            num_actives -= 1

    return mask

def create_mask(mask_type,shape,acc):
    if mask_type =='cartesian':
        return cartesian_mask(shape,acc)
    if mask_type =='poisson':
        b,c,h,w= shape
        mask = poisson((h,w),acc,seed=0)
        return torch.reshape(mask,shape)
    
        