import numpy as np
import torch
import src.radam as radam
import src.model as mmodel

import os
import math
import time

import taichi as ti


def batchify_query(inputs, query_function, batch_size=2 ** 22):
    """
    args:
        inputs: [..., input_dim]
    return:
        outputs: [..., output_dim]
    """
    input_dim = inputs.shape[-1]
    input_shape = inputs.shape
    inputs = inputs.view(-1, input_dim)  # flatten all but last dim
    N = inputs.shape[0]
    outputs = []
    for i in range(0, N, batch_size):
        output = query_function(inputs[i:i + batch_size])
        if isinstance(output, tuple):
            output = output[0]
        outputs.append(output)
    outputs = torch.cat(outputs, dim=0)
    return outputs.view(*input_shape[:-1], -1)  # unflatten


@ti.func
def sample(qf: ti.template(), u: float, v: float, w: float):
    u_dim, v_dim, w_dim = qf.shape
    i = ti.max(0, ti.min(int(u), u_dim - 1))
    j = ti.max(0, ti.min(int(v), v_dim - 1))
    k = ti.max(0, ti.min(int(w), w_dim - 1))
    return qf[i, j, k]


@ti.kernel
def split_central_vector(vc: ti.template(), vx: ti.template(), vy: ti.template(), vz: ti.template()):
    for i, j, k in vx:
        r = sample(vc, i, j, k)
        l = sample(vc, i - 1, j, k)
        vx[i, j, k] = 0.5 * (r.x + l.x)
    for i, j, k in vy:
        t = sample(vc, i, j, k)
        b = sample(vc, i, j - 1, k)
        vy[i, j, k] = 0.5 * (t.y + b.y)
    for i, j, k in vz:
        c = sample(vc, i, j, k)
        a = sample(vc, i, j, k - 1)
        vz[i, j, k] = 0.5 * (c.z + a.z)


@ti.kernel
def get_central_vector(vx: ti.template(), vy: ti.template(), vz: ti.template(), vc: ti.template()):
    for i, j, k in vc:
        vc[i, j, k].x = 0.5 * (vx[i + 1, j, k] + vx[i, j, k])
        vc[i, j, k].y = 0.5 * (vy[i, j + 1, k] + vy[i, j, k])
        vc[i, j, k].z = 0.5 * (vz[i, j, k + 1] + vz[i, j, k])


@ti.data_oriented
class MGPCG:
    '''
Grid-based MGPCG solver for the possion equation.

.. note::

    This solver only runs on CPU and CUDA backends since it requires the
    ``pointer`` SNode.
    '''

    def __init__(self, boundary_types, N, dim=2, base_level=3, real=float):
        '''
        :parameter dim: Dimensionality of the fields.
        :parameter N: Grid resolutions.
        :parameter n_mg_levels: Number of multigrid levels.
        '''

        # grid parameters
        self.use_multigrid = True

        self.N = N
        self.n_mg_levels = int(math.log2(min(N))) - base_level + 1
        self.pre_and_post_smoothing = 2
        self.bottom_smoothing = 50
        self.dim = dim
        self.real = real

        # setup sparse simulation data arrays
        self.r = [ti.field(dtype=self.real)
                  for _ in range(self.n_mg_levels)]  # residual
        self.z = [ti.field(dtype=self.real)
                  for _ in range(self.n_mg_levels)]  # M^-1 self.r
        self.x = ti.field(dtype=self.real)  # solution
        self.p = ti.field(dtype=self.real)  # conjugate gradient
        self.Ap = ti.field(dtype=self.real)  # matrix-vector product
        self.alpha = ti.field(dtype=self.real)  # step size
        self.beta = ti.field(dtype=self.real)  # step size
        self.sum = ti.field(dtype=self.real)  # storage for reductions
        self.r_mean = ti.field(dtype=self.real)  # storage for avg of r
        self.num_entries = math.prod(self.N)

        indices = ti.ijk if self.dim == 3 else ti.ij
        self.grid = ti.root.pointer(indices, [n // 4 for n in self.N]).dense(
            indices, 4).place(self.x, self.p, self.Ap)

        for l in range(self.n_mg_levels):
            self.grid = ti.root.pointer(indices,
                                        [n // (4 * 2 ** l) for n in self.N]).dense(
                indices,
                4).place(self.r[l], self.z[l])

        ti.root.place(self.alpha, self.beta, self.sum, self.r_mean)

        self.boundary_types = boundary_types

    @ti.func
    def init_r(self, I, r_I):
        self.r[0][I] = r_I
        self.z[0][I] = 0
        self.Ap[I] = 0
        self.p[I] = 0
        self.x[I] = 0

    @ti.kernel
    def init(self, r: ti.template(), k: ti.template()):
        '''
        Set up the solver for $\nabla^2 x = k r$, a scaled Poisson problem.
        :parameter k: (scalar) A scaling factor of the right-hand side.
        :parameter r: (ti.field) Unscaled right-hand side.
        '''
        for I in ti.grouped(ti.ndrange(*self.N)):
            self.init_r(I, r[I] * k)

    @ti.kernel
    def get_result(self, x: ti.template()):
        '''
        Get the solution field.

        :parameter x: (ti.field) The field to store the solution
        '''
        for I in ti.grouped(ti.ndrange(*self.N)):
            x[I] = self.x[I]

    @ti.func
    def neighbor_sum(self, x, I):
        dims = x.shape
        ret = ti.cast(0.0, self.real)
        for i in ti.static(range(self.dim)):
            offset = ti.Vector.unit(self.dim, i)
            # add right if has right
            if I[i] < dims[i] - 1:
                ret += x[I + offset]
            # add left if has left
            if I[i] > 0:
                ret += x[I - offset]
        return ret

    @ti.func
    def num_fluid_neighbors(self, x, I):
        dims = x.shape
        num = 2.0 * self.dim
        for i in ti.static(range(self.dim)):
            if I[i] <= 0 and self.boundary_types[i, 0] == 2:
                num -= 1.0
            if I[i] >= dims[i] - 1 and self.boundary_types[i, 1] == 2:
                num -= 1.0
        return num

    @ti.kernel
    def compute_Ap(self):
        for I in ti.grouped(self.Ap):
            multiplier = self.num_fluid_neighbors(self.p, I)
            self.Ap[I] = multiplier * self.p[I] - self.neighbor_sum(
                self.p, I)

    @ti.kernel
    def get_Ap(self, p: ti.template(), Ap: ti.template()):
        for I in ti.grouped(Ap):
            multiplier = self.num_fluid_neighbors(p, I)
            Ap[I] = multiplier * p[I] - self.neighbor_sum(
                p, I)

    @ti.kernel
    def reduce(self, p: ti.template(), q: ti.template()):
        self.sum[None] = 0
        for I in ti.grouped(p):
            self.sum[None] += p[I] * q[I]

    @ti.kernel
    def update_x(self):
        for I in ti.grouped(self.p):
            self.x[I] += self.alpha[None] * self.p[I]

    @ti.kernel
    def update_r(self):
        for I in ti.grouped(self.p):
            self.r[0][I] -= self.alpha[None] * self.Ap[I]

    @ti.kernel
    def update_p(self):
        for I in ti.grouped(self.p):
            self.p[I] = self.z[0][I] + self.beta[None] * self.p[I]

    @ti.kernel
    def restrict(self, l: ti.template()):
        for I in ti.grouped(self.r[l]):
            multiplier = self.num_fluid_neighbors(self.z[l], I)
            res = self.r[l][I] - (multiplier * self.z[l][I] -
                                  self.neighbor_sum(self.z[l], I))
            self.r[l + 1][I // 2] += res * 1.0 / (self.dim - 1.0)

    @ti.kernel
    def prolongate(self, l: ti.template()):
        for I in ti.grouped(self.z[l]):
            self.z[l][I] += self.z[l + 1][I // 2]

    @ti.kernel
    def smooth(self, l: ti.template(), phase: ti.template()):
        # phase = red/black Gauss-Seidel phase
        for I in ti.grouped(self.r[l]):
            if (I.sum()) & 1 == phase:
                multiplier = self.num_fluid_neighbors(self.z[l], I)
                self.z[l][I] = (self.r[l][I] + self.neighbor_sum(
                    self.z[l], I)) / multiplier

    @ti.kernel
    def recenter(self, r: ti.template()):  # so that the mean value of r is 0
        self.r_mean[None] = 0.0
        for I in ti.grouped(r):
            self.r_mean[None] += r[I] / self.num_entries
        for I in ti.grouped(r):
            r[I] -= self.r_mean[None]

    def apply_preconditioner(self):
        self.z[0].fill(0)
        for l in range(self.n_mg_levels - 1):
            for i in range(self.pre_and_post_smoothing):
                self.smooth(l, 0)
                self.smooth(l, 1)
            self.z[l + 1].fill(0)
            self.r[l + 1].fill(0)
            self.restrict(l)

        for i in range(self.bottom_smoothing):
            self.smooth(self.n_mg_levels - 1, 0)
            self.smooth(self.n_mg_levels - 1, 1)

        for l in reversed(range(self.n_mg_levels - 1)):
            self.prolongate(l)
            for i in range(self.pre_and_post_smoothing):
                self.smooth(l, 1)
                self.smooth(l, 0)

    def solve(self,
              max_iters=-1,
              eps=1e-12,
              tol=1e-12,
              verbose=False):
        '''
        Solve a Poisson problem.

        :parameter max_iters: Specify the maximal iterations. -1 for no limit.
        :parameter eps: Specify a non-zero value to prevent ZeroDivisionError.
        :parameter abs_tol: Specify the absolute tolerance of loss.
        :parameter rel_tol: Specify the tolerance of loss relative to initial loss.
        '''
        all_neumann = (self.boundary_types.sum() == 2 * 2 * self.dim)

        # self.r = b - Ax = b    since self.x = 0
        # self.p = self.r = self.r + 0 self.p

        if all_neumann:
            self.recenter(self.r[0])
        if self.use_multigrid:
            self.apply_preconditioner()
        else:
            self.z[0].copy_from(self.r[0])

        self.update_p()

        self.reduce(self.z[0], self.r[0])
        old_zTr = self.sum[None]
        # print("[MGPCG] Starting error: ", math.sqrt(old_zTr))

        # Conjugate gradients
        it = 0
        start_t = time.time()
        while max_iters == -1 or it < max_iters:
            # self.alpha = rTr / pTAp
            self.compute_Ap()
            self.reduce(self.p, self.Ap)
            pAp = self.sum[None]
            self.alpha[None] = old_zTr / (pAp + eps)

            # self.x = self.x + self.alpha self.p
            self.update_x()

            # self.r = self.r - self.alpha self.Ap
            self.update_r()

            # check for convergence
            self.reduce(self.r[0], self.r[0])
            rTr = self.sum[None]

            if verbose:
                print(f'iter {it}, |residual|_2={math.sqrt(rTr)}')

            if rTr < tol:
                end_t = time.time()
                # print("[MGPCG] final error: ", math.sqrt(rTr), " using time: ", end_t - start_t)
                return

            if all_neumann:
                self.recenter(self.r[0])
            # self.z = M^-1 self.r
            if self.use_multigrid:
                self.apply_preconditioner()
            else:
                self.z[0].copy_from(self.r[0])

            # self.beta = new_rTr / old_rTr
            self.reduce(self.z[0], self.r[0])
            new_zTr = self.sum[None]
            self.beta[None] = new_zTr / (old_zTr + eps)

            # self.p = self.z + self.beta self.p
            self.update_p()
            old_zTr = new_zTr

            it += 1

        end_t = time.time()
        # print("[MGPCG] Return without converging at iter: ", it, " with final error: ", math.sqrt(rTr), " using time: ",
        #       end_t - start_t)


class MGPCG_3(MGPCG):

    def __init__(self, boundary_types, N, base_level=3, real=float):
        super().__init__(boundary_types, N, dim=3, base_level=base_level, real=real)

        rx, ry, rz = N
        self.u_div = ti.field(float, shape=N)
        self.p = ti.field(float, shape=N)
        self.boundary_types = boundary_types
        self.u_x = ti.field(float, shape=(rx + 1, ry, rz))
        self.u_y = ti.field(float, shape=(rx, ry + 1, rz))
        self.u_z = ti.field(float, shape=(rx, ry, rz + 1))
        self.u = ti.Vector.field(3, float, shape=(rx, ry, rz))
        self.u_y_bottom = ti.field(float, shape=(rx, 1, rz))

    @ti.kernel
    def apply_bc(self, u_x: ti.template(), u_y: ti.template(), u_z: ti.template()):
        u_dim, v_dim, w_dim = u_x.shape
        for i, j, k in u_x:
            if i == 0 and self.boundary_types[0, 0] == 2:
                u_x[i, j, k] = 0
            if i == u_dim - 1 and self.boundary_types[0, 1] == 2:
                u_x[i, j, k] = 0
        u_dim, v_dim, w_dim = u_y.shape
        for i, j, k in u_y:
            if j == 0 and self.boundary_types[1, 0] == 2:
                u_y[i, j, k] = self.u_y_bottom[i, j, k]
                # u_y[i, j, k] = 0.5
            if j == v_dim - 1 and self.boundary_types[1, 1] == 2:
                u_y[i, j, k] = 0
        u_dim, v_dim, w_dim = u_z.shape
        for i, j, k in u_z:
            if k == 0 and self.boundary_types[2, 0] == 2:
                u_z[i, j, k] = 0
            if k == w_dim - 1 and self.boundary_types[2, 1] == 2:
                u_z[i, j, k] = 0

    @ti.kernel
    def divergence(self, u_x: ti.template(), u_y: ti.template(), u_z: ti.template()):
        u_dim, v_dim, w_dim = self.u_div.shape
        for i, j, k in self.u_div:
            vl = sample(u_x, i, j, k)
            vr = sample(u_x, i + 1, j, k)
            vb = sample(u_y, i, j, k)
            vt = sample(u_y, i, j + 1, k)
            va = sample(u_z, i, j, k)
            vc = sample(u_z, i, j, k + 1)
            self.u_div[i, j, k] = vr - vl + vt - vb + vc - va

    @ti.kernel
    def subtract_grad_p(self, u_x: ti.template(), u_y: ti.template(), u_z: ti.template()):
        u_dim, v_dim, w_dim = self.p.shape
        for i, j, k in u_x:
            pr = sample(self.p, i, j, k)
            pl = sample(self.p, i - 1, j, k)
            if i - 1 < 0:
                pl = 0
            if i >= u_dim:
                pr = 0
            u_x[i, j, k] -= (pr - pl)
        for i, j, k in u_y:
            pt = sample(self.p, i, j, k)
            pb = sample(self.p, i, j - 1, k)
            if j - 1 < 0:
                pb = 0
            if j >= v_dim:
                pt = 0
            u_y[i, j, k] -= pt - pb
        for i, j, k in u_z:
            pc = sample(self.p, i, j, k)
            pa = sample(self.p, i, j, k - 1)
            if k - 1 < 0:
                pa = 0
            if j >= w_dim:
                pc = 0
            u_z[i, j, k] -= pc - pa

    def solve_pressure_MGPCG(self, verbose):
        self.init(self.u_div, -1)
        self.solve(max_iters=400, verbose=verbose, tol=1.e-12)
        self.get_result(self.p)

    @ti.kernel
    def set_uy_bottom(self):
        for i, j, k in self.u_y:
            if j == 0 and self.boundary_types[1, 0] == 2:
                self.u_y_bottom[i, j, k] = self.u_y[i, j, k]

    def Poisson(self, vel, verbose=False):
        """
        args:
            vel: torch tensor of shape (X, Y, Z, 3)
        returns:
            vel: torch tensor of shape (X, Y, Z, 3), projected
        """
        self.u.from_torch(vel)
        split_central_vector(self.u, self.u_x, self.u_y, self.u_z)
        self.set_uy_bottom()
        self.apply_bc(self.u_x, self.u_y, self.u_z)
        self.divergence(self.u_x, self.u_y, self.u_z)
        self.solve_pressure_MGPCG(verbose=verbose)
        self.subtract_grad_p(self.u_x, self.u_y, self.u_z)
        self.apply_bc(self.u_x, self.u_y, self.u_z)
        get_central_vector(self.u_x, self.u_y, self.u_z, self.u)
        vel = self.u.to_torch()
        return vel


def get_rays_np_continuous(H, W, c2w):
    # Generate random offsets for pixel coordinates
    random_offset = np.random.uniform(0, 1, size=(H, W, 2))
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    pixel_coords = np.stack((i, j), axis=-1) + random_offset

    # Clip pixel coordinates
    pixel_coords[..., 0] = np.clip(pixel_coords[..., 0], 0, W - 1)
    pixel_coords[..., 1] = np.clip(pixel_coords[..., 1], 0, H - 1)

    # Compute ray directions in camera space
    dirs = np.stack([
        (pixel_coords[..., 0] - K[0][2]) / K[0][0],
        -(pixel_coords[..., 1] - K[1][2]) / K[1][1],
        -np.ones_like(pixel_coords[..., 0])
    ], axis=-1)

    # Transform ray directions to world space
    rays_d = dirs @ c2w[:3, :3].T

    # Compute ray origins in world space
    rays_o = np.broadcast_to(c2w[:3, -1], rays_d.shape)

    return rays_o, rays_d, pixel_coords[..., 0], pixel_coords[..., 1]


def sample_bilinear(img, xy):
    """
    Sample image with bilinear interpolation
    :param img: (T, V, H, W, 3)
    :param xy: (V, 2, H, W)
    :return: img: (T, V, H, W, 3)
    """
    T, V, H, W, _ = img.shape
    u, v = xy[:, 0], xy[:, 1]

    u = np.clip(u, 0, W - 1)
    v = np.clip(v, 0, H - 1)

    u_floor, v_floor = np.floor(u).astype(int), np.floor(v).astype(int)
    u_ceil, v_ceil = np.ceil(u).astype(int), np.ceil(v).astype(int)

    u_ratio, v_ratio = u - u_floor, v - v_floor
    u_ratio, v_ratio = u_ratio[None, ..., None], v_ratio[None, ..., None]

    bottom_left = img[:, np.arange(V)[:, None, None], v_floor, u_floor]
    bottom_right = img[:, np.arange(V)[:, None, None], v_floor, u_ceil]
    top_left = img[:, np.arange(V)[:, None, None], v_ceil, u_floor]
    top_right = img[:, np.arange(V)[:, None, None], v_ceil, u_ceil]

    bottom = (1 - u_ratio) * bottom_left + u_ratio * bottom_right
    top = (1 - u_ratio) * top_left + u_ratio * top_right

    interpolated = (1 - v_ratio) * bottom + v_ratio * top

    return interpolated


def do_resample_rays(H, W):
    rays_list = []
    ij = []
    for p in POSES_TRAIN_np[:, :3, :4]:
        r_o, r_d, i_, j_ = get_rays_np_continuous(H, W, p)
        rays_list.append([r_o, r_d])
        ij.append([i_, j_])
    ij = np.stack(ij, 0)
    images_train_sample = sample_bilinear(IMAGE_TRAIN_np, ij)
    ret_IMAGE_TRAIN_gpu = torch.tensor(images_train_sample, device=device, dtype=torch.float32).flatten(start_dim=1, end_dim=3)

    rays_np = np.stack(rays_list, 0)
    rays_np = np.transpose(rays_np, [0, 2, 3, 1, 4])
    rays_np = np.reshape(rays_np, [-1, 2, 3])  # [VHW, ro+rd=2, 3]
    rays_np = rays_np.astype(np.float32)
    ret_RAYs_gpu = torch.tensor(rays_np, device=device, dtype=torch.float32)
    ret_RAY_IDX_gpu = torch.randperm(ret_RAYs_gpu.shape[0], device=device, dtype=torch.int32)

    return ret_IMAGE_TRAIN_gpu, ret_RAYs_gpu, ret_RAY_IDX_gpu


def get_ray_batch(RAYs: torch.Tensor, RAYs_IDX: torch.Tensor, start: int, end: int):
    BATCH_RAYs_IDX = RAYs_IDX[start:end]  # [batch_size]
    BATCH_RAYs_O, BATCH_RAYs_D = torch.transpose(RAYs[BATCH_RAYs_IDX], 0, 1)  # [batch_size, 3]
    return BATCH_RAYs_O, BATCH_RAYs_D, BATCH_RAYs_IDX


def get_frames_at_times(IMAGEs: torch.Tensor, N_frames: int, N_times: int):
    assert N_frames > 1
    TIMEs_IDX = torch.randperm(N_frames, device=IMAGEs.device, dtype=torch.float32)[:N_times] + torch.randn(N_times, device=IMAGEs.device, dtype=torch.float32)  # [N_times]
    TIMEs_IDX_FLOOR = torch.clamp(torch.floor(TIMEs_IDX).long(), 0, N_frames - 1)  # [N_times]
    TIMEs_IDX_CEIL = torch.clamp(torch.ceil(TIMEs_IDX).long(), 0, N_frames - 1)  # [N_times]
    TIMEs_IDX_RESIDUAL = TIMEs_IDX - TIMEs_IDX_FLOOR.float()  # [N_times]
    TIME_STEPs = TIMEs_IDX / (N_frames - 1)  # [N_times]

    FRAMES_INTERPOLATED = IMAGEs[TIMEs_IDX_FLOOR] * (1 - TIMEs_IDX_RESIDUAL).view(-1, 1, 1) + IMAGEs[TIMEs_IDX_CEIL] * TIMEs_IDX_RESIDUAL.view(-1, 1, 1)
    return FRAMES_INTERPOLATED, TIME_STEPs


def get_points(RAYs_O: torch.Tensor, RAYs_D: torch.Tensor, near: float, far: float, N_depths: int, randomize: bool):
    T_VALs = torch.linspace(0., 1., steps=N_depths, device=RAYs_D.device, dtype=torch.float32)  # [N_depths]
    Z_VALs = near * torch.ones_like(RAYs_D[..., :1]) * (1. - T_VALs) + far * torch.ones_like(RAYs_D[..., :1]) * T_VALs  # [batch_size, N_depths]

    if randomize:
        MID_VALs = .5 * (Z_VALs[..., 1:] + Z_VALs[..., :-1])  # [batch_size, N_depths-1]
        UPPER_VALs = torch.cat([MID_VALs, Z_VALs[..., -1:]], -1)  # [batch_size, N_depths]
        LOWER_VALs = torch.cat([Z_VALs[..., :1], MID_VALs], -1)  # [batch_size, N_depths]
        T_RAND = torch.rand(Z_VALs.shape, device=RAYs_D.device, dtype=torch.float32)  # [batch_size, N_depths]
        Z_VALs = LOWER_VALs + (UPPER_VALs - LOWER_VALs) * T_RAND  # [batch_size, N_depths]

    DIST_VALs = Z_VALs[..., 1:] - Z_VALs[..., :-1]  # [batch_size, N_depths-1]
    POINTS = RAYs_O[..., None, :] + RAYs_D[..., None, :] * Z_VALs[..., :, None]  # [batch_size, N_depths, 3]
    return POINTS, DIST_VALs


## START
def _get_minibatch_jacobian(y, x):
    """Computes the Jacobian of y wrt x assuming minibatch-mode.
    Args:
      y: (N, ...) with a total of D_y elements in ...
      x: (N, ...) with a total of D_x elements in ...
    Returns:
      The minibatch Jacobian matrix of shape (N, D_y, D_x)
    """
    assert y.shape[0] == x.shape[0]
    y = y.view(y.shape[0], -1)
    # Compute Jacobian row by row.
    jac = []
    for j in range(y.shape[1]):
        dy_j_dx = torch.autograd.grad(
            y[:, j],
            x,
            torch.ones_like(y[:, j], device=y.get_device()),
            retain_graph=True,
            create_graph=True,
        )[0].view(x.shape[0], -1)

        jac.append(torch.unsqueeze(dy_j_dx, 1))
    jac = torch.cat(jac, 1)
    return jac


def PDE_EQs(D_t, D_x, D_y, D_z, U, F, U_t=None, U_x=None, U_y=None, U_z=None, detach=False):
    eqs = []
    dts = [D_t]
    dxs = [D_x]
    dys = [D_y]
    dzs = [D_z]

    F = torch.cat([torch.zeros_like(F[:, :1]), F], dim=1) * 0  # (N,4)
    u, v, w = U.split(1, dim=-1)  # (N,1)
    F_t, F_x, F_y, F_z = F.split(1, dim=-1)  # (N,1)
    dfs = [F_t, F_x, F_y, F_z]

    if None not in [U_t, U_x, U_y, U_z]:
        dts += U_t.split(1, dim=-1)  # [d_t, u_t, v_t, w_t] # (N,1)
        dxs += U_x.split(1, dim=-1)  # [d_x, u_x, v_x, w_x]
        dys += U_y.split(1, dim=-1)  # [d_y, u_y, v_y, w_y]
        dzs += U_z.split(1, dim=-1)  # [d_z, u_z, v_z, w_z]
    else:
        dfs = [F_t]

    for i, (dt, dx, dy, dz, df) in enumerate(zip(dts, dxs, dys, dzs, dfs)):
        if i == 0:
            _e = dt + (u * dx + v * dy + w * dz) + df
        else:
            if detach:
                _e = dt + (u.detach() * dx + v.detach() * dy + w.detach() * dz) + df
            else:
                _e = dt + (u * dx + v * dy + w * dz) + df
        eqs += [_e]

    if None not in [U_t, U_x, U_y, U_z]:
        # eqs += [ u_x + v_y + w_z ]
        eqs += [dxs[1] + dys[2] + dzs[3]]

    return eqs


def get_ray_pts_velocity_and_derivitives(
        pts,
        network_vel_fn,
        N_samples,
        **kwargs):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    if kwargs['no_vel_der']:
        vel_output, f_output = network_vel_fn(pts)
        ret = {}
        ret['raw_vel'] = vel_output
        ret['raw_f'] = f_output
        return ret

    def g(x):
        return model(x)[0]

    model = kwargs['network_fn']
    embed_fn = kwargs['embed_fn']
    h = embed_fn(pts)
    vel_output, f_output = model(h)
    ret = {}
    ret['raw_vel'] = vel_output
    ret['raw_f'] = f_output
    if not kwargs['no_vel_der']:
        jac = torch.vmap(torch.func.jacrev(g))(h)
        jac_x = _get_minibatch_jacobian(h, pts)
        jac = jac @ jac_x
        assert jac.shape == (pts.shape[0], 3, 4)
        _u_x, _u_y, _u_z, _u_t = [torch.squeeze(_, -1) for _ in jac.split(1, dim=-1)]  # (N,1)
        d = _u_x[:, 0] + _u_y[:, 1] + _u_z[:, 2]
        ret['raw_vel'] = vel_output
        ret['_u_x'] = _u_x
        ret['_u_y'] = _u_y
        ret['_u_z'] = _u_z
        ret['_u_t'] = _u_t

    return ret


def batchify_get_ray_pts_velocity_and_derivitive(pts, chunk=1024 * 64, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, pts.shape[0], chunk):
        ret = get_ray_pts_velocity_and_derivitives(pts[i:i + chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def get_velocity_and_derivitives(pts,
                                 **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      K: float. Focal length of pinhole camera.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """

    # Render and reshape
    all_ret = batchify_get_ray_pts_velocity_and_derivitive(pts, **kwargs)

    k_extract = ['raw_vel', 'raw_f'] if kwargs['no_vel_der'] else ['raw_vel', 'raw_f', '_u_x', '_u_y', '_u_z', '_u_t']
    ret_list = [all_ret[k] for k in k_extract]
    return ret_list


def mean_squared_error(pred, exact):
    if type(pred) is np.ndarray:
        return np.mean(np.square(pred - exact))
    return torch.mean(torch.square(pred - exact))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    val = 0
    avg = 0
    sum = 0
    count = 0
    tot_count = 0

    def __init__(self):
        self.reset()
        self.tot_count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.tot_count += n
        self.avg = self.sum / self.count


## END

def get_raw(POINTS_TIME: torch.Tensor, DISTs: torch.Tensor, RAYs_D_FLAT: torch.Tensor, BBOX_MODEL, MODEL, ENCODER):
    assert POINTS_TIME.dim() == 3 and POINTS_TIME.shape[-1] == 4
    assert POINTS_TIME.shape[0] == DISTs.shape[0] == RAYs_D_FLAT.shape[0]
    POINTS_TIME_FLAT = POINTS_TIME.view(-1, POINTS_TIME.shape[-1])  # [batch_size * N_depths, 4]
    out_dim = 1
    bbox_mask = BBOX_MODEL.insideMask(POINTS_TIME_FLAT[..., :3], to_float=False)
    if bbox_mask.sum() == 0:
        bbox_mask[0] = True
        assert False
    POINTS_TIME_FLAT_FINAL = POINTS_TIME_FLAT[bbox_mask]

    ## START
    POINTS_TIME_FLAT_FINAL.requires_grad = True

    def g(x):
        return MODEL(x)

    h = ENCODER(POINTS_TIME_FLAT_FINAL)
    raw_d = MODEL(h)
    jac = torch.vmap(torch.func.jacrev(g))(h)
    jac_x = _get_minibatch_jacobian(h, POINTS_TIME_FLAT_FINAL)
    jac = jac @ jac_x
    ret = {'raw_d': raw_d, 'pts': POINTS_TIME_FLAT_FINAL}
    _d_x, _d_y, _d_z, _d_t = [torch.squeeze(_, -1) for _ in jac.split(1, dim=-1)]
    ret['_d_x'] = _d_x
    ret['_d_y'] = _d_y
    ret['_d_z'] = _d_z
    ret['_d_t'] = _d_t
    ## END

    RAW_FLAT = torch.zeros([POINTS_TIME_FLAT.shape[0], out_dim], device=POINTS_TIME_FLAT.device, dtype=torch.float32)
    RAW_FLAT[bbox_mask] = raw_d
    RAW = RAW_FLAT.reshape(*POINTS_TIME.shape[:-1], out_dim)
    assert RAW.dim() == 3 and RAW.shape[-1] == 1

    DISTs_cat = torch.cat([DISTs, torch.tensor([1e10], device=DISTs.device).expand(DISTs[..., :1].shape)], -1)  # [batch_size, N_depths]
    DISTS_final = DISTs_cat * torch.norm(RAYs_D_FLAT[..., None, :], dim=-1)  # [batch_size, N_depths]

    RGB_TRAINED = torch.ones(3, device=POINTS_TIME.device) * (0.6 + torch.tanh(MODEL.rgb) * 0.4)
    raw2alpha = lambda raw, dists, act_fn=torch.nn.functional.relu: 1. - torch.exp(-act_fn(raw) * dists)
    noise = 0.
    alpha = raw2alpha(RAW[..., -1] + noise, DISTS_final)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=device), 1. - alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[..., None] * RGB_TRAINED, -2)
    return rgb_map, _d_x, _d_y, _d_z, _d_t, POINTS_TIME_FLAT_FINAL, raw_d


if __name__ == '__main__':
    device = torch.device("cuda")
    ti.init(arch=ti.cuda, device_memory_GB=12.0)
    np.random.seed(0)

    args_npz = np.load("args.npz", allow_pickle=True)
    from types import SimpleNamespace

    args = SimpleNamespace(**{
        key: value.item() if isinstance(value, np.ndarray) and value.size == 1 else
        value.tolist() if isinstance(value, np.ndarray) else
        value
        for key, value in args_npz.items()
    })

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)

    pinf_data = np.load("train_dataset.npz")
    IMAGE_TRAIN_np = pinf_data['images_train']
    POSES_TRAIN_np = pinf_data['poses_train']
    HWF_np = pinf_data['hwf']
    RENDER_POSE_np = pinf_data['render_poses']
    RENDER_TIMESTEPs_np = pinf_data['render_timesteps']

    NEAR_float = pinf_data['near'].item()
    FAR_float = pinf_data['far'].item()
    H_int = int(HWF_np[0])
    W_int = int(HWF_np[1])
    FOCAL_float = float(HWF_np[2])
    K = np.array([[FOCAL_float, 0, 0.5 * W_int], [0, FOCAL_float, 0.5 * H_int], [0, 0, 1]])

    ############################## Load Encoder ##############################
    from src.encoder import HashEncoderHyFluid

    max_res = np.array([args.finest_resolution, args.finest_resolution, args.finest_resolution, args.finest_resolution_t])
    min_res = np.array([args.base_resolution, args.base_resolution, args.base_resolution, args.base_resolution_t])
    ENCODER_gpu = HashEncoderHyFluid(max_res=max_res, min_res=min_res, num_scales=args.num_levels, max_params=2 ** args.log2_hashmap_size).to(device)
    max_res_v = np.array([args.finest_resolution_v, args.finest_resolution_v, args.finest_resolution_v, args.finest_resolution_v_t])
    min_res_v = np.array([args.base_resolution_v, args.base_resolution_v, args.base_resolution_v, args.base_resolution_v_t])
    ENCODER_v_gpu = HashEncoderHyFluid(max_res=max_res_v, min_res=min_res_v, num_scales=args.num_levels, max_params=2 ** args.log2_hashmap_size).to(device)
    ############################## Load Encoder ##############################

    ############################## Load Model ##############################
    MODEL_gpu = mmodel.NeRFSmall(num_layers=2,
                                 hidden_dim=64,
                                 geo_feat_dim=15,
                                 num_layers_color=2,
                                 hidden_dim_color=16,
                                 input_ch=ENCODER_gpu.num_scales * 2).to(device)
    MODEL_v_gpu = mmodel.NeRFSmallPotential(num_layers=args.vel_num_layers,
                                            hidden_dim=64,
                                            geo_feat_dim=15,
                                            num_layers_color=2,
                                            hidden_dim_color=16,
                                            input_ch=ENCODER_v_gpu.num_scales * 2,
                                            use_f=args.use_f).to(device)
    ############################## Load Model ##############################

    ############################## Load Optimizer ##############################
    OPTIMIZER = radam.RAdam([
        {'params': MODEL_gpu.parameters(), 'weight_decay': 1e-6},
        {'params': ENCODER_gpu.parameters(), 'eps': 1e-15}
    ], lr=args.lrate_den, betas=(0.9, 0.99))
    OPTIMIZER_v = torch.optim.RAdam([
        {'params': MODEL_v_gpu.parameters(), 'weight_decay': 1e-6},
        {'params': ENCODER_v_gpu.parameters(), 'eps': 1e-15}
    ], lr=args.lrate, betas=(0.9, 0.99))
    ############################## Load Optimizer ##############################

    ############################## Load BoundingBox ##############################
    import src.bbox as bbox

    VOXEL_TRAN_np = pinf_data['voxel_tran']
    VOXEL_SCALE_np = pinf_data['voxel_scale']
    voxel_tran_inv = np.linalg.inv(VOXEL_TRAN_np)
    BBOX_MODEL_gpu = bbox.BBox_Tool(voxel_tran_inv, VOXEL_SCALE_np)
    ############################## Load BoundingBox ##############################

    network_query_fn = lambda x: MODEL_gpu(ENCODER_gpu(x))
    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_samples': args.N_samples,
        'network_fn': MODEL_gpu,
        'embed_fn': ENCODER_gpu,
        'near': NEAR_float,
        'far': FAR_float,
    }


    def network_vel_fn(x):
        with torch.enable_grad():
            if not args.no_vel_der:
                h = ENCODER_v_gpu(x)
                v, f = MODEL_v_gpu(h)
                return v, f, h
            else:
                v, f = MODEL_v_gpu(ENCODER_v_gpu(x))
                return v, f


    render_kwargs_train_vel = {
        'network_vel_fn': network_vel_fn,
        'perturb': args.perturb,
        'N_samples': args.N_samples,
        'network_fn': MODEL_v_gpu,
        'embed_fn': ENCODER_v_gpu,
        'near': NEAR_float,
        'far': FAR_float,
    }

    import tqdm
    import os

    batch_size = 256
    time_size = 1
    depth_size = 192
    global_step = 1

    lrate = 0.01
    lrate_decay = 100000

    loss_meter, psnr_meter = AverageMeter(), AverageMeter()
    flow_loss_meter, scale_meter, norm_meter = AverageMeter(), AverageMeter(), AverageMeter()
    u_loss_meter, v_loss_meter, w_loss_meter, d_loss_meter = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    proj_loss_meter = AverageMeter()
    den2vel_loss_meter = AverageMeter()
    vel_loss_meter = AverageMeter()

    ## START
    rx, ry, rz, proj_y, use_project, y_start = args.sim_res_x, args.sim_res_y, args.sim_res_z, args.proj_y, args.use_project, args.y_start

    xs, ys, zs = torch.meshgrid([torch.linspace(0, 1, rx, device=device), torch.linspace(0, 1, ry, device=device), torch.linspace(0, 1, rz, device=device)], indexing='ij')
    boundary_types = ti.Matrix([[1, 1], [2, 1], [1, 1]], ti.i32)  # boundaries: 1 means Dirichlet, 2 means Neumann
    project_solver = MGPCG_3(boundary_types=boundary_types, N=[rx, proj_y, rz], base_level=3)
    coord_3d_sim = torch.stack([xs, ys, zs], dim=-1)  # [X, Y, Z, 3]
    coord_3d_world = BBOX_MODEL_gpu.sim2world(coord_3d_sim)  # [X, Y, Z, 3]
    ## END

    for ITERATION in range(1, 2):
        IMAGE_TRAIN_gpu, RAYs_gpu, RAY_IDX_gpu = do_resample_rays(H_int, W_int)
        for i in tqdm.trange(0, RAY_IDX_gpu.shape[0], batch_size):
            BATCH_RAYs_O_gpu, BATCH_RAYs_D_gpu, BATCH_RAYs_IDX_gpu = get_ray_batch(RAYs_gpu, RAY_IDX_gpu, i, i + batch_size)  # [batch_size, 3], [batch_size, 3], [batch_size]
            FRAMES_INTERPOLATED_gpu, TIME_STEPs_gpu = get_frames_at_times(IMAGE_TRAIN_gpu, IMAGE_TRAIN_gpu.shape[0], time_size)  # [N_times, N x H x W, 3], [N_times]
            TARGET_S_gpu = FRAMES_INTERPOLATED_gpu[:, BATCH_RAYs_IDX_gpu].flatten(0, 1)  # [batch_size * N_times, 3]

            OPTIMIZER.zero_grad()
            OPTIMIZER_v.zero_grad()

            POINTS_gpu, DISTs_gpu = get_points(BATCH_RAYs_O_gpu, BATCH_RAYs_D_gpu, NEAR_float, FAR_float, depth_size, randomize=True)  # [batch_size, N_depths, 3]
            for TIME_STEP_gpu in TIME_STEPs_gpu:
                POINTS_TIME_gpu = torch.cat([POINTS_gpu, TIME_STEP_gpu.expand(POINTS_gpu[..., :1].shape)], dim=-1)  # [batch_size, N_depths, 4]

                ## START
                RGB_MAP, _d_x, _d_y, _d_z, _d_t, pts, raw_d = get_raw(POINTS_TIME_gpu, DISTs_gpu, BATCH_RAYs_D_gpu, BBOX_MODEL_gpu, MODEL_gpu, ENCODER_gpu)
                if args.no_vel_der:
                    raw_vel, raw_f = get_velocity_and_derivitives(pts, no_vel_der=True, **render_kwargs_train_vel)
                    _u_x, _u_y, _u_z, _u_t = None, None, None, None
                else:
                    raw_vel, raw_f, _u_x, _u_y, _u_z, _u_t = get_velocity_and_derivitives(pts, no_vel_der=False, **render_kwargs_train_vel)
                split_nse = PDE_EQs(_d_t, _d_x, _d_y, _d_z, raw_vel, raw_f, _u_t, _u_x, _u_y, _u_z, detach=args.detach_vel)
                nse_errors = [mean_squared_error(x, 0.0) for x in split_nse]
                if torch.stack(nse_errors).sum() > 10000:
                    print(f'skip large loss {torch.stack(nse_errors).sum():.3g}, timestep={pts[0, 3]}')
                    continue
                nseloss_fine = 0.0
                split_nse_wei = [args.flow_weight, args.vel_weight, args.vel_weight, args.vel_weight, args.d_weight] if not args.no_vel_der else [args.flow_weight]

                img2mse = lambda x, y: torch.mean((x - y) ** 2)
                img_loss = img2mse(RGB_MAP, TARGET_S_gpu)
                loss_meter.update(img_loss.item())
                flow_loss_meter.update(split_nse_wei[0] * nse_errors[0].item())
                scale_meter.update(nse_errors[-1].item())
                norm_meter.update((split_nse_wei[-1] * nse_errors[-1]).item())
                if not args.no_vel_der:
                    u_loss_meter.update((nse_errors[1]).item())
                    v_loss_meter.update((nse_errors[2]).item())
                    w_loss_meter.update((nse_errors[3]).item())
                    d_loss_meter.update((nse_errors[4]).item())
                for ei, wi in zip(nse_errors, split_nse_wei):
                    nseloss_fine = ei * wi + nseloss_fine

                if args.proj_weight > 0:
                    # initialize density field
                    coord_time_step = torch.ones_like(coord_3d_world[..., :1]) * TIME_STEP_gpu
                    coord_4d_world = torch.cat([coord_3d_world, coord_time_step], dim=-1)  # [X, Y, Z, 4]
                    vel_world = batchify_query(coord_4d_world, render_kwargs_train_vel['network_vel_fn'])  # [X, Y, Z, 3]
                    # y_start = args.y_start
                    vel_world_supervised = vel_world.detach().clone()
                    # vel_world_supervised[:, y_start:y_start + proj_y] = project_solver.Poisson(
                    #     vel_world_supervised[:, y_start:y_start + proj_y])

                    vel_world_supervised[..., 2] *= -1
                    vel_world_supervised[:, y_start:y_start + proj_y] = project_solver.Poisson(
                        vel_world_supervised[:, y_start:y_start + proj_y])
                    vel_world_supervised[..., 2] *= -1

                    proj_loss = img2mse(vel_world_supervised, vel_world)
                else:
                    proj_loss = torch.zeros_like(img_loss)

                if args.d2v_weight > 0:
                    viz_dens_mask = raw_d.detach() > 0.1
                    vel_norm = raw_vel.norm(dim=-1, keepdim=True)
                    min_vel_mask = vel_norm.detach() < args.coef_den2vel * raw_d.detach()
                    vel_reg_mask = min_vel_mask & viz_dens_mask
                    min_vel_reg_map = (args.coef_den2vel * raw_d - vel_norm) * vel_reg_mask.float()
                    min_vel_reg = min_vel_reg_map.pow(2).mean()
                    # ipdb.set_trace()
                else:
                    min_vel_reg = torch.zeros_like(img_loss)

                proj_loss_meter.update(proj_loss.item())
                den2vel_loss_meter.update(min_vel_reg.item())

                vel_loss = nseloss_fine + args.rec_weight * img_loss + args.proj_weight * proj_loss + args.d2v_weight * min_vel_reg
                vel_loss_meter.update(vel_loss.item())
                vel_loss.backward()

                OPTIMIZER.step()
                OPTIMIZER_v.step()

                decay_rate = 0.1
                decay_steps = lrate_decay
                new_lrate = lrate * (decay_rate ** (global_step / decay_steps))
                for param_group in OPTIMIZER.param_groups:
                    param_group['lr'] = new_lrate
                for param_group in OPTIMIZER_v.param_groups:
                    param_group['lr'] = new_lrate

                if i % args.i_print == 0:
                    tqdm.tqdm.write(
                        f"[TRAIN] Iter: {i} Rec Loss:{loss_meter.avg:.2g} PSNR:{psnr_meter.avg:.4g} Flow Loss: {flow_loss_meter.avg:.2g}, "
                        f"U loss: {u_loss_meter.avg:.2g}, V loss: {v_loss_meter.avg:.2g}, W loss: {w_loss_meter.avg:.2g},"
                        f" d loss: {d_loss_meter.avg:.2g}, proj Loss:{proj_loss_meter.avg:.2g}, den2vel loss:{den2vel_loss_meter.avg:.2g}, Vel Loss: {vel_loss_meter.avg:.2g} ")
                    loss_meter.reset()
                    psnr_meter.reset()
                    flow_loss_meter.reset()
                    scale_meter.reset()
                    vel_loss_meter.reset()
                    norm_meter.reset()
                    u_loss_meter.reset()
                    v_loss_meter.reset()
                    w_loss_meter.reset()
                    d_loss_meter.reset()
                ## END

                global_step += 1

        os.makedirs("checkpoint", exist_ok=True)
        path = os.path.join("checkpoint", 'den_{:06d}.tar'.format(ITERATION))
        torch.save({
            'global_step': global_step,
            'network_fn_state_dict': MODEL_gpu.state_dict(),
            'embed_fn_state_dict': ENCODER_gpu.state_dict(),
            'optimizer_state_dict': OPTIMIZER.state_dict(),
        }, path)
        path = os.path.join("checkpoint", 'vel_{:06d}.tar'.format(ITERATION))
        torch.save({
            'global_step': global_step,
            'network_fn_state_dict': MODEL_v_gpu.state_dict(),
            'embed_fn_state_dict': ENCODER_v_gpu.state_dict(),
            'optimizer_state_dict': OPTIMIZER_v.state_dict(),
        }, path)
