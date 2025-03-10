import torch


def advect_SL(q_grid, vel_world_prev, coord_3d_sim, dt, RK=2, y_start=48, proj_y=128,
              use_project=False, project_solver=None, bbox_model=None, **kwargs):
    """Advect a scalar quantity using a given velocity field.
    Args:
        q_grid: [X', Y', Z', C]
        vel_world_prev: [X, Y, Z, 3]
        coord_3d_sim: [X, Y, Z, 3]
        dt: float
        RK: int, number of Runge-Kutta steps
        y_start: where to start at y-axis
        proj_y: simulation domain resolution at y-axis
        use_project: whether to use Poisson solver
        project_solver: Poisson solver
        bbox_model: bounding box model
    Returns:
        advected_quantity: [X, Y, Z, 1]
        vel_world: [X, Y, Z, 3]
    """
    if RK == 1:
        vel_world = vel_world_prev.clone()
        vel_world[:, y_start:y_start + proj_y] = project_solver.Poisson(vel_world[:, y_start:y_start + proj_y]) if use_project else vel_world[:, y_start:y_start + proj_y]
        vel_sim = bbox_model.world2sim_rot(vel_world)  # [X, Y, Z, 3]
    elif RK == 2:
        vel_world = vel_world_prev.clone()  # [X, Y, Z, 3]
        vel_world[:, y_start:y_start + proj_y] = project_solver.Poisson(vel_world[:, y_start:y_start + proj_y]) if use_project else vel_world[:, y_start:y_start + proj_y]
        # breakpoint()
        vel_sim = bbox_model.world2sim_rot(vel_world)  # [X, Y, Z, 3]
        coord_3d_sim_midpoint = coord_3d_sim - 0.5 * dt * vel_sim  # midpoint
        midpoint_sampled = coord_3d_sim_midpoint * 2 - 1  # [X, Y, Z, 3]
        vel_sim = torch.nn.functional.grid_sample(vel_sim.permute(3, 2, 1, 0)[None], midpoint_sampled.permute(2, 1, 0, 3)[None], align_corners=True, padding_mode='zeros').squeeze(0).permute(3, 2, 1, 0)  # [X, Y, Z, 3]
    else:
        raise NotImplementedError
    backtrace_coord = coord_3d_sim - dt * vel_sim  # [X, Y, Z, 3]
    backtrace_coord_sampled = backtrace_coord * 2 - 1  # ranging [-1, 1]
    q_grid = q_grid[None, ...].permute([0, 4, 3, 2, 1])  # [N, C, Z, Y, X] i.e., [N, C, D, H, W]
    q_backtraced = torch.nn.functional.grid_sample(q_grid, backtrace_coord_sampled.permute(2, 1, 0, 3)[None, ...], align_corners=True, padding_mode='zeros')  # [N, C, D, H, W]
    q_backtraced = q_backtraced.squeeze(0).permute([3, 2, 1, 0])  # [X, Y, Z, C]
    return q_backtraced, vel_world


def advect_maccormack(q_grid, vel_world_prev, coord_3d_sim, dt, **kwargs):
    """
    Args:
        q_grid: [X', Y', Z', C]
        vel_world_prev: [X, Y, Z, 3]
        coord_3d_sim: [X, Y, Z, 3]
        dt: float
    Returns:
        advected_quantity: [X, Y, Z, C]
        vel_world: [X, Y, Z, 3]
    """
    q_grid_next, _ = advect_SL(q_grid, vel_world_prev, coord_3d_sim, dt, **kwargs)
    q_grid_back, vel_world = advect_SL(q_grid_next, vel_world_prev, coord_3d_sim, -dt, **kwargs)
    q_advected = q_grid_next + (q_grid - q_grid_back) / 2
    C = q_advected.shape[-1]
    for i in range(C):
        q_max, q_min = q_grid[..., i].max(), q_grid[..., i].min()
        q_advected[..., i] = q_advected[..., i].clamp_(q_min, q_max)
    return q_advected, vel_world
