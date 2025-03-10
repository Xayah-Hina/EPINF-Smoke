import numpy as np
import torch
import src.model as mmodel
import src.mgpcg as mmgpcg
import src.advect as madvect

import taichi as ti
import os


def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W, device=device), torch.linspace(0, H - 1, H, device=device),
                          indexing='ij')  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3],
                       -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


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


def get_raw2(POINTS_TIME: torch.Tensor, DISTs: torch.Tensor, RAYs_D_FLAT: torch.Tensor, BBOX_MODEL_gpu, MODEL, ENCODER):
    assert POINTS_TIME.dim() == 3 and POINTS_TIME.shape[-1] == 4
    assert POINTS_TIME.shape[0] == DISTs.shape[0] == RAYs_D_FLAT.shape[0]
    POINTS_TIME_FLAT = POINTS_TIME.view(-1, POINTS_TIME.shape[-1])  # [batch_size * N_depths, 4]
    out_dim = 1
    RAW_FLAT = torch.zeros([POINTS_TIME_FLAT.shape[0], out_dim], device=POINTS_TIME_FLAT.device, dtype=torch.float32)
    bbox_mask = BBOX_MODEL_gpu.insideMask(POINTS_TIME_FLAT[..., :3], to_float=False)
    if bbox_mask.sum() == 0:
        bbox_mask[0] = True
        assert False
    POINTS_TIME_FLAT_FINAL = POINTS_TIME_FLAT[bbox_mask]

    chunk = 512 * 64
    ret_list = []
    for _ in range(0, POINTS_TIME_FLAT_FINAL.shape[0], chunk):
        ret = MODEL(ENCODER(POINTS_TIME_FLAT_FINAL[_:min(_ + chunk, POINTS_TIME_FLAT_FINAL.shape[0])]))
        ret_list.append(ret)
    all_ret = torch.cat(ret_list, 0)
    RAW_FLAT[bbox_mask] = all_ret
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
    return rgb_map


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


if __name__ == '__main__':
    device = torch.device("cuda")
    ti.init(arch=ti.cuda, device_memory_GB=12.0)
    np.random.seed(0)

    ############################## Load Args ##############################
    args_npz = np.load("args.npz", allow_pickle=True)
    from types import SimpleNamespace

    args = SimpleNamespace(**{
        key: value.item() if isinstance(value, np.ndarray) and value.size == 1 else
        value.tolist() if isinstance(value, np.ndarray) else
        value
        for key, value in args_npz.items()
    })
    ############################## Load Args ##############################

    ############################## Load Data ##############################
    pinf_data_test = np.load("test_dataset.npz")
    IMAGE_TEST_np = pinf_data_test['images_test']
    POSES_TEST_np = pinf_data_test['poses_test']
    HWF_np = pinf_data_test['hwf']
    VOXEL_TRAN_np = pinf_data_test['voxel_tran']
    VOXEL_SCALE_np = pinf_data_test['voxel_scale']
    NEAR_float = pinf_data_test['near'].item()
    FAR_float = pinf_data_test['far'].item()
    H = int(HWF_np[0])
    W = int(HWF_np[1])
    FOCAL = float(HWF_np[2])
    K = np.array([[FOCAL, 0, 0.5 * W], [0, FOCAL, 0.5 * H], [0, 0, 1]])
    ############################## Load Data ##############################

    test_view_pose = torch.tensor(POSES_TEST_np[0], device=device, dtype=torch.float32)
    N_timesteps = IMAGE_TEST_np.shape[0]
    test_timesteps = torch.arange(N_timesteps, device=device) / (N_timesteps - 1)
    rays_o, rays_d = get_rays(H, W, K, test_view_pose)
    points, dists = get_points(rays_o, rays_d, NEAR_float, FAR_float, 192, randomize=False)

    points_flat = points.flatten(0, 1)
    dists_flat = dists.flatten(0, 1)
    rays_d_flat = rays_d.flatten(0, 1)

    ############################## Load Encoder ##############################
    ckpt_d = torch.load("checkpoint/den_000007.tar")
    ckpt_v = torch.load("checkpoint/vel_000007.tar")
    ############################## Load Encoder ##############################

    ############################## Load Encoder ##############################
    from src.encoder import HashEncoderHyFluid

    max_res = np.array([args.finest_resolution, args.finest_resolution, args.finest_resolution, args.finest_resolution_t])
    min_res = np.array([args.base_resolution, args.base_resolution, args.base_resolution, args.base_resolution_t])
    ENCODER_gpu = HashEncoderHyFluid(max_res=max_res, min_res=min_res, num_scales=args.num_levels, max_params=2 ** args.log2_hashmap_size).to(device)
    ENCODER_gpu.load_state_dict(ckpt_d['embed_fn_state_dict'])
    max_res_v = np.array([args.finest_resolution_v, args.finest_resolution_v, args.finest_resolution_v, args.finest_resolution_v_t])
    min_res_v = np.array([args.base_resolution_v, args.base_resolution_v, args.base_resolution_v, args.base_resolution_v_t])
    ENCODER_v_gpu = HashEncoderHyFluid(max_res=max_res_v, min_res=min_res_v, num_scales=args.num_levels, max_params=2 ** args.log2_hashmap_size).to(device)
    ENCODER_v_gpu.load_state_dict(ckpt_v['embed_fn_state_dict'])
    ############################## Load Encoder ##############################

    ############################## Load Model ##############################
    MODEL_gpu = mmodel.NeRFSmall(num_layers=2,
                                 hidden_dim=64,
                                 geo_feat_dim=15,
                                 num_layers_color=2,
                                 hidden_dim_color=16,
                                 input_ch=ENCODER_gpu.num_scales * 2).to(device)
    MODEL_gpu.load_state_dict(ckpt_d['network_fn_state_dict'])
    MODEL_v_gpu = mmodel.NeRFSmallPotential(num_layers=args.vel_num_layers,
                                            hidden_dim=64,
                                            geo_feat_dim=15,
                                            num_layers_color=2,
                                            hidden_dim_color=16,
                                            input_ch=ENCODER_v_gpu.num_scales * 2,
                                            use_f=args.use_f).to(device)
    MODEL_v_gpu.load_state_dict(ckpt_v['network_fn_state_dict'])
    ############################## Load Model ##############################

    ############################## Load BoundingBox ##############################
    import src.bbox as bbox

    voxel_tran_inv = np.linalg.inv(VOXEL_TRAN_np)
    BBOX_MODEL_gpu = bbox.BBox_Tool(voxel_tran_inv, VOXEL_SCALE_np)
    ############################## Load BoundingBox ##############################

    import tqdm
    import imageio

    os.makedirs("output_v", exist_ok=True)
    with torch.no_grad():
        for i in tqdm.trange(0, test_timesteps.shape[0]):
            test_timesteps_expended = test_timesteps[i].expand(points_flat[..., :1].shape)
            points_time_flat_gpu = torch.cat([points_flat, test_timesteps_expended], dim=-1)
            rgb_map_flat = get_raw2(points_time_flat_gpu, dists_flat, rays_d_flat, BBOX_MODEL_gpu, MODEL_gpu, ENCODER_gpu)
            rgb_map = rgb_map_flat.view(points.shape[0], points.shape[1], rgb_map_flat.shape[-1])
            to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
            rgb8 = to8b(rgb_map.cpu().numpy())
            imageio.imsave(os.path.join("output_v", 'rgb_{:03d}.png'.format(i)), rgb8)

    with torch.no_grad():
        rx, ry, rz, proj_y, use_project, y_start = args.sim_res_x, args.sim_res_y, args.sim_res_z, args.proj_y, args.use_project, args.y_start
        xs, ys, zs = torch.meshgrid([torch.linspace(0, 1, rx, device=device), torch.linspace(0, 1, ry, device=device), torch.linspace(0, 1, rz, device=device)], indexing='ij')
        boundary_types = ti.Matrix([[1, 1], [2, 1], [1, 1]], ti.i32)  # boundaries: 1 means Dirichlet, 2 means Neumann
        project_solver = mmgpcg.MGPCG_3(boundary_types=boundary_types, N=[rx, proj_y, rz], base_level=3)
        coord_3d_sim = torch.stack([xs, ys, zs], dim=-1)  # [X, Y, Z, 3]
        coord_3d_world = BBOX_MODEL_gpu.sim2world(coord_3d_sim)  # [X, Y, Z, 3]
        dt = test_timesteps[1] - test_timesteps[0]

        ############################## Get Source Density ##############################

        for i in tqdm.trange(0, test_timesteps.shape[0]):
            time_step = torch.ones_like(coord_3d_world[..., :1]) * test_timesteps[i]
            coord_4d_world = torch.cat([coord_3d_world, time_step], dim=-1)
            network_query_fn = lambda x: MODEL_gpu(ENCODER_gpu(x))
            den = batchify_query(coord_4d_world, network_query_fn)
            network_query_fn_vel = lambda x: MODEL_v_gpu(ENCODER_v_gpu(x))
            vel = batchify_query(coord_4d_world, network_query_fn_vel)  # [X, Y, Z, 3]
            os.makedirs("output_grid", exist_ok=True)
            np.savez_compressed("output_grid/den_vel_{:03d}.npz".format(i), den=den.cpu().numpy(), vel=vel.cpu().numpy())










#         time_step = torch.ones_like(coord_3d_world[..., :1]) * test_timesteps[0]
#         coord_4d_world = torch.cat([coord_3d_world, time_step], dim=-1)  # [X, Y, Z, 4]
#         network_query_fn = lambda x: MODEL_gpu(ENCODER_gpu(x))
#         den_source = batchify_query(coord_4d_world, network_query_fn)
#         network_query_fn_vel = lambda x: MODEL_v_gpu(ENCODER_v_gpu(x))
#         vel_source = batchify_query(coord_4d_world, network_query_fn_vel)  # [X, Y, Z, 3]
#
#         source_height = 0.25
#         y_start = int(source_height * ry)
#         den_cur = den_source.clone()
#         for frame in tqdm.trange(1, test_timesteps.shape[0]):
#             mask_to_sim = coord_3d_sim[..., 1] > source_height
#             coord_4d_world[..., 3] = test_timesteps[frame - 1]
#             vel_cur = batchify_query(coord_4d_world, network_query_fn_vel)  # [X, Y, Z, 3]
#             den_cur, vel_cur = madvect.advect_maccormack(q_grid=den_cur, vel_world_prev=vel_cur, coord_3d_sim=coord_3d_world, dt=dt, y_start=y_start, proj_y=proj_y, use_project=use_project, project_solver=project_solver, bbox_model=BBOX_MODEL_gpu)
#
# # TODO: