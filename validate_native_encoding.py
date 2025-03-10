import torch
import numpy as np
import imageio.v2 as imageio
import tqdm
import os

import src.bbox as bbox
import src.encoder2 as encoder
import src.model as model


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


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device("cuda")
    ###################################################################################
    pinf_data_test = np.load("data/test_dataset.npz")
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

    test_view_pose = torch.tensor(POSES_TEST_np[0], device=device, dtype=torch.float32)
    N_timesteps = IMAGE_TEST_np.shape[0]
    test_timesteps = torch.arange(N_timesteps, device=device) / (N_timesteps - 1)
    rays_o, rays_d = get_rays(H, W, K, test_view_pose)
    points, dists = get_points(rays_o, rays_d, NEAR_float, FAR_float, 192, randomize=False)

    points_flat = points.flatten(0, 1)
    dists_flat = dists.flatten(0, 1)
    rays_d_flat = rays_d.flatten(0, 1)
    ###################################################################################
    ENCODER = encoder.HashEncoderNative(device=device).to(device)

    MODEL = model.NeRFSmall(num_layers=2,
                            hidden_dim=64,
                            geo_feat_dim=15,
                            num_layers_color=2,
                            hidden_dim_color=16,
                            input_ch=ENCODER.num_levels * 2).to(device)

    voxel_tran_inv = np.linalg.inv(VOXEL_TRAN_np)
    BBOX_MODEL_gpu = bbox.BBox_Tool(voxel_tran_inv, VOXEL_SCALE_np, [0.15, 0.0, 0.15], [0.85, 1., 0.85], device=device)
    ###################################################################################
    ckpts = [os.path.join("checkpoint", f) for f in sorted(os.listdir("checkpoint")) if 'tar' in f]
    ckpt_path = ckpts[-1]
    ckpt = torch.load(ckpt_path)
    MODEL.load_state_dict(ckpt['network_fn_state_dict'])
    ENCODER.load_state_dict(ckpt['embed_fn_state_dict'])
    ###################################################################################
    os.makedirs("output", exist_ok=True)
    with torch.no_grad():
        for i in tqdm.trange(0, test_timesteps.shape[0]):
            test_timesteps_expended = test_timesteps[i].expand(points_flat[..., :1].shape)
            points_time_flat_gpu = torch.cat([points_flat, test_timesteps_expended], dim=-1)
            rgb_map_flat = get_raw2(points_time_flat_gpu, dists_flat, rays_d_flat, BBOX_MODEL_gpu, MODEL, ENCODER)
            rgb_map = rgb_map_flat.view(points.shape[0], points.shape[1], rgb_map_flat.shape[-1])
            to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
            rgb8 = to8b(rgb_map.cpu().numpy())
            imageio.imsave(os.path.join("output", 'rgb_{:03d}.png'.format(i)), rgb8)
