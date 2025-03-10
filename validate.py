import torch
import numpy as np
import imageio.v2 as imageio
import os
import sys
import tqdm

sys.path.append(work_item.attrib("cwd")[0])
os.makedirs(os.path.join(work_item.attrib("output")[0], "iter{:06d}".format(int(work_item.index)), exist_ok=True))

device = torch.device("cuda")

ckpt_path = os.path.join(work_item.attrib("checkpoint")[0], '{:06d}.tar'.format(int(work_item.index)))
ckpt = torch.load(ckpt_path)

pinf_data_test = np.load("data/test_dataset.npz")
IMAGE_TEST_np = pinf_data_test['images_test']
POSES_TEST_np = pinf_data_test['poses_test']
HWF_np = pinf_data_test['hwf']
NEAR_float = pinf_data_test['near'].item()
FAR_float = pinf_data_test['far'].item()
H_int = int(HWF_np[0])
W_int = int(HWF_np[1])
FOCAL_float = float(HWF_np[2])
VOXEL_TRAN_np = pinf_data_test['voxel_tran']
VOXEL_SCALE_np = pinf_data_test['voxel_scale']
depth_size = 192

############################## Load BoundingBox ##############################
import src.bbox as bbox
voxel_tran_inv = np.linalg.inv(VOXEL_TRAN_np)
BBOX_MODEL_gpu = bbox.BBox_Tool(voxel_tran_inv, VOXEL_SCALE_np)
############################## Load BoundingBox ##############################

############################## Load Encoder ##############################
import src.encoder2 as encoder
ENCODER_gpu = encoder.HashEncoderNative(device=device).to(device)
ENCODER_gpu.load_state_dict(ckpt['embed_fn_state_dict'])
############################## Load Encoder ##############################

############################## Load Model ##############################
import src.model as model
MODEL_gpu = model.NeRFSmall(num_layers=2,
                            hidden_dim=64,
                            geo_feat_dim=15,
                            num_layers_color=2,
                            hidden_dim_color=16,
                            input_ch=ENCODER_gpu.num_levels * 2).to(device)
MODEL_gpu.load_state_dict(ckpt['network_fn_state_dict'])
############################## Load Model ##############################



def generate_test_rays(c2w, W, H, FOCAL):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([
        (i - 0.5 * W) / FOCAL,
        -(j - 0.5 * H) / FOCAL,
        -np.ones_like(i)
    ], axis=-1)
    rays_d = dirs @ c2w[:3, :3].T
    rays_o = c2w[:3, -1]
    return rays_d, rays_o

rays_d_np, rays_o_np = generate_test_rays(POSES_TEST_np[0], W_int, H_int, FOCAL_float)
RAYs_D_gpu = torch.tensor(rays_d_np, device=device, dtype=torch.float32) # (960, 540, 3)
RAYs_O_gpu = torch.tensor(rays_o_np, device=device, dtype=torch.float32).expand(RAYs_D_gpu.shape) # (960, 540, 3)

T_VALs = torch.linspace(0., 1., steps=depth_size, device=device, dtype=torch.float32) # (depth_size)
Z_VALs = NEAR_float * torch.ones_like(RAYs_D_gpu[..., :1]) * (1. - T_VALs) + FAR_float * torch.ones_like(RAYs_D_gpu[..., :1]) * T_VALs # (H, W, depth_size)
DISTs_gpu = Z_VALs[..., 1:] - Z_VALs[..., :-1]  # (H, W, depth_size-1)
POINTs_gpu = RAYs_O_gpu[..., None, :] + RAYs_D_gpu[..., None, :] * Z_VALs[..., :, None]  # (H, W, depth_size, 3)


DISTs_cat = torch.cat([DISTs_gpu, torch.tensor([1e10], device=device).expand(DISTs_gpu[..., :1].shape)], -1)  # (H, W, depth_size)
DISTs_final = DISTs_cat * torch.norm(RAYs_D_gpu[..., None, :], dim=-1)  # (960, 540, depth_size, 1)
DISTs_final = DISTs_final.flatten(0, 1)
RGB_TRAINED = torch.ones(3, device=device) * (0.6 + torch.tanh(MODEL_gpu.rgb) * 0.4)


N_timesteps = IMAGE_TEST_np.shape[0]
test_timesteps = torch.arange(N_timesteps, device=device, dtype=torch.float32) / (N_timesteps - 1)
with torch.no_grad():
    for i in tqdm.trange(0, test_timesteps.shape[0]):
        test_timesteps_expended = test_timesteps[i].expand(POINTs_gpu[..., :1].shape)
        POINTs_TIME_gpu = torch.cat([POINTs_gpu, test_timesteps_expended], dim=-1)
        POINTs_TIME_FLAT_gpu = POINTs_TIME_gpu.view(-1, POINTs_TIME_gpu.shape[-1])

        out_dim = 1
        RAW_FLAT = torch.zeros([POINTs_TIME_FLAT_gpu.shape[0], out_dim], device=device, dtype=torch.float32)
        bbox_mask = BBOX_MODEL_gpu.insideMask(POINTs_TIME_FLAT_gpu[..., :3], to_float=False)
        POINTs_TIME_FLAT_FINAL = POINTs_TIME_FLAT_gpu[bbox_mask]

        chunk = 512 * 64
        ret_list = []
        for _ in range(0, POINTs_TIME_FLAT_FINAL.shape[0], chunk):
            ret = MODEL_gpu(ENCODER_gpu(POINTs_TIME_FLAT_FINAL[_:min(_ + chunk, POINTs_TIME_FLAT_FINAL.shape[0])]))
            ret_list.append(ret)
        all_ret = torch.cat(ret_list, 0)
        RAW_FLAT[bbox_mask] = all_ret
        RAW = RAW_FLAT.reshape(-1, depth_size, out_dim)

        noise = 0.
        alpha = 1. - torch.exp(-torch.nn.functional.relu(RAW[..., -1] + noise) * DISTs_final)
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=device), 1. - alpha + 1e-10], -1), -1)[:, :-1]
        RGB_MAP_FLAT = torch.sum(weights[..., None] * RGB_TRAINED, -2)
        RGB_MAP = RGB_MAP_FLAT.view(POINTs_TIME_gpu.shape[0], POINTs_TIME_gpu.shape[1], RGB_MAP_FLAT.shape[-1])

        to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
        rgb8 = to8b(RGB_MAP.cpu().numpy())
        imageio.imsave(os.path.join(work_item.attrib("output")[0], "iter{:06d}".format(int(work_item.index)), 'rgb_{:03d}.png'.format(i)), rgb8)
