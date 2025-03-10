import numpy as np
import cv2
import os

pinf_data = np.load("data/train_dataset.npz")
IMAGE_TRAIN_np = pinf_data['images_train']
POSES_TRAIN_np = pinf_data['poses_train']
HWF_np = pinf_data['hwf']
H_int = int(HWF_np[0])
W_int = int(HWF_np[1])
FOCAL_float = float(HWF_np[2])
NEAR_float = pinf_data['near'].item()
FAR_float = pinf_data['far'].item()

os.makedirs(work_item.attrib("output")[0], exist_ok=True)

def resample_dataset(images, poses, W, H, FOCAL):
    """
    Sample image with bilinear interpolation
    :param images: (T, V, H, W, 3)
    :param poses: (V, 2, H, W)
    :param W: int
    :param H: int
    :param FOCAL: float
    :return: images_resampled, rays_np
    """
    rays_o_list = []
    rays_d_list = []
    ij_list = []
    for c2w in poses[:, :3, :4]:
        _i, _j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
        random_offset = np.random.uniform(0, 1, size=(H, W, 2))
        i = _i + random_offset[..., 0]
        j = _j + random_offset[..., 1]
        dirs = np.stack([
            (i - 0.5 * W) / FOCAL,
            -(j - 0.5 * H) / FOCAL,
            -np.ones_like(i)
        ], axis=-1)
        r_d = dirs @ c2w[:3, :3].T
        r_o = c2w[:3, -1]
        rays_d_list.append(r_d)
        rays_o_list.append(r_o)
        ij_list.append([i, j])
    rays_d_np = np.stack(rays_d_list, 0).astype(np.float32)
    rays_o_np = np.stack(rays_o_list, 0).astype(np.float32)
    ij_np = np.stack(ij_list, 0).astype(np.float32)

    images_resampled = np.zeros_like(images)
    T, V, H, W, C = images.shape
    for t in range(T):
        for v_idx in range(V):
            images_resampled[t, v_idx] = cv2.remap(
                src=images[t, v_idx],
                map1=ij_np[v_idx, 0],
                map2=ij_np[v_idx, 1],
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
    return images_resampled, rays_d_np, rays_o_np













IMAGES_TRAIN_np, RAYS_D_TRAIN_np, RAYS_O_TRAIN_np = resample_dataset(IMAGE_TRAIN_np, POSES_TRAIN_np, W_int, H_int, FOCAL_float)
index = work_item.index
filename = f'output/train_cache/train_data_{index:03d}.npz'
np.savez_compressed(filename, images=IMAGES_TRAIN_np, rays_d=RAYS_D_TRAIN_np, rays_o=RAYS_O_TRAIN_np)
del IMAGES_TRAIN_np
del RAYS_D_TRAIN_np
del RAYS_O_TRAIN_np



















import torch
import numpy as np
import sys
sys.path.append(work_item.attrib("cwd")[0])

device = torch.device("cuda")

pinf_data = np.load("data/train_dataset.npz")
HWF_np = pinf_data['hwf']
H_int = int(HWF_np[0])
W_int = int(HWF_np[1])
FOCAL_float = float(HWF_np[2])
NEAR_float = pinf_data['near'].item()
FAR_float = pinf_data['far'].item()
VOXEL_TRAN_np = pinf_data['voxel_tran']
VOXEL_SCALE_np = pinf_data['voxel_scale']

############################## Load BoundingBox ##############################
import src.bbox as bbox
voxel_tran_inv = np.linalg.inv(VOXEL_TRAN_np)
BBOX_MODEL_gpu = bbox.BBox_Tool(voxel_tran_inv, VOXEL_SCALE_np)
############################## Load BoundingBox ##############################

############################## Load Encoder ##############################
import src.encoder2 as encoder
ENCODER_gpu = encoder.HashEncoderNative(device=device).to(device)
############################## Load Encoder ##############################

############################## Load Model ##############################
import src.model as model
MODEL_gpu = model.NeRFSmall(num_layers=2,
                            hidden_dim=64,
                            geo_feat_dim=15,
                            num_layers_color=2,
                            hidden_dim_color=16,
                            input_ch=ENCODER_gpu.num_levels * 2).to(device)
############################## Load Model ##############################

############################## Load Optimizer ##############################
import src.radam as radam
lrate = 0.01
lrate_decay = 10000
OPTIMIZER = radam.RAdam([
    {'params': MODEL_gpu.parameters(), 'weight_decay': 1e-6},
    {'params': ENCODER_gpu.parameters(), 'eps': 1e-15}
], lr=lrate, betas=(0.9, 0.99))
############################## Load Optimizer ##############################
GRAD_vars = list(MODEL_gpu.parameters()) + list(ENCODER_gpu.parameters())

import os
import tqdm
os.makedirs(work_item.attrib("checkpoint")[0], exist_ok=True)

index = work_item.index
file_path = os.path.join(work_item.attrib("output")[0], f'train_data_{index:03d}.npz')
loaded = np.load(file_path)
images_np = loaded["images"] # (120, 4, 960, 540, 3)
rays_d_np = loaded["rays_d"] # (4, 960, 540, 3)
rays_o_np = loaded["rays_o"] # (4, 3)

IMAGEs_gpu = torch.tensor(images_np, device=device, dtype=torch.float32)
RAYs_D_gpu = torch.tensor(rays_d_np, device=device, dtype=torch.float32).reshape(-1, 3)
RAYs_O_gpu = torch.tensor(rays_o_np, device=device, dtype=torch.float32).unsqueeze(1).unsqueeze(2).expand(rays_d_np.shape).reshape(-1, 3)
RAYs_IDX_gpu = torch.randperm(RAYs_D_gpu.shape[0], device=device, dtype=torch.int32)

# batch_size = int(work_item.attrib("batch_size")[0])
batch_size = int(work_item.attrib("batch_size")[0])
n_frames = IMAGEs_gpu.size(0)
depth_size = 192
randomize = True
global_step = 1


for start in tqdm.trange(0, RAYs_IDX_gpu.shape[0], batch_size):
    BATCH_RAYs_IDX = RAYs_IDX_gpu[start:start+batch_size]
    BATCH_RAYs_D = RAYs_D_gpu[BATCH_RAYs_IDX]
    BATCH_RAYs_O = RAYs_O_gpu[BATCH_RAYs_IDX]

    float_frame_index = torch.rand(1).item() * (n_frames - 1)
    lower_frame_index = int(float_frame_index)
    upper_frame_index = min(lower_frame_index + 1, n_frames - 1)
    alpha = float_frame_index - lower_frame_index
    TIME_STEP_gpu = torch.tensor(float_frame_index / (n_frames - 1), device=device, dtype=torch.float32)
    SUB_IMAGEs_gpu = IMAGEs_gpu.reshape(IMAGEs_gpu.shape[0], -1, IMAGEs_gpu.shape[-1])[:, BATCH_RAYs_IDX, :]

    lower_frame = SUB_IMAGEs_gpu[lower_frame_index]
    upper_frame = SUB_IMAGEs_gpu[upper_frame_index]
    TARGET_S_gpu = (1 - alpha) * lower_frame + alpha * upper_frame

    T_VALs = torch.linspace(0., 1., steps=depth_size, device=device, dtype=torch.float32) # (depth_size)
    Z_VALs = NEAR_float * torch.ones_like(BATCH_RAYs_D[..., :1]) * (1. - T_VALs) + FAR_float * torch.ones_like(BATCH_RAYs_D[..., :1]) * T_VALs # (batch_size, depth_size)
    if randomize:
        MID_VALs = .5 * (Z_VALs[..., 1:] + Z_VALs[..., :-1])  # [batch_size, depth_size-1]
        UPPER_VALs = torch.cat([MID_VALs, Z_VALs[..., -1:]], -1)  # [batch_size, depth_size]
        LOWER_VALs = torch.cat([Z_VALs[..., :1], MID_VALs], -1)  # [batch_size, depth_size]
        T_RAND = torch.rand(Z_VALs.shape, device=device, dtype=torch.float32)  # [batch_size, depth_size]
        Z_VALs = LOWER_VALs + (UPPER_VALs - LOWER_VALs) * T_RAND  # [batch_size, depth_size]
    DISTs_gpu = Z_VALs[..., 1:] - Z_VALs[..., :-1]  # [batch_size, depth_size-1]
    POINTs_gpu = BATCH_RAYs_O[..., None, :] + BATCH_RAYs_D[..., None, :] * Z_VALs[..., :, None]  # [batch_size, depth_size, 3]

    POINTs_TIME_gpu = torch.cat([POINTs_gpu, TIME_STEP_gpu.expand(POINTs_gpu[..., :1].shape)], dim=-1)  # [batch_size, depth_size, 4]


    POINTs_TIME_FLAT_gpu = POINTs_TIME_gpu.view(-1, POINTs_TIME_gpu.shape[-1])
    out_dim = 1
    RAW_FLAT = torch.zeros([POINTs_TIME_FLAT_gpu.shape[0], out_dim], device=device, dtype=torch.float32)
    bbox_mask = BBOX_MODEL_gpu.insideMask(POINTs_TIME_FLAT_gpu[..., :3], to_float=False)
    POINTs_TIME_FLAT_FINAL = POINTs_TIME_FLAT_gpu[bbox_mask]
    RAW_FLAT[bbox_mask] = MODEL_gpu(ENCODER_gpu(POINTs_TIME_FLAT_FINAL))
    RAW = RAW_FLAT.reshape(*POINTs_TIME_gpu.shape[:-1], out_dim)
    DISTs_cat = torch.cat([DISTs_gpu, torch.tensor([1e10], device=device).expand(DISTs_gpu[..., :1].shape)], -1)  # [batch_size, depth_size]
    DISTS_final = DISTs_cat * torch.norm(BATCH_RAYs_D[..., None, :], dim=-1)  # [batch_size, depth_size]
    RGB_TRAINED = torch.ones(3, device=device) * (0.6 + torch.tanh(MODEL_gpu.rgb) * 0.4)

    noise = 0.
    alpha = 1. - torch.exp(-torch.nn.functional.relu(RAW[..., -1] + noise) * DISTS_final)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=device), 1. - alpha + 1e-10], -1), -1)[:, :-1]
    RGB_MAP = torch.sum(weights[..., None] * RGB_TRAINED, -2)

    img_loss = torch.nn.functional.mse_loss(RGB_MAP, TARGET_S_gpu)
    loss = img_loss

    for param in GRAD_vars:  # slightly faster than optimizer.zero_grad()
        param.grad = None
    loss.backward()
    OPTIMIZER.step()

    decay_rate = 0.1
    decay_steps = lrate_decay
    new_lrate = lrate * (decay_rate ** (global_step / decay_steps))
    for param_group in OPTIMIZER.param_groups:
        param_group['lr'] = new_lrate
    global_step += 1

path = os.path.join(work_item.attrib("checkpoint")[0], '{:06d}.tar'.format(int(work_item.index)))
torch.save({
    'global_step': global_step,
    'network_fn_state_dict': MODEL_gpu.state_dict(),
    'embed_fn_state_dict': ENCODER_gpu.state_dict(),
    'optimizer_state_dict': OPTIMIZER.state_dict(),
}, path)