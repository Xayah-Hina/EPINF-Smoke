import numpy as np
import torch

import src.bbox as bbox
import src.encoder2 as encoder
import src.model as model
import src.radam as radam


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


def get_raw(POINTS_TIME: torch.Tensor, DISTs: torch.Tensor, RAYs_D_FLAT: torch.Tensor, BBOX_MODEL, MODEL, ENCODER):
    assert POINTS_TIME.dim() == 3 and POINTS_TIME.shape[-1] == 4
    assert POINTS_TIME.shape[0] == DISTs.shape[0] == RAYs_D_FLAT.shape[0]
    POINTS_TIME_FLAT = POINTS_TIME.view(-1, POINTS_TIME.shape[-1])  # [batch_size * N_depths, 4]
    out_dim = 1
    RAW_FLAT = torch.zeros([POINTS_TIME_FLAT.shape[0], out_dim], device=POINTS_TIME_FLAT.device, dtype=torch.float32)
    bbox_mask = BBOX_MODEL.insideMask(POINTS_TIME_FLAT[..., :3], to_float=False)
    if bbox_mask.sum() == 0:
        bbox_mask[0] = True
        assert False
    POINTS_TIME_FLAT_FINAL = POINTS_TIME_FLAT[bbox_mask]

    RAW_FLAT[bbox_mask] = MODEL(ENCODER(POINTS_TIME_FLAT_FINAL))

    # for _ in range(0, POINTS_TIME_FLAT_FINAL.shape[0], batch_size):
    #     RAW_FLAT[bbox_mask][_:_ + batch_size] = MODEL(ENCODER(POINTS_TIME_FLAT_FINAL[_:_ + batch_size]))
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


if __name__ == '__main__':
    device = torch.device("cuda")
    ############################## Datasets ##############################
    pinf_data = np.load("data/train_dataset.npz")
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
    ############################## Datasets ##############################

    # import cv2
    #
    # output_array = np.zeros((120, 4, H_int // 2, W_int // 2, 3), dtype=np.float32)
    # for i in range(IMAGE_TRAIN_np.shape[0]):
    #     for j in range(IMAGE_TRAIN_np.shape[1]):
    #         output_array[i, j] = cv2.resize(IMAGE_TRAIN_np[i, j], (W_int // 2, H_int // 2))
    # IMAGE_TRAIN_np = output_array
    # H_int = H_int // 2
    # W_int = W_int // 2

    ############################## Load Encoder ##############################
    ENCODER_gpu = encoder.HashEncoderNative(device=device).to(device)
    ############################## Load Encoder ##############################

    ############################## Load Model ##############################
    MODEL_gpu = model.NeRFSmall(num_layers=2,
                                hidden_dim=64,
                                geo_feat_dim=15,
                                num_layers_color=2,
                                hidden_dim_color=16,
                                input_ch=ENCODER_gpu.num_levels * 2).to(device)
    ############################## Load Model ##############################

    ############################## Load Optimizer ##############################
    lrate = 0.01
    lrate_decay = 10000
    OPTIMIZER = radam.RAdam([
        {'params': MODEL_gpu.parameters(), 'weight_decay': 1e-6},
        {'params': ENCODER_gpu.parameters(), 'eps': 1e-15}
    ], lr=lrate, betas=(0.9, 0.99))
    ############################## Load Optimizer ##############################

    ############################## Load BoundingBox ##############################
    VOXEL_TRAN_np = pinf_data['voxel_tran']
    VOXEL_SCALE_np = pinf_data['voxel_scale']
    voxel_tran_inv = np.linalg.inv(VOXEL_TRAN_np)
    BBOX_MODEL_gpu = bbox.BBox_Tool(voxel_tran_inv, VOXEL_SCALE_np)
    ############################## Load BoundingBox ##############################

    import tqdm
    import os

    batch_size = 256
    time_size = 1
    depth_size = 192
    global_step = 1
    loss_meter = AverageMeter()
    GRAD_vars = list(MODEL_gpu.parameters()) + list(ENCODER_gpu.parameters())
    for ITERATION in range(1, 2):
        IMAGE_TRAIN_gpu, RAYs_gpu, RAY_IDX_gpu = do_resample_rays(H_int, W_int)
        for i in tqdm.trange(0, RAY_IDX_gpu.shape[0], batch_size):
            BATCH_RAYs_O_gpu, BATCH_RAYs_D_gpu, BATCH_RAYs_IDX_gpu = get_ray_batch(RAYs_gpu, RAY_IDX_gpu, i, i + batch_size)  # [batch_size, 3], [batch_size, 3], [batch_size]
            FRAMES_INTERPOLATED_gpu, TIME_STEPs_gpu = get_frames_at_times(IMAGE_TRAIN_gpu, IMAGE_TRAIN_gpu.shape[0], time_size)  # [N_times, N x H x W, 3], [N_times]
            TARGET_S_gpu = FRAMES_INTERPOLATED_gpu[:, BATCH_RAYs_IDX_gpu].flatten(0, 1)  # [batch_size * N_times, 3]
            POINTS_gpu, DISTs_gpu = get_points(BATCH_RAYs_O_gpu, BATCH_RAYs_D_gpu, NEAR_float, FAR_float, depth_size, randomize=True)  # [batch_size, N_depths, 3]
            for TIME_STEP_gpu in TIME_STEPs_gpu:
                POINTS_TIME_gpu = torch.cat([POINTS_gpu, TIME_STEP_gpu.expand(POINTS_gpu[..., :1].shape)], dim=-1)  # [batch_size, N_depths, 4]
                RGB_MAP = get_raw(POINTS_TIME_gpu, DISTs_gpu, BATCH_RAYs_D_gpu, BBOX_MODEL_gpu, MODEL_gpu, ENCODER_gpu)
                img2mse = lambda x, y: torch.mean((x - y) ** 2)
                img_loss = img2mse(RGB_MAP, TARGET_S_gpu)
                loss = img_loss

                for param in GRAD_vars:  # slightly faster than optimizer.zero_grad()
                    param.grad = None
                loss.backward()
                OPTIMIZER.step()

                loss_meter.update(loss.item())
                if global_step % 100 == 0:
                    tqdm.tqdm.write(f"[TRAIN] Iter: {global_step} Loss: {loss_meter.avg:.2g}")
                    loss_meter.reset()

                decay_rate = 0.1
                decay_steps = lrate_decay
                new_lrate = lrate * (decay_rate ** (global_step / decay_steps))
                for param_group in OPTIMIZER.param_groups:
                    param_group['lr'] = new_lrate
                global_step += 1
        os.makedirs("checkpoint", exist_ok=True)
        path = os.path.join("checkpoint", '{:06d}.tar'.format(ITERATION))
        torch.save({
            'global_step': global_step,
            'network_fn_state_dict': MODEL_gpu.state_dict(),
            'embed_fn_state_dict': ENCODER_gpu.state_dict(),
            'optimizer_state_dict': OPTIMIZER.state_dict(),
        }, path)
