import torch
import taichi as ti
import numpy as np
from torch.cuda.amp import custom_bwd, custom_fwd


@ti.func
def linear_step(t):
    return t


@ti.func
def d_linear_step(t):
    return 1


@ti.kernel
def torch2ti(field: ti.template(), data: ti.types.ndarray()):
    for I in ti.grouped(data):
        field[I] = data[I]


@ti.kernel
def ti2torch(field: ti.template(), data: ti.types.ndarray()):
    for I in ti.grouped(data):
        data[I] = field[I]


@ti.kernel
def ti2torch_grad(field: ti.template(), grad: ti.types.ndarray()):
    for I in ti.grouped(grad):
        grad[I] = field.grad[I]


@ti.kernel
def torch2ti_grad(field: ti.template(), grad: ti.types.ndarray()):
    for I in ti.grouped(grad):
        field.grad[I] = grad[I]


@ti.func
def fast_hash(pos_grid_local):
    result = ti.uint32(0)
    primes = ti.math.uvec4(ti.uint32(1), ti.uint32(2654435761), ti.uint32(805459861), ti.uint32(3674653429))
    for i in ti.static(range(4)):
        result ^= ti.uint32(pos_grid_local[i]) * primes[i]
    return result


# ravel (i, j, k, t) to i + i_dim * j + (i_dim * j_dim) * k + (i_dim * j_dim * k_dim) * t
@ti.func
def under_hash(pos_grid_local, resolution):
    result = ti.uint32(0)
    stride = ti.uint32(1)
    for i in ti.static(range(4)):
        result += ti.uint32(pos_grid_local[i] * stride)
        stride *= resolution[i] + 1  # note the +1 here, because 256 x 256 grid actually has 257 x 257 entries
    return result


@ti.func
def grid_pos2hash_index(indicator, pos_grid_local, plane_res, map_size):
    hash_result = ti.uint32(0)
    if indicator == 1:
        hash_result = under_hash(pos_grid_local, plane_res)
    else:
        hash_result = fast_hash(pos_grid_local)

    return hash_result % map_size


@ti.kernel
def hash_encode_kernel(
        xyzts: ti.template(),
        table: ti.template(),
        xyzts_embedding: ti.template(),
        hash_map_indicator: ti.template(),
        hash_map_sizes_field: ti.template(),
        hash_map_shapes_field: ti.template(),
        offsets: ti.template(),
        B: ti.i32,
        num_scales: ti.i32):
    ti.loop_config(block_dim=16)
    for i, level in ti.ndrange(B, num_scales):
        res_x = hash_map_shapes_field[level, 0]
        res_y = hash_map_shapes_field[level, 1]
        res_z = hash_map_shapes_field[level, 2]
        res_t = hash_map_shapes_field[level, 3]
        plane_res = ti.Vector([res_x, res_y, res_z, res_t])
        pos = ti.Vector([xyzts[i, 0], xyzts[i, 1], xyzts[i, 2], xyzts[i, 3]]) * plane_res

        pos_grid_uint = ti.cast(ti.floor(pos), ti.uint32)  # floor
        pos_grid_uint = ti.math.clamp(pos_grid_uint, 0, plane_res - 1)
        pos -= pos_grid_uint  # pos now represents frac
        pos = ti.math.clamp(pos, 0.0, 1.0)

        offset = offsets[level]

        indicator = hash_map_indicator[level]
        map_size = hash_map_sizes_field[level]

        local_feature_0 = 0.0
        local_feature_1 = 0.0

        for idx in ti.static(range(16)):
            w = 1.
            pos_grid_local = ti.math.uvec4(0)

            for d in ti.static(range(4)):
                t = linear_step(pos[d])
                if (idx & (1 << d)) == 0:
                    pos_grid_local[d] = pos_grid_uint[d]
                    w *= 1 - t
                else:
                    pos_grid_local[d] = pos_grid_uint[d] + 1
                    w *= t

            index = grid_pos2hash_index(indicator, pos_grid_local, plane_res, map_size)
            index_table = offset + index * 2  # the flat index for the 1st entry
            index_table_int = ti.cast(index_table, ti.int32)
            local_feature_0 += w * table[index_table_int]
            local_feature_1 += w * table[index_table_int + 1]

        xyzts_embedding[i, level * 2] = local_feature_0
        xyzts_embedding[i, level * 2 + 1] = local_feature_1


@ti.kernel
def hash_encode_kernel_grad(
        xyzts: ti.template(),
        table: ti.template(),
        xyzts_embedding: ti.template(),
        hash_map_indicator: ti.template(),
        hash_map_sizes_field: ti.template(),
        hash_map_shapes_field: ti.template(),
        offsets: ti.template(),
        B: ti.i32,
        num_scales: ti.i32,
        xyzts_grad: ti.template(),
        table_grad: ti.template(),
        output_grad: ti.template()):
    # # # get hash table embedding

    ti.loop_config(block_dim=16)
    for i, level in ti.ndrange(B, num_scales):
        res_x = hash_map_shapes_field[level, 0]
        res_y = hash_map_shapes_field[level, 1]
        res_z = hash_map_shapes_field[level, 2]
        res_t = hash_map_shapes_field[level, 3]
        plane_res = ti.Vector([res_x, res_y, res_z, res_t])
        pos = ti.Vector([xyzts[i, 0], xyzts[i, 1], xyzts[i, 2], xyzts[i, 3]]) * plane_res

        pos_grid_uint = ti.cast(ti.floor(pos), ti.uint32)  # floor
        pos_grid_uint = ti.math.clamp(pos_grid_uint, 0, plane_res - 1)
        pos -= pos_grid_uint  # pos now represents frac
        pos = ti.math.clamp(pos, 0.0, 1.0)

        offset = offsets[level]

        indicator = hash_map_indicator[level]
        map_size = hash_map_sizes_field[level]

        for idx in ti.static(range(16)):
            w = 1.
            pos_grid_local = ti.math.uvec4(0)
            dw = ti.Vector([0., 0., 0., 0.])
            # prods = ti.Vector([0., 0., 0.,0.])
            for d in ti.static(range(4)):
                t = linear_step(pos[d])
                dt = d_linear_step(pos[d])
                if (idx & (1 << d)) == 0:
                    pos_grid_local[d] = pos_grid_uint[d]
                    w *= 1 - t
                    dw[d] = -dt

                else:
                    pos_grid_local[d] = pos_grid_uint[d] + 1
                    w *= t
                    dw[d] = dt

            index = grid_pos2hash_index(indicator, pos_grid_local, plane_res, map_size)
            index_table = offset + index * 2  # the flat index for the 1st entry
            index_table_int = ti.cast(index_table, ti.int32)
            table_grad[index_table_int] += w * output_grad[i, 2 * level]
            table_grad[index_table_int + 1] += w * output_grad[i, 2 * level + 1]
            for d in ti.static(range(4)):
                # eps = 1e-15
                # prod = w / ((linear_step(pos[d]) if idx & (1 << d) > 0 else 1 - linear_step(pos[d])) + eps)
                # prod=1.0
                # for k in range(4):
                #     if k == d:
                #         prod *= dw[k]
                #     else:
                #         prod *= 1- linear_step(pos[k]) if (idx & (1 << k) == 0) else linear_step(pos[k])
                prod = dw[d] * (
                    linear_step(pos[(d + 1) % 4]) if (idx & (1 << ((d + 1) % 4)) > 0) else 1 - linear_step(
                        pos[(d + 1) % 4])
                ) * (
                           linear_step(pos[(d + 2) % 4]) if (idx & (1 << ((d + 2) % 4)) > 0) else 1 - linear_step(
                               pos[(d + 2) % 4])
                       ) * (
                           linear_step(pos[(d + 3) % 4]) if (idx & (1 << ((d + 3) % 4)) > 0) else 1 - linear_step(
                               pos[(d + 3) % 4])
                       )
                xyzts_grad[i, d] += table[index_table_int] * prod * plane_res[d] * output_grad[i, 2 * level]
                xyzts_grad[i, d] += table[index_table_int + 1] * prod * plane_res[d] * output_grad[i, 2 * level + 1]


class HashEncoderHyFluid(torch.nn.Module):
    def __init__(
            self,
            min_res: np.array,
            max_res: np.array,
            num_scales: int,
            max_params=2 ** 19,
            features_per_level: int = 2,
            max_num_queries=10000000,
    ):
        super().__init__()
        b = np.exp((np.log(max_res) - np.log(min_res)) / (num_scales - 1))

        hash_map_shapes = []
        hash_map_sizes = []
        hash_map_indicator = []
        offsets = []
        total_hash_size = 0
        for scale_i in range(num_scales):
            res = np.ceil(min_res * np.power(b, scale_i)).astype(int)
            params_in_level_raw = np.int64(res[0] + 1) * np.int64(res[1] + 1) * np.int64(res[2] + 1) * np.int64(res[3] + 1)
            params_in_level = int(params_in_level_raw) if params_in_level_raw % 8 == 0 else int((params_in_level_raw + 8 - 1) / 8) * 8
            params_in_level = min(max_params, params_in_level)
            hash_map_shapes.append(res)
            hash_map_sizes.append(params_in_level)
            hash_map_indicator.append(1 if params_in_level_raw <= params_in_level else 0)
            offsets.append(total_hash_size)
            total_hash_size += params_in_level * features_per_level

        ####################################################################################################
        self.hash_map_shapes_field = ti.field(dtype=ti.i32, shape=(num_scales, 4))
        self.hash_map_shapes_field.from_numpy(np.array(hash_map_shapes))

        self.hash_map_sizes_field = ti.field(dtype=ti.i32, shape=(num_scales,))
        self.hash_map_sizes_field.from_numpy(np.array(hash_map_sizes))

        self.hash_map_indicator_field = ti.field(dtype=ti.i32, shape=(num_scales,))
        self.hash_map_indicator_field.from_numpy(np.array(hash_map_indicator))

        self.offsets_fields = ti.field(ti.i32, shape=(num_scales,))
        self.offsets_fields.from_numpy(np.array(offsets))

        self.hash_table = torch.nn.Parameter((torch.rand(size=(total_hash_size,), dtype=torch.float32) * 2.0 - 1.0) * 1e-4, requires_grad=True)

        self.parameter_fields = ti.field(dtype=ti.f32, shape=(total_hash_size,), needs_grad=True)
        self.parameter_fields_grad = ti.field(dtype=ti.f32, shape=(total_hash_size,), needs_grad=True)

        self.output_fields = ti.field(dtype=ti.f32, shape=(max_num_queries, num_scales * features_per_level), needs_grad=True)
        self.output_grad = ti.field(dtype=ti.f32, shape=(max_num_queries, num_scales * features_per_level), needs_grad=True)

        self.input_fields = ti.field(dtype=ti.f32, shape=(max_num_queries, 4), needs_grad=True)
        self.input_fields_grad = ti.field(dtype=ti.f32, shape=(max_num_queries, 4), needs_grad=True)

        self.num_scales = num_scales
        self.features_per_level = features_per_level
        ####################################################################################################

        self.register_buffer('hash_grad', torch.zeros(total_hash_size, dtype=torch.float32), persistent=False)
        self.register_buffer('hash_grad2', torch.zeros(total_hash_size, dtype=torch.float32), persistent=False)
        self.register_buffer('input_grad', torch.zeros(max_num_queries, 4, dtype=torch.float32), persistent=False)
        self.register_buffer('input_grad2', torch.zeros(max_num_queries, 4, dtype=torch.float32), persistent=False)
        self.register_buffer('output_embedding', torch.zeros(max_num_queries, num_scales * 2, dtype=torch.float32), persistent=False)

        ####################################################################################################
        class ModuleFunction(torch.autograd.Function):
            @staticmethod
            @custom_fwd(cast_inputs=torch.float32)
            def forward(ctx, input_pos, params):
                output_embedding = self.output_embedding[:input_pos.shape[0]].contiguous()
                torch2ti(self.input_fields, input_pos.contiguous())
                torch2ti(self.parameter_fields, params.contiguous())

                hash_encode_kernel(
                    self.input_fields,
                    self.parameter_fields,
                    self.output_fields,
                    self.hash_map_indicator_field,
                    self.hash_map_sizes_field,
                    self.hash_map_shapes_field,
                    self.offsets_fields,
                    input_pos.shape[0],
                    self.num_scales,
                )

                ti2torch(self.output_fields, output_embedding)
                ctx.save_for_backward(input_pos, params)
                return output_embedding

            @staticmethod
            @custom_bwd
            def backward(ctx, doutput):
                self.input_fields.grad.fill(0.)
                self.input_fields_grad.fill(0.)
                self.parameter_fields.grad.fill(0.)
                self.parameter_fields_grad.fill(0.)

                input_pos, params = ctx.saved_tensors
                return self.module_function_grad.apply(input_pos, params, doutput)

        class ModuleFunctionGrad(torch.autograd.Function):
            @staticmethod
            @custom_fwd(cast_inputs=torch.float32)
            def forward(ctx, input_pos, params, doutput):
                torch2ti(self.input_fields, input_pos.contiguous())
                torch2ti(self.parameter_fields, params.contiguous())
                torch2ti(self.output_grad, doutput.contiguous())

                hash_encode_kernel_grad(
                    self.input_fields,
                    self.parameter_fields,
                    self.output_fields,
                    self.hash_map_indicator_field,
                    self.hash_map_sizes_field,
                    self.hash_map_shapes_field,
                    self.offsets_fields,
                    doutput.shape[0],
                    self.num_scales,
                    self.input_fields_grad,
                    self.parameter_fields_grad,
                    self.output_grad
                )

                ti2torch(self.input_fields_grad, self.input_grad.contiguous())
                ti2torch(self.parameter_fields_grad, self.hash_grad.contiguous())
                return self.input_grad[:doutput.shape[0]], self.hash_grad

            @staticmethod
            @custom_bwd
            def backward(ctx, d_input_grad, d_hash_grad):
                self.parameter_fields.grad.fill(0.)
                self.input_fields.grad.fill(0.)
                torch2ti_grad(self.input_fields_grad, d_input_grad.contiguous())
                torch2ti_grad(self.parameter_fields_grad, d_hash_grad.contiguous())

                hash_encode_kernel_grad.grad(
                    self.input_fields,
                    self.parameter_fields,
                    self.output_fields,
                    self.hash_map_indicator_field,
                    self.hash_map_sizes_field,
                    self.hash_map_shapes_field,
                    self.offsets_fields,
                    d_input_grad.shape[0],
                    self.num_scales,
                    self.input_fields_grad,
                    self.parameter_fields_grad,
                    self.output_grad
                )

                ti2torch_grad(self.input_fields, self.input_grad2.contiguous()[:d_input_grad.shape[0]])
                ti2torch_grad(self.parameter_fields, self.hash_grad2.contiguous())
                # set_trace(term_size=(120,30))
                return self.input_grad2[:d_input_grad.shape[0]], self.hash_grad2, None

        self.module_function = ModuleFunction
        self.module_function_grad = ModuleFunctionGrad
        ####################################################################################################

    def forward(self, positions):
        # positions: (N, 4), normalized to [-1, 1]
        positions = positions * 0.5 + 0.5
        return self.module_function.apply(positions, self.hash_table)
