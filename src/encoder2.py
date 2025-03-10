import torch
import numpy as np


class HashEncoderNative(torch.nn.Module):
    def __init__(
            self,
            num_levels: int = 16,
            min_res: int = 16,
            max_res: int = 256,
            log2_hashmap_size: int = 19,
            features_per_level: int = 2,
            hash_init_scale: float = 0.001,
            device=torch.device("cuda"),
    ):
        super().__init__()
        self.device = device
        self.num_levels = num_levels
        self.min_res = min_res
        self.max_res = max_res
        self.features_per_level = features_per_level
        self.primes = torch.tensor([1, 2654435761, 805459861, 3674653429], device=device)
        self.hash_table_size = 2 ** log2_hashmap_size

        levels = torch.arange(self.num_levels)
        self.growth_factor = np.exp((np.log(self.max_res) - np.log(self.min_res)) / (self.num_levels - 1)) if self.num_levels > 1 else 1
        self.scalings = torch.floor(min_res * self.growth_factor ** levels)
        self.hash_offset = levels * self.hash_table_size

        self.hash_table = torch.rand(size=(self.hash_table_size * self.num_levels, self.features_per_level), device=device) * 2 - 1
        self.hash_table *= 0.001
        self.hash_table = torch.nn.Parameter(self.hash_table)

    def hash_fn(self, in_tensor):
        in_tensor = in_tensor * self.primes
        x = torch.bitwise_xor(in_tensor[..., 0], in_tensor[..., 1])
        x = torch.bitwise_xor(x, in_tensor[..., 2])
        x = torch.bitwise_xor(x, in_tensor[..., 3])
        x %= self.hash_table_size
        x += self.hash_offset.to(x.device)
        return x

    def forward(self, xyzt):
        xyzt = xyzt[..., None, :]
        scaled = xyzt * self.scalings.view(-1, 1).to(xyzt.device)
        scaled_c = torch.ceil(scaled).type(torch.int32)
        scaled_f = torch.floor(scaled).type(torch.int32)
        offset = scaled - scaled_f

        # Compute hashed indices for all 16 vertices in 4D space
        hashed = []
        for i in range(16):
            # Compute each vertex by selecting ceil or floor for each dimension
            mask = [(i >> d) & 1 for d in range(4)]  # Determine ceil (1) or floor (0) for each dimension
            vertex = torch.cat([scaled_c[..., d:d + 1] if mask[d] else scaled_f[..., d:d + 1] for d in range(4)], dim=-1)  # [..., L, 4]
            hashed.append(self.hash_fn(vertex))  # Compute hash index for this vertex

        # Fetch features for all 16 vertices
        features = [self.hash_table[h] for h in hashed]  # List of [..., num_levels, features_per_level]

        # Compute weights and perform 4D interpolation
        for d in range(4):
            next_features = []
            for j in range(0, len(features), 2):  # Process pairs of vertices
                f0, f1 = features[j], features[j + 1]
                weight = offset[..., d:d + 1]  # Weight along dimension d
                next_features.append(f0 * (1 - weight) + f1 * weight)
            features = next_features  # Update features for the next dimension

        # After 4 dimensions, we should have a single interpolated result
        encoded_value = features[0]  # [..., num_levels, features_per_level]

        return torch.flatten(encoded_value, start_dim=-2, end_dim=-1)  # [..., num_levels * features_per_level]


if __name__ == "__main__":
    encoder = HashEncoderNative()
    print(encoder)
