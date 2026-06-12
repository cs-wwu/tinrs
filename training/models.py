"""
INR architectures for terrain representation.

All models follow the interface:
    model(coords: Tensor[N, 2]) -> Tensor[N, 1]
    model.param_count() -> int
    model.trainable_param_count() -> int
"""

import math

import torch
import torch.nn as nn

# -- SIREN --------------------------------------------------------------------


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, omega=30.0, is_first=False):
        super().__init__()
        self.omega = omega
        self.linear = nn.Linear(in_features, out_features)
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1.0 / in_features, 1.0 / in_features)
            else:
                self.linear.weight.uniform_(
                    -math.sqrt(6.0 / in_features) / omega,
                    math.sqrt(6.0 / in_features) / omega,
                )

    def forward(self, x):
        return torch.sin(self.omega * self.linear(x))


class SIREN(nn.Module):
    """f(lat, lon) -> elevation using sinusoidal activations."""

    def __init__(self, hidden_dim=256, num_layers=3, omega=30.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.omega = omega

        layers = [SineLayer(2, hidden_dim, omega=omega, is_first=True)]
        for _ in range(num_layers - 1):
            layers.append(SineLayer(hidden_dim, hidden_dim, omega=omega))
        self.net = nn.Sequential(*layers)

        self.final = nn.Linear(hidden_dim, 1)
        with torch.no_grad():
            self.final.weight.uniform_(
                -math.sqrt(6.0 / hidden_dim) / omega,
                math.sqrt(6.0 / hidden_dim) / omega,
            )

    def forward(self, coords):
        return self.final(self.net(coords))

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def trainable_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# -- BACON ---------------------------------------------------------------------


class FourierFilter(nn.Module):
    """Frozen-weight Fourier filter with trainable biases."""

    def __init__(
        self, coord_dim, hidden_dim, weight_scale, quantization_interval=2 * math.pi
    ):
        super().__init__()
        self.linear = nn.Linear(coord_dim, hidden_dim)
        with torch.no_grad():
            for i in range(coord_dim):
                n_intervals = int(2 * weight_scale / quantization_interval) + 1
                init = torch.randint(
                    0, max(n_intervals, 1), (hidden_dim,), dtype=torch.float32
                )
                init = init * quantization_interval - weight_scale
                self.linear.weight.data[:, i] = init
            self.linear.bias.uniform_(-math.pi, math.pi)
        self.linear.weight.requires_grad = False

    def forward(self, coords):
        return torch.sin(self.linear(coords))


class BACON(nn.Module):
    """
    Band-limited Coordinate Network (Lindell et al., CVPR 2022).

    Each layer has a bounded frequency, enabling multi-scale LOD outputs.
    Fourier filter weights are frozen; only biases + interstitial linears train.
    """

    def __init__(self, hidden_dim=256, num_layers=5, max_freq=64.0, coord_dim=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_freq = max_freq
        self.coord_dim = coord_dim
        self.freq_per_layer = max_freq / num_layers

        qi = math.pi
        ws = math.pi * self.freq_per_layer

        self.filters = nn.ModuleList(
            [FourierFilter(coord_dim, hidden_dim, ws, qi) for _ in range(num_layers)]
        )

        self.linears = nn.ModuleList()
        for _ in range(num_layers - 1):
            lin = nn.Linear(hidden_dim, hidden_dim)
            with torch.no_grad():
                limit = math.sqrt(6.0 / hidden_dim)
                lin.weight.uniform_(-limit, limit)
            self.linears.append(lin)

        self.output_heads = nn.ModuleList()
        for _ in range(num_layers):
            head = nn.Linear(hidden_dim, 1)
            with torch.no_grad():
                limit = math.sqrt(6.0 / hidden_dim)
                head.weight.uniform_(-limit, limit)
            self.output_heads.append(head)

    def forward(self, coords, return_all_layers=False):
        outputs = []
        h = self.filters[0](coords)
        if return_all_layers:
            outputs.append(self.output_heads[0](h))
        for i in range(1, self.num_layers):
            h = self.filters[i](coords) * self.linears[i - 1](h)
            if return_all_layers:
                outputs.append(self.output_heads[i](h))
        return outputs if return_all_layers else self.output_heads[-1](h)

    def forward_to_layer(self, coords, layer_idx):
        h = self.filters[0](coords)
        if layer_idx == 0:
            return self.output_heads[0](h)
        for i in range(1, layer_idx + 1):
            h = self.filters[i](coords) * self.linears[i - 1](h)
        return self.output_heads[layer_idx](h)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def trainable_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_freq_info(self):
        return [
            (self.freq_per_layer, self.freq_per_layer * (i + 1))
            for i in range(self.num_layers)
        ]


# -- SHACIRA compression components (Girish et al., ICCV 2023) ----------------


def sga_quantize(latents, temperature, training):
    """Stochastic Gumbel Annealing quantization.

    Training + temp > 0: soft quantization via Gumbel-softmax gate between
    floor and ceil. Eval or temp <= 0: hard round with straight-through
    estimator.
    """
    if not training or temperature <= 0:
        return latents + (torch.round(latents) - latents).detach()

    floor_val = torch.floor(latents)
    frac = latents - floor_val

    eps = 1e-8
    noise_f = -torch.log(-torch.log(torch.rand_like(frac).clamp(eps, 1.0 - eps)))
    noise_c = -torch.log(-torch.log(torch.rand_like(frac).clamp(eps, 1.0 - eps)))

    logit_f = torch.log((1.0 - frac).clamp(min=eps)) + noise_f
    logit_c = torch.log(frac.clamp(min=eps)) + noise_c

    gate = torch.sigmoid((logit_f - logit_c) / temperature)
    return gate * floor_val + (1.0 - gate) * (floor_val + 1.0)


class LatentDecoder(nn.Module):
    """Shared linear decoder: latent_dim -> features_per_level."""

    def __init__(self, latent_dim, features_per_level):
        super().__init__()
        self.linear = nn.Linear(latent_dim, features_per_level)

    def forward(self, x):
        return self.linear(x)


class Bitparm(nn.Module):
    """One layer of the BitEstimator CDF model."""

    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(1))
        self.b = nn.Parameter(torch.zeros(1))
        self.c = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return (
            x * torch.nn.functional.softplus(self.a)
            + self.b
            + torch.tanh(x) * torch.tanh(self.c)
        )


class BitEstimator(nn.Module):
    """Learned CDF model for estimating bit cost of quantized latents."""

    def __init__(self, num_layers=1):
        super().__init__()
        self.layers = nn.ModuleList([Bitparm() for _ in range(num_layers)])
        self.final_a = nn.Parameter(torch.zeros(1))
        self.final_b = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return torch.sigmoid(
            x * torch.nn.functional.softplus(self.final_a) + self.final_b
        )

    def entropy_loss(self, latents):
        """Average bit cost per latent value. Latents should have U(-0.5, 0.5) noise."""
        prob = (self.forward(latents + 0.5) - self.forward(latents - 0.5)).clamp(
            min=1e-9
        )
        return (-torch.log2(prob)).mean()


# -- Hash-Encoded MLP (Instant-NGP style) -------------------------------------

# Spatial hash primes (from Mueller et al.)
_HASH_PRIME_1 = 1
_HASH_PRIME_2 = 2654435761


class HashEncoding(nn.Module):
    """
    Multi-resolution hash encoding (Müller et al., SIGGRAPH 2022).

    For each of num_levels resolution levels, hash 2D grid cell corners into
    a learned feature table and bilinearly interpolate.
    """

    def __init__(
        self,
        num_levels=16,
        features_per_level=2,
        log2_hashmap_size=14,
        base_resolution=16,
        finest_resolution=3601,
        latent_dim=None,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.features_per_level = features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.hashmap_size = 2**log2_hashmap_size
        self.output_dim = num_levels * features_per_level
        self.latent_dim = latent_dim

        if num_levels > 1:
            self.growth_factor = math.exp(
                math.log(finest_resolution / base_resolution) / (num_levels - 1)
            )
        else:
            self.growth_factor = 1.0

        self.resolutions = [
            int(base_resolution * (self.growth_factor**level))
            for level in range(num_levels)
        ]

        # SHACIRA: store latents + decoder, otherwise raw features
        entry_dim = latent_dim if latent_dim is not None else features_per_level
        self.hash_tables = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(self.hashmap_size, entry_dim).uniform_(-1e-4, 1e-4)
                )
                for _ in range(num_levels)
            ]
        )

        if latent_dim is not None:
            self.decoder = LatentDecoder(latent_dim, features_per_level)
            self.bit_estimator = BitEstimator(num_layers=1)
            self._sga_temperature = 1.0
        else:
            self.decoder = None
            self.bit_estimator = None

    def _decode(self, raw):
        """Quantize latents and decode to features, or pass through directly."""
        if self.latent_dim is not None:
            q = sga_quantize(raw, self._sga_temperature, self.training)
            return self.decoder(q)
        return raw

    def forward(self, coords):
        """coords: (N, 2) in [-1, 1]. Returns: (N, num_levels * features_per_level)."""
        coords_01 = (coords + 1.0) * 0.5  # map to [0, 1]
        all_features = []

        for level in range(self.num_levels):
            res = self.resolutions[level]
            scaled = coords_01 * res

            base = torch.floor(scaled).long()
            frac = scaled - base.float()
            fx = frac[:, 0:1]
            fy = frac[:, 1:2]

            # Hash the 4 corners of each 2D cell
            bx, by = base[:, 0], base[:, 1]
            idx00 = ((bx) * _HASH_PRIME_1 ^ (by) * _HASH_PRIME_2) % self.hashmap_size
            idx10 = (
                (bx + 1) * _HASH_PRIME_1 ^ (by) * _HASH_PRIME_2
            ) % self.hashmap_size
            idx01 = (
                (bx) * _HASH_PRIME_1 ^ (by + 1) * _HASH_PRIME_2
            ) % self.hashmap_size
            idx11 = (
                (bx + 1) * _HASH_PRIME_1 ^ (by + 1) * _HASH_PRIME_2
            ) % self.hashmap_size

            table = self.hash_tables[level]
            # SHACIRA: quantize + decode entire table once per level so
            # the same entry gets consistent features across corners
            if self.latent_dim is not None:
                table = self._decode(table)
            f00 = table[idx00]
            f10 = table[idx10]
            f01 = table[idx01]
            f11 = table[idx11]

            # Bilinear interpolation
            interpolated = (
                f00 * (1 - fx) * (1 - fy)
                + f10 * fx * (1 - fy)
                + f01 * (1 - fx) * fy
                + f11 * fx * fy
            )
            all_features.append(interpolated)

        return torch.cat(all_features, dim=-1)

    def compute_entropy_loss(self):
        """Average entropy (bits) across all latent hash tables."""
        if self.bit_estimator is None:
            return torch.tensor(0.0)
        total_bits = 0.0
        total_entries = 0
        for table in self.hash_tables:
            noisy = table + torch.empty_like(table).uniform_(-0.5, 0.5)
            total_bits += self.bit_estimator.entropy_loss(noisy) * table.numel()
            total_entries += table.numel()
        return total_bits / total_entries

    def compressed_size_bytes(self):
        """Estimated compressed size using learned entropy model."""
        if self.bit_estimator is None:
            return sum(t.numel() for t in self.hash_tables)
        total_bits = 0.0
        with torch.no_grad():
            for table in self.hash_tables:
                rounded = torch.round(table)
                prob = (
                    self.bit_estimator(rounded + 0.5)
                    - self.bit_estimator(rounded - 0.5)
                ).clamp(min=1e-9)
                total_bits += (-torch.log2(prob)).sum().item()
        # Decoder + entropy model overhead (INT8, same as MLP accounting)
        overhead = sum(p.numel() for p in self.decoder.parameters())
        overhead += sum(p.numel() for p in self.bit_estimator.parameters())
        return total_bits / 8.0 + overhead


class HashMLP(nn.Module):
    """
    Hash-encoded MLP: multiresolution hash encoding + tiny ReLU MLP.

    The hash table provides spatial features; the MLP resolves collisions.
    Most parameters are in the hash tables, not the MLP.
    """

    def __init__(
        self,
        num_levels=16,
        features_per_level=2,
        log2_hashmap_size=14,
        base_resolution=16,
        finest_resolution=3601,
        mlp_hidden_dim=64,
        mlp_num_layers=2,
        latent_dim=None,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.features_per_level = features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.mlp_hidden_dim = mlp_hidden_dim
        self.mlp_num_layers = mlp_num_layers

        self.encoding = HashEncoding(
            num_levels=num_levels,
            features_per_level=features_per_level,
            log2_hashmap_size=log2_hashmap_size,
            base_resolution=base_resolution,
            finest_resolution=finest_resolution,
            latent_dim=latent_dim,
        )

        input_dim = num_levels * features_per_level
        layers = [nn.Linear(input_dim, mlp_hidden_dim), nn.ReLU()]
        for _ in range(mlp_num_layers - 1):
            layers.extend([nn.Linear(mlp_hidden_dim, mlp_hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(mlp_hidden_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, coords):
        return self.mlp(self.encoding(coords))

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def trainable_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def compressed_size_kb(self):
        """Size in KB accounting for SHACIRA compression if active."""
        hash_bytes = self.encoding.compressed_size_bytes()
        mlp_bytes = sum(p.numel() for p in self.mlp.parameters())  # INT8
        return (hash_bytes + mlp_bytes) / 1024


# -- Feature Plane + MLP (MERF-style) -----------------------------------------


class FeaturePlaneMLP(nn.Module):
    """
    Dense 2D feature grid + tiny MLP decoder for terrain.

    Terrain-specialized simplification of MERF (Reiser et al., SIGGRAPH 2023).
    Since terrain is 2D->1D, a single feature plane replaces tri-planes/hash.
    Zero hash collisions -- every grid cell stores exactly what that region needs.
    Hardware texture sampling for bilinear interpolation (near-free on any GPU).

    Literature backing: Kim & Fridovich-Keil (NeurIPS 2025) proved grids
    outperform all INR architectures for 2D bandlimited signals.
    """

    def __init__(self, resolution=256, features=4, mlp_hidden_dim=32, resolution_w=0):
        super().__init__()
        self.resolution = resolution  # height (lat rows)
        self.resolution_w = resolution_w if resolution_w > 0 else resolution  # width (lon cols)
        self.features = features
        self.mlp_hidden_dim = mlp_hidden_dim

        # Learnable 2D feature grid: (1, F, H, W) for F.grid_sample
        self.grid = nn.Parameter(
            torch.empty(1, features, self.resolution, self.resolution_w).uniform_(
                -1e-4, 1e-4
            )
        )

        self.mlp = nn.Sequential(
            nn.Linear(features, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 1),
        )

    def forward(self, coords):
        """coords: (N, 2) in [-1, 1]. Returns: (N, 1)."""
        # grid_sample expects (batch, C, H, W) input and (batch, H_out, W_out, 2) grid.
        # coords[:, 0] = x (width), coords[:, 1] = y (height), matching grid_sample's
        # convention. The grid is a learned parameter so axis assignment is implicit:
        # it learns the correct mapping during training.
        sample_grid = coords.unsqueeze(0).unsqueeze(0)  # (1, 1, N, 2)
        sampled = torch.nn.functional.grid_sample(
            self.grid,
            sample_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )  # (1, F, 1, N)
        features = sampled[0, :, 0, :].T  # (N, F)
        return self.mlp(features)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def trainable_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def grid_size_kb(self):
        """Grid storage in KB (INT8)."""
        return self.grid.numel() / 1024

    def mlp_size_kb(self):
        """MLP storage in KB (INT8)."""
        return sum(p.numel() for p in self.mlp.parameters()) / 1024


# -- Multi-Resolution Grid + MLP -----------------------------------------------


class MultiResGridEncoding(nn.Module):
    """
    Multi-resolution explicit feature grid encoding.

    Like HashEncoding but with explicit grids at each level instead of
    hash tables. Zero hash collisions -- every grid cell stores exactly
    what that spatial region needs.

    Levels use geometric spacing from base_resolution to finest_resolution.
    """

    def __init__(
        self,
        num_levels=5,
        features_per_level=4,
        base_resolution=8,
        finest_resolution=128,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.features_per_level = features_per_level
        self.base_resolution = base_resolution
        self.finest_resolution = finest_resolution
        self.output_dim = num_levels * features_per_level

        if num_levels > 1:
            self.growth_factor = math.exp(
                math.log(finest_resolution / base_resolution) / (num_levels - 1)
            )
        else:
            self.growth_factor = 1.0

        self.resolutions = [
            int(base_resolution * (self.growth_factor**level))
            for level in range(num_levels)
        ]

        # Explicit grid at each resolution level: (1, F, R_l, R_l)
        self.grids = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(1, features_per_level, res, res).uniform_(-1e-4, 1e-4)
                )
                for res in self.resolutions
            ]
        )

    def forward(self, coords):
        """coords: (N, 2) in [-1, 1]. Returns: (N, num_levels * features_per_level)."""
        sample_grid = coords.unsqueeze(0).unsqueeze(0)  # (1, 1, N, 2)
        all_features = []

        for grid in self.grids:
            sampled = torch.nn.functional.grid_sample(
                grid,
                sample_grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )  # (1, F, 1, N)
            all_features.append(sampled[0, :, 0, :].T)  # (N, F)

        return torch.cat(all_features, dim=-1)


class MultiResGridMLP(nn.Module):
    """
    Multi-resolution explicit feature grid + tiny MLP decoder.

    Collision-free alternative to HashMLP. Each resolution level stores
    an explicit 2D feature grid instead of a hash table. Bilinear
    interpolation at each level, features concatenated, fed to MLP.

    Natural LOD: skip fine levels for distant terrain.
    """

    def __init__(
        self,
        num_levels=5,
        features_per_level=4,
        base_resolution=8,
        finest_resolution=128,
        mlp_hidden_dim=32,
        mlp_num_layers=1,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.features_per_level = features_per_level
        self.mlp_hidden_dim = mlp_hidden_dim
        self.mlp_num_layers = mlp_num_layers

        self.encoding = MultiResGridEncoding(
            num_levels=num_levels,
            features_per_level=features_per_level,
            base_resolution=base_resolution,
            finest_resolution=finest_resolution,
        )

        input_dim = self.encoding.output_dim
        layers = [nn.Linear(input_dim, mlp_hidden_dim), nn.ReLU()]
        for _ in range(mlp_num_layers - 1):
            layers.extend([nn.Linear(mlp_hidden_dim, mlp_hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(mlp_hidden_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, coords):
        return self.mlp(self.encoding(coords))

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def trainable_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def grid_size_kb(self):
        """Total grid storage in KB (INT8)."""
        return sum(g.numel() for g in self.encoding.grids) / 1024

    def mlp_size_kb(self):
        """MLP storage in KB (INT8)."""
        return sum(p.numel() for p in self.mlp.parameters()) / 1024


# -- Gabor/WIRE ----------------------------------------------------------------


class GaborLayer(nn.Module):
    """
    Gabor wavelet activation: sin(omega * h) * exp(-h^2 / (2 * sigma^2))

    Provides spatial locality (Gaussian envelope) + frequency selectivity
    (sinusoidal carrier). Same parameter count as SineLayer.
    """

    def __init__(
        self, in_features, out_features, omega=20.0, sigma=5.0, is_first=False
    ):
        super().__init__()
        self.omega = omega
        self.sigma = sigma
        self.linear = nn.Linear(in_features, out_features)
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1.0 / in_features, 1.0 / in_features)
            else:
                limit = math.sqrt(6.0 / in_features) / omega
                self.linear.weight.uniform_(-limit, limit)

    def forward(self, x):
        h = self.linear(x)
        return torch.sin(self.omega * h) * torch.exp(-0.5 * (h / self.sigma) ** 2)


class GaborNet(nn.Module):
    """
    WIRE-style network: SIREN structure with Gabor wavelet activations.

    Same param count as SIREN. Only the activation function differs,
    adding spatial locality via the Gaussian envelope.
    """

    def __init__(self, hidden_dim=256, num_layers=3, omega=20.0, sigma=5.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.omega = omega
        self.sigma = sigma

        layers = [GaborLayer(2, hidden_dim, omega=omega, sigma=sigma, is_first=True)]
        for _ in range(num_layers - 1):
            layers.append(GaborLayer(hidden_dim, hidden_dim, omega=omega, sigma=sigma))
        self.net = nn.Sequential(*layers)

        self.final = nn.Linear(hidden_dim, 1)
        with torch.no_grad():
            limit = math.sqrt(6.0 / hidden_dim)
            self.final.weight.uniform_(-limit, limit)

    def forward(self, coords):
        return self.final(self.net(coords))

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def trainable_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# -- Residual MLP (HNRT) -------------------------------------------------------


class ResidualMLP(nn.Module):
    """
    Tiny MLP for learning bilinear interpolation residuals in HNRT.

    Learns: f(lat, lon) -> correction to bilinear_interp(heightmap, lat, lon)

    Supports both ReLU and SIREN activations for comparison.
    The residual signal has bounded dynamic range (unlike full terrain),
    so tiny networks (width=16) may suffice.
    """

    def __init__(self, hidden_dim=16, num_layers=3, activation="relu", omega=60.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation = activation
        self.omega = omega

        if activation == "siren":
            layers = [SineLayer(2, hidden_dim, omega=omega, is_first=True)]
            for _ in range(num_layers - 1):
                layers.append(SineLayer(hidden_dim, hidden_dim, omega=omega))
            self.net = nn.Sequential(*layers)
            self.final = nn.Linear(hidden_dim, 1)
            with torch.no_grad():
                self.final.weight.uniform_(
                    -math.sqrt(6.0 / hidden_dim) / omega,
                    math.sqrt(6.0 / hidden_dim) / omega,
                )
        else:
            layers = []
            in_dim = 2
            for _ in range(num_layers):
                layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
                in_dim = hidden_dim
            self.net = nn.Sequential(*layers)
            self.final = nn.Linear(hidden_dim, 1)

    def forward(self, coords):
        return self.final(self.net(coords))

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def trainable_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
