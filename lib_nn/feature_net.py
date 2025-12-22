import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = torch.einsum("...,b->...b", x, self.W) * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class FiLMBlock(nn.Module):
    def __init__(self):
        super(FiLMBlock, self).__init__()

    def forward(self, x, gamma, beta):
        x = gamma * x + beta
        return x


class FiLMConditioner(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FiLMConditioner, self).__init__()
        assert out_dim % 2 == 0, "Output dimension must be even for FiLM conditioning."
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(in_dim, out_dim))

        self.film_block = FiLMBlock()
        self.out_dim = out_dim

    def forward(self, x, embed):
        embed = self.mlp(embed)
        gamma, beta = embed.chunk(2, dim=-1)
        x = self.film_block(x, gamma, beta)
        return x


## Time and Latent Code conditioned Feature encoder ##
class TimeLatentFeatureEncoder(nn.Module):
    def __init__(
        self,
        dims,
        in_dim=3,
        enc_dim=256,
        activation="relu",
        out_dim=1,
        latent_size=256,
        latent_in=(),
        cond_fn_type_in="cc",
        cond_fn_type="cc",
        dropout=None,
    ):
        super(TimeLatentFeatureEncoder, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = activation
        self.latent_size = latent_size
        self.latent_in = latent_in

        if dropout is None:
            dropout = dict()
        self.dropout_layers_p = dropout.get("layers_prob", None)
        self.dropout_layers = dropout.get("layers", [])

        self.cond_fn_type_in = cond_fn_type_in
        self.cond_fn_type = cond_fn_type
        if cond_fn_type_in == "cc":
            print("Using concatenated latent for input")
            dims0 = [latent_size + enc_dim + in_dim]
        elif cond_fn_type_in == "film":
            print("Using FiLM conditioning for input")
            self.cond_fn_in = FiLMConditioner(latent_size, enc_dim * 2)
            dims0 = [enc_dim + in_dim]
        else:
            raise ValueError(f"Unknown conditioning function: {cond_fn_type_in}")

        dims = dims0 + dims + [out_dim]
        self.num_layers = len(dims) - 1

        # timestep Embedding
        self.time_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=enc_dim),
            nn.Linear(enc_dim, enc_dim),
            nn.SiLU(),
        )
        # point Embedding
        self.x_embed = nn.Sequential(
            nn.Linear(in_dim, enc_dim),
            nn.SiLU(),
        )

        self.cond_fns = nn.ModuleList()
        for layer in range(0, self.num_layers):
            in_dim = dims[layer]
            out_dim = dims[layer + 1]
            if layer in latent_in:
                if cond_fn_type == "cc":
                    in_dim += dims[0]
                elif cond_fn_type == "film":
                    self.cond_fns.append(FiLMConditioner(dims[0], in_dim * 2))
                else:
                    raise ValueError(f"Unknown conditioning function: {cond_fn_type}")
            else:
                if cond_fn_type == "film":
                    self.cond_fns.append(None)

            # weight_norm-ed linear layers
            setattr(
                self,
                "lin" + str(layer),
                nn.utils.weight_norm(nn.Linear(in_dim, out_dim)),
            )

        assert self.activation in ["relu"]

    # input: N x (L+3)
    def forward(self, x_in, timesteps, latent_vecs):
        # if self.dropout_latent_p is not None:
        #     latent_vecs = F.dropout(
        #         latent_vecs, p=self.dropout_latent_p, training=self.training
        #     )

        # encode timestep
        t_emb = self.time_embed(timesteps)
        # encode point
        x_emb = self.x_embed(x_in)

        xt_emb = x_emb + t_emb
        if self.cond_fn_type_in == "cc":
            x = torch.cat([latent_vecs, xt_emb, x_in], -1)
        elif self.cond_fn_type_in == "film":
            xt_emb = self.cond_fn_in(xt_emb, latent_vecs)
            x = torch.cat([xt_emb, x_in], -1)
        else:
            raise ValueError(f"Unknown conditioning function: {self.cond_fn_type_in}")
        x0 = x.clone()

        for layer in range(0, self.num_layers):
            if layer in self.latent_in:
                if self.cond_fn_type == "cc":
                    x = torch.cat([x, x0], -1)
                elif self.cond_fn_type == "film":
                    x = self.cond_fns[layer](x, x0)
                else:
                    raise ValueError(
                        f"Unknown conditioning function: {self.cond_fn_type}"
                    )

            lin = getattr(self, "lin" + str(layer))
            x = lin(x)

            if layer < self.num_layers - 1:

                # apply activation
                if self.activation == "relu":
                    x = x.relu()
                elif self.activation == "sine":
                    x = (x * 30).sin()  # const=30 from siren paper
                else:
                    raise ValueError(f"Unknown activation function: {self.activation}")

                if layer in self.dropout_layers:
                    x = F.dropout(x, p=self.dropout_layers_p, training=self.training)

        x = x.tanh()
        return x


if __name__ == "__main__":

    B = 10
    N = 30
    x_dim = 3
    lat_dim = 128
    out_dim = 7

    def evaluate(model):
        x_in = torch.randn((B, N, x_dim))
        t_in = torch.rand_like(x_in[..., 0])
        lat_in = torch.randn((B, N, lat_dim))
        z_out = model(x_in, t_in, lat_in)
        print(z_out.shape)

    model = TimeLatentFeatureEncoder(
        enc_dim=132, latent_size=lat_dim, dims=[512, 512], out_dim=out_dim
    )
    evaluate(model)
