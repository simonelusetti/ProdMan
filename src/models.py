import torch
import torch.nn as nn


class ProductProjector(nn.Module):
    """Feed-forward projection module used to isolate latent subspaces."""

    def __init__(self, latent_dim: int, hidden_dim: int):
        super().__init__()
        layers = []
        if hidden_dim > 0:
            layers.append(nn.Linear(latent_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, latent_dim))
        else:
            layers.append(nn.Linear(latent_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ProductManifoldModel(nn.Module):
    """Token-level product-manifold projection module."""

    def __init__(self, cfg, input_dim: int):
        super().__init__()
        self.cfg = cfg
        model_cfg = cfg.product_model
        latent_dim = int(model_cfg.latent_dim)
        hidden = int(model_cfg.encoder_hidden)
        projector_hidden = int(model_cfg.projector_hidden)
        self.num_subspaces = int(model_cfg.num_subspaces)

        if self.num_subspaces < 1:
            raise ValueError("product_model.num_subspaces must be >= 1")

        self.latent_dim = latent_dim
        self.encoder = self._build_mlp(input_dim, hidden, latent_dim)
        self.token_decoder = self._build_mlp(latent_dim, hidden, input_dim)
        self.projectors = nn.ModuleList(
            ProductProjector(latent_dim, projector_hidden) for _ in range(self.num_subspaces)
        )

    @staticmethod
    def _build_mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Module:
        layers = []
        if hidden_dim > 0:
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, output_dim),
                ]
            )
        else:
            layers.append(nn.Linear(input_dim, output_dim))
        return nn.Sequential(*layers)

    def _encode_latents(
        self, token_embeddings: torch.Tensor, mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        z_hat = self.encoder(token_embeddings)
        projected = [proj(z_hat) for proj in self.projectors]
        subspaces = torch.stack(projected, dim=2)  # [B, T, K, D]
        aggregated = subspaces.sum(dim=2)  # [B, T, D]
        aggregated = aggregated * mask.unsqueeze(-1).type_as(aggregated)

        mask_float = mask.unsqueeze(-1).unsqueeze(-1).type_as(subspaces)
        masked_subspaces = subspaces * mask_float
        token_counts = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        factor_means = masked_subspaces.sum(dim=1) / token_counts.unsqueeze(-1)

        return {
            "z_hat": z_hat,
            "subspaces": subspaces,
            "aggregated_tokens": aggregated,
            "factors": factor_means,
        }

    def decode_tokens(self, aggregated_tokens: torch.Tensor) -> torch.Tensor:
        return self.token_decoder(aggregated_tokens)

    def forward(self, token_embeddings: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        latents = self._encode_latents(token_embeddings, mask)
        token_reconstruction = self.decode_tokens(latents["aggregated_tokens"])

        mask_float = mask.unsqueeze(-1).type_as(token_reconstruction)
        sentence_reconstruction = (
            (token_reconstruction * mask_float).sum(dim=1)
            / mask_float.sum(dim=1).clamp_min(1.0)
        )

        latents.update(
            {
                "token_reconstruction": token_reconstruction,
                "sentence_reconstruction": sentence_reconstruction,
                "mask": mask,
            }
        )
        return latents

    def encode_tokens(self, token_embeddings: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        return self._encode_latents(token_embeddings, mask)
