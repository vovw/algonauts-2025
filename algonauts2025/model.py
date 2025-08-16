# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import numpy as np
import pydantic
import torch
import torch.nn.functional as F
from data_utils.dataloader import SegmentData
from einops import rearrange
from modeling_utils.models.common import MlpConfig, SubjectLayers
from modeling_utils.models.transformer import TransformerEncoderConfig
from torch import nn


class FmriEncoderConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    name: tp.Literal["FmriEncoder"] = "FmriEncoder"
    n_subjects: int | None = None
    feature_aggregation: tp.Literal["sum", "cat"] = "cat"
    layer_aggregation: tp.Literal["mean", "cat"] = "cat"
    subject_embedding: bool = False
    modality_dropout: float = 0.0

    # Contrastive alignment (e.g., with VJEPA2 video features)
    contrastive_enabled: bool = False
    contrastive_modalities: list[str] = ["video"]
    contrastive_weight: float = 0.1
    contrastive_temperature: float = 0.07

    def build(
        self, feature_dims: dict[int], n_outputs: int, n_output_timesteps: int
    ) -> nn.Module:
        return FmriEncoder(
            feature_dims,
            n_outputs,
            n_output_timesteps,
            config=self,
        )


class FmriEncoder(nn.Module):
    def __init__(
        self,
        feature_dims: dict[str, tuple[int, int]],
        n_outputs: int,
        n_output_timesteps: int,
        config: FmriEncoderConfig,
    ):
        super().__init__()
        self.config = config
        self.feature_dims = feature_dims
        self.n_outputs = n_outputs
        self.projectors = nn.ModuleDict()
        self.contrastive_heads = nn.ModuleDict()
        self.pooler = nn.AdaptiveAvgPool1d(n_output_timesteps)
        hidden = 3072
        for modality, tup in feature_dims.items():
            if tup is None:
                print(
                    f"Warning: {modality} has no feature dimensions. Skipping projector."
                )
                continue
            else:
                num_layers, feature_dim = tup
            input_dim = (
                feature_dim * num_layers
                if config.layer_aggregation == "cat"
                else feature_dim
            )
            output_dim = (
                hidden // len(feature_dims)
                if config.feature_aggregation == "cat"
                else hidden
            )
            self.projectors[modality] = MlpConfig(
                norm_layer="layer", activation_layer="gelu", dropout=0.0
            ).build(input_dim, output_dim)

            # Build contrastive projection heads to map raw modality embeddings to hidden
            # dimension independently of feature aggregation. Only for selected modalities.
            if (
                self.config.contrastive_enabled
                and modality in self.config.contrastive_modalities
            ):
                self.contrastive_heads[modality] = MlpConfig(
                    norm_layer="layer", activation_layer="gelu", dropout=0.0
                ).build(input_dim, hidden)
        input_dim = (
            (hidden // len(feature_dims)) * len(feature_dims)
            if config.feature_aggregation == "cat"
            else hidden
        )
        self.combiner = nn.Identity()
        self.predictor = SubjectLayers(
            in_channels=hidden,
            out_channels=n_outputs,
            n_subjects=config.n_subjects,
            average_subjects=False,
            bias=True,
        )
        self.time_pos_embed = nn.Parameter(torch.randn(1, 1024, hidden))
        if config.subject_embedding:
            self.subject_embed = nn.Embedding(config.n_subjects, hidden)
        self.encoder = TransformerEncoderConfig(
            attn_dropout=0.0, ff_dropout=0.0, layer_dropout=0.0, depth=8
        ).build(dim=hidden)

    def forward(self, batch: SegmentData, pool_outputs: bool = True) -> torch.Tensor:
        x = self.aggregate_features(batch)  # B, T, H
        subject_id = batch.data.get("subject_id", None)
        x = self.transformer_forward(x, subject_id)
        x = x.transpose(1, 2)  # B, H, T
        x = self.predictor(x, subject_id)  # B, O, T
        if pool_outputs:
            out = self.pooler(x)  # B, O, T'
        else:
            out = x
        return out

    def aggregate_features(self, batch):
        tensors = []
        # get B, T
        for modality in batch.data.keys():
            if modality in self.feature_dims:
                break
        x = batch.data[modality]
        B, T = x.shape[0], x.shape[-1]
        # select the modalities to dropout, keep at least one modality
        modalities_to_dropout = []
        for modality in self.feature_dims.keys():
            if torch.rand(1).item() < self.config.modality_dropout and self.training:
                modalities_to_dropout.append(modality)
        if len(modalities_to_dropout) == len(self.feature_dims):
            modalities_to_dropout = np.random.choice(
                modalities_to_dropout, len(modalities_to_dropout) - 1, replace=False
            )
        for modality in self.feature_dims.keys():
            if modality not in self.projectors:
                data = torch.zeros(B, T, 3072 // len(self.feature_dims)).to(x.device)
            else:
                data = batch.data[modality]  # B, L, H, T
                data = data.to(torch.float32)
                if data.ndim == 3:
                    data = data.unsqueeze(1)
                # mean over layers
                if self.config.layer_aggregation == "mean":
                    data = data.mean(dim=1)
                elif self.config.layer_aggregation == "cat":
                    data = rearrange(data, "b l d t -> b (l d) t")
                data = data.transpose(1, 2)
                assert data.ndim == 3  # B, T, D
                data = self.projectors[modality](data)  # B, T, H
                if modality in modalities_to_dropout:
                    data = torch.zeros_like(data)
            tensors.append(data)
        if self.config.feature_aggregation == "cat":
            out = torch.cat(tensors, dim=-1)
        elif self.config.feature_aggregation == "sum":
            out = sum(tensors)
        return out

    def transformer_forward(self, x, subject_id=None):
        x = self.combiner(x)
        if hasattr(self, "time_pos_embed"):
            x = x + self.time_pos_embed[:, : x.size(1)]
        if hasattr(self, "subject_embed"):
            x = x + self.subject_embed(subject_id)
        x = self.encoder(x)
        return x

    # --- Contrastive alignment helpers ---
    def get_brain_latents(self, batch: SegmentData) -> torch.Tensor:
        """Return sequence latents before predictor: shape [B, T, H]."""
        x = self.aggregate_features(batch)  # B, T, H (hidden)
        subject_id = batch.data.get("subject_id", None)
        x = self.transformer_forward(x, subject_id)
        return x

    def _prepare_single_modality(self, batch: SegmentData, modality: str) -> torch.Tensor:
        """Prepare a single modality tensor [B, T, D_mod] with layer aggregation."""
        data = batch.data.get(modality, None)
        if data is None:
            raise KeyError(f"Modality '{modality}' not found in batch.data")
        data = data.to(torch.float32)
        if data.ndim == 3:
            data = data.unsqueeze(1)  # B, 1, D, T
        if self.config.layer_aggregation == "mean":
            data = data.mean(dim=1)
        elif self.config.layer_aggregation == "cat":
            data = rearrange(data, "b l d t -> b (l d) t")
        data = data.transpose(1, 2)  # B, T, D_mod
        return data

    def get_modality_latents(self, batch: SegmentData, modality: str) -> torch.Tensor:
        """Project a single modality to hidden dimension: [B, T, H]."""
        assert (
            modality in self.contrastive_heads
        ), f"No contrastive head found for modality '{modality}'"
        data = self._prepare_single_modality(batch, modality)
        proj = self.contrastive_heads[modality](data)  # B, T, H
        return proj

    @staticmethod
    def _info_nce(q: torch.Tensor, k: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
        """
        Symmetric InfoNCE over flattened [B,T,H] sequences.
        q, k: [B, T, H]
        """
        bt, h = q.shape[0] * q.shape[1], q.shape[2]
        q = F.normalize(q.reshape(bt, h), dim=-1)
        k = F.normalize(k.reshape(bt, h), dim=-1)
        logits = (q @ k.t()) / tau
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_qk = F.cross_entropy(logits, labels)
        loss_kq = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_qk + loss_kq)

    def compute_contrastive_loss(self, batch: SegmentData) -> dict[str, torch.Tensor]:
        """Compute InfoNCE losses per selected modality against brain latents."""
        if not self.config.contrastive_enabled:
            return {}
        tau = self.config.contrastive_temperature
        brain_latents = self.get_brain_latents(batch)  # B, T, H
        losses: dict[str, torch.Tensor] = {}
        for modality in self.config.contrastive_modalities:
            if modality not in self.contrastive_heads or modality not in batch.data:
                continue
            mod_latents = self.get_modality_latents(batch, modality)
            # If time dims mismatch, pool to match brain T
            if mod_latents.size(1) != brain_latents.size(1):
                mod_latents_t = mod_latents.transpose(1, 2)  # B, H, Tm
                pool = nn.AdaptiveAvgPool1d(brain_latents.size(1))
                mod_latents = pool(mod_latents_t).transpose(1, 2)
            loss = self._info_nce(brain_latents, mod_latents, tau=tau)
            losses[modality] = loss
        return losses
