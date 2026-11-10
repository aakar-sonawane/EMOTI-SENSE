"""
Attention-Based Fusion with Adaptive Learning for CMU-MOSI  (AUDIO + VIDEO only)
=============================================================
Multimodal sentiment regression (audio + video) using:
  1. Per-modality encoders (BiLSTM/Linear projections)
  2. Cross-modal multi-head attention (audio attends video, video attends audio)
  3. Adaptive fusion gate -- a small network that LEARNS, per sample,
     how much weight to give each modality's fused representation
     before combining them into the final prediction.

Feature dims in this project (from the extraction cell):
  - audio (librosa MFCC + delta + ZCR + RMS) : 28
  - video (48x48 face crop, flattened)        : 2304
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# 1. Per-modality encoder: projects raw features into a shared d_model space
# ----------------------------------------------------------------------
class ModalityEncoder(nn.Module):
    """BiLSTM encoder + linear projection into common embedding space."""

    def __init__(self, input_dim, d_model, hidden_dim=128, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(hidden_dim * 2, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths=None):
        # x: (B, T, input_dim)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        else:
            out, _ = self.lstm(x)  # (B, T, 2*hidden_dim)
        out = self.proj(out)  # (B, T, d_model)
        out = self.norm(out)
        return self.dropout(out)


# ----------------------------------------------------------------------
# 2. Cross-modal attention block: modality A (query) attends to modality B (key/value)
# ----------------------------------------------------------------------
class CrossModalAttention(nn.Module):
    """Standard multi-head cross attention: query modality attends over context modality."""

    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, query, context, key_padding_mask=None):
        attn_out, attn_weights = self.mha(
            query, context, context, key_padding_mask=key_padding_mask
        )
        x = self.norm(query + attn_out)          # residual + norm
        x = self.norm2(x + self.ffn(x))           # feed-forward + residual + norm
        return x, attn_weights


# ----------------------------------------------------------------------
# 3. Adaptive Fusion Gate -- learns per-sample modality importance weights
# ----------------------------------------------------------------------
class AdaptiveFusionGate(nn.Module):
    """
    Takes pooled representations from each modality stream and learns
    scalar gate weights (summing to 1 via softmax) that adapt PER SAMPLE
    based on the content of that sample.
    """

    def __init__(self, d_model, num_modalities=2, hidden_dim=64):
        super().__init__()
        self.num_modalities = num_modalities
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * num_modalities, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_modalities),
        )

    def forward(self, modality_reprs):
        # modality_reprs: list of (B, d_model) tensors, one per modality
        concat = torch.cat(modality_reprs, dim=-1)          # (B, d_model * M)
        gate_logits = self.gate_net(concat)                  # (B, M)
        gate_weights = F.softmax(gate_logits, dim=-1)         # (B, M) sums to 1

        stacked = torch.stack(modality_reprs, dim=1)          # (B, M, d_model)
        fused = torch.sum(gate_weights.unsqueeze(-1) * stacked, dim=1)  # (B, d_model)
        return fused, gate_weights


# ----------------------------------------------------------------------
# 4. Full Model: Attention-Based Fusion with Adaptive Learning (audio + video)
# ----------------------------------------------------------------------
class AttentiveAdaptiveFusionModel(nn.Module):
    def __init__(
        self,
        audio_dim=28,
        video_dim=2304,
        d_model=128,
        num_heads=4,
        dropout=0.2,
    ):
        super().__init__()

        # --- per-modality encoders ---
        self.audio_encoder = ModalityEncoder(audio_dim, d_model, dropout=dropout)
        self.video_encoder = ModalityEncoder(video_dim, d_model, dropout=dropout)

        # --- cross-modal attention: audio <-> video ---
        self.audio_attends_video = CrossModalAttention(d_model, num_heads, dropout)
        self.video_attends_audio = CrossModalAttention(d_model, num_heads, dropout)

        # --- adaptive fusion gate (2 modalities) ---
        self.fusion_gate = AdaptiveFusionGate(d_model, num_modalities=2)

        # --- prediction head ---
        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),   # CMU-MOSI: scalar sentiment score in [-3, 3]
        )

    @staticmethod
    def _masked_mean_pool(x, mask=None):
        # x: (B, T, d_model), mask: (B, T) True where PAD
        if mask is None:
            return x.mean(dim=1)
        lengths = (~mask).sum(dim=1, keepdim=True).clamp(min=1)
        x = x.masked_fill(mask.unsqueeze(-1), 0.0)
        return x.sum(dim=1) / lengths

    def forward(self, audio, video, audio_mask=None, video_mask=None):
        """
        audio: (B, T_a, audio_dim)
        video: (B, T_v, video_dim)
        *_mask: (B, T) boolean, True at PAD positions (optional)
        """
        # 1) encode each modality into shared space
        a = self.audio_encoder(audio)  # (B, T_a, d_model)
        v = self.video_encoder(video)  # (B, T_v, d_model)

        # 2) cross-modal attention (each modality queries the other)
        a_fused, _ = self.audio_attends_video(a, v, key_padding_mask=video_mask)
        v_fused, _ = self.video_attends_audio(v, a, key_padding_mask=audio_mask)

        # 3) pool each modality's sequence into a single vector
        a_pooled = self._masked_mean_pool(a_fused, audio_mask)
        v_pooled = self._masked_mean_pool(v_fused, video_mask)

        # 4) adaptive fusion -- learns per-sample modality weighting
        fused_repr, gate_weights = self.fusion_gate([a_pooled, v_pooled])

        # 5) prediction
        pred = self.regressor(fused_repr).squeeze(-1)  # (B,)

        return pred, gate_weights  # return gate_weights too, useful for analysis/interpretability


# ----------------------------------------------------------------------
# Quick shape sanity check (run: python model.py)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    B = 8
    T_a, T_v = 50, 50
    AUDIO_DIM, VIDEO_DIM = 28, 2304

    model = AttentiveAdaptiveFusionModel(
        audio_dim=AUDIO_DIM, video_dim=VIDEO_DIM, d_model=128
    )

    audio = torch.randn(B, T_a, AUDIO_DIM)
    video = torch.randn(B, T_v, VIDEO_DIM)

    pred, gates = model(audio, video)
    print("Prediction shape:", pred.shape)          # (B,)
    print("Gate weights shape:", gates.shape)        # (B, 2)
    print("Sample gate weights (audio, video):\n", gates[0])
    print("Total trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
