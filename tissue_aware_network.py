import torch
import torch.nn as nn
import torch.nn.functional as F
import math

TAN_CHANNELS = [32, 64, 128, 128]
NUM_ATTN_HEADS = 8
ATTN_HIDDEN_DIM = 256
GROUPNORM_GROUPS = 8
TAN_DROPOUT_P = 0.1
CLASSIFIER_HIDDEN_DIM = 128


class MaskedMaxPool2d(nn.Module):

    def __init__(self, kernel_size=2):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x, mask):
        mask_b = mask.unsqueeze(1) > 0.5  # (B, 1, H, W), bool
        neg_inf = torch.finfo(x.dtype).min
        x_filled = torch.where(mask_b, x, torch.full_like(x, neg_inf))
        x_pooled = F.max_pool2d(x_filled, self.kernel_size)
        mask_pooled = F.max_pool2d(mask.unsqueeze(1), self.kernel_size).squeeze(1)
        fully_invalid = (mask_pooled.unsqueeze(1) <= 0.5)
        x_pooled = torch.where(fully_invalid, torch.zeros_like(x_pooled), x_pooled)
        return x_pooled, mask_pooled


class MaskedGroupNorm(nn.Module):

    def __init__(self, num_groups, num_channels, eps=1e-5):
        super().__init__()
        assert num_channels % num_groups == 0, \
            f"num_channels={num_channels} not divisible by num_groups={num_groups}"
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x, mask):
        B, C, H, W = x.shape
        G = self.num_groups
        cpg = C // G  # channels per group
        x_g = x.view(B, G, cpg, H, W)
        m = mask.view(B, 1, 1, H, W).to(x.dtype)  # broadcasts across groups/channels-per-group

        n_valid_pixels = mask.view(B, -1).sum(dim=1).view(B, 1)  # (B, 1), same for every group
        denom = n_valid_pixels * cpg + self.eps

        sum_x = (x_g * m).sum(dim=(2, 3, 4))            # (B, G)
        sum_x2 = ((x_g ** 2) * m).sum(dim=(2, 3, 4))    # (B, G)
        mean = sum_x / denom
        var = sum_x2 / denom - mean ** 2
        var = var.clamp(min=0.0)  # guard tiny negative values from floating-point error

        mean = mean.view(B, G, 1, 1, 1)
        std = (var.view(B, G, 1, 1, 1) + self.eps).sqrt()
        x_norm = (x_g - mean) / std
        x_norm = x_norm.view(B, C, H, W)
        return x_norm * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class SolidConv(nn.Module):

    def __init__(self, in_channels, out_channels, dropout_p=TAN_DROPOUT_P):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_tan = nn.Parameter(torch.empty(1, out_channels, in_channels, 9))
        self.bias = nn.Parameter(torch.empty(out_channels))
        num_groups = min(GROUPNORM_GROUPS, out_channels)
        while out_channels % num_groups != 0:
            num_groups -= 1
        self.norm = MaskedGroupNorm(num_groups, out_channels)
        self.dropout = nn.Dropout2d(p=dropout_p)
        nn.init.kaiming_uniform_(self.w_tan, a=math.sqrt(5))
        nn.init.zeros_(self.bias)

    def forward(self, x, mask):
        B, C, H, W = x.shape
        mask_pad = F.pad(mask.unsqueeze(1), (1, 1, 1, 1), value=0.0)
        mask_unf = mask_pad.unfold(2, 3, 1).unfold(3, 3, 1)
        solid = mask_unf.reshape(B, H, W, 9).min(dim=-1).values
        solid_v = solid.reshape(-1).bool()

        x_pad = F.pad(x, (1, 1, 1, 1), value=0.0)
        x_unf = x_pad.unfold(2, 3, 1).unfold(3, 3, 1)
        x_unf = x_unf.reshape(B, C, H, W, 9)
        x_unf = x_unf[..., [4, 0, 1, 2, 3, 5, 6, 7, 8]]
        d_flat = x_unf.permute(0, 2, 3, 1, 4).reshape(-1, C, 9)

        out_flat = torch.zeros(B * H * W, self.out_channels, device=x.device)
        if solid_v.any():
            out_flat[solid_v] = torch.einsum(
                'bcn,mcn->bm', d_flat[solid_v], self.w_tan[0]
            )

        out = out_flat.view(B, H, W, self.out_channels).permute(0, 3, 1, 2)
        out = F.leaky_relu(self.norm(out + self.bias.view(1, -1, 1, 1), mask))
        out = self.dropout(out)
        return out * mask.unsqueeze(1), mask


class MultiHeadAttentionPooling(nn.Module):

    def __init__(self, tan_dim, num_heads=NUM_ATTN_HEADS, hidden_dim=ATTN_HIDDEN_DIM):
        super().__init__()
        self.num_heads = num_heads
        self.tan_dim = tan_dim
        self.attn = nn.Sequential(
            nn.Linear(tan_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_heads, bias=False),
        )

    def forward(self, x, mask):
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        mask_flat = mask.reshape(B, H * W) > 0.5  # (B, H*W), True = valid
        scores = self.attn(x_flat).permute(0, 2, 1)  # (B, num_heads, H*W)

        all_invalid = ~mask_flat.any(dim=1)  # (B,) -- degenerate-case guard
        mask_for_fill = mask_flat.clone()
        if all_invalid.any():
            mask_for_fill[all_invalid] = True  # treat as "all valid" for these samples only

        mask_expand = mask_for_fill.unsqueeze(1).expand(-1, self.num_heads, -1)  # (B, heads, H*W)
        neg_inf = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~mask_expand, neg_inf)

        attn_w = F.softmax(scores, dim=-1)
        head_embs = torch.bmm(attn_w, x_flat)
        return head_embs.reshape(B, self.num_heads * C)


class TissueAwareNetwork(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        ch = TAN_CHANNELS
        self.tan1 = SolidConv(3, ch[0])
        self.pool1 = MaskedMaxPool2d(2)
        self.tan2 = SolidConv(ch[0], ch[1])
        self.pool2 = MaskedMaxPool2d(2)
        self.tan3 = SolidConv(ch[1], ch[2])
        self.pool3 = MaskedMaxPool2d(2)
        self.tan4 = SolidConv(ch[2], ch[3])
        self.pool4 = MaskedMaxPool2d(2)
        self.attn = MultiHeadAttentionPooling(tan_dim=ch[-1])

        attn_out = ch[-1] * NUM_ATTN_HEADS
        self.embedding_dim = attn_out
        self.classifier = nn.Sequential(
            nn.Linear(attn_out, CLASSIFIER_HIDDEN_DIM),
            nn.LayerNorm(CLASSIFIER_HIDDEN_DIM),
            nn.SiLU(),
            nn.Dropout(p=0.3),
            nn.Linear(CLASSIFIER_HIDDEN_DIM, num_classes),
        )

    def forward(self, x, mask):
        x, mask = self.tan1(x, mask); x, mask = self.pool1(x, mask)
        x, mask = self.tan2(x, mask); x, mask = self.pool2(x, mask)
        x, mask = self.tan3(x, mask); x, mask = self.pool3(x, mask)
        x, mask = self.tan4(x, mask); x, mask = self.pool4(x, mask)
        emb = self.attn(x, mask)
        return self.classifier(emb), emb


TAN = TissueAwareNetwork
