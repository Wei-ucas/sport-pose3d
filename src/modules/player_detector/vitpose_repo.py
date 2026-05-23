import runpy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn


def _to_2tuple(value):
    if isinstance(value, tuple):
        return value
    return (value, value)


def _drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


def _trunc_normal_(tensor: torch.Tensor, std: float = 0.02):
    if hasattr(nn.init, "trunc_normal_"):
        return nn.init.trunc_normal_(tensor, std=std)
    return nn.init.normal_(tensor, std=std)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor):
        return _drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MoEMlp(nn.Module):
    def __init__(
        self,
        num_expert: int,
        in_features: int,
        hidden_features: int,
        out_features: int,
        part_features: int,
        drop: float = 0.0,
    ):
        super().__init__()
        self.part_features = part_features
        self.num_expert = num_expert
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features - part_features)
        self.drop = nn.Dropout(drop)
        self.experts = nn.ModuleList([nn.Linear(hidden_features, part_features) for _ in range(num_expert)])

    def forward(self, x: torch.Tensor, indices: torch.Tensor):
        expert_x = torch.zeros_like(x[:, :, -self.part_features:])
        x = self.fc1(x)
        x = self.act(x)
        shared_x = self.fc2(x)
        indices = indices.view(-1, 1, 1)
        for expert_idx in range(self.num_expert):
            selected = (indices == expert_idx).to(dtype=x.dtype)
            current = self.experts[expert_idx](x) * selected
            expert_x = expert_x + current
        x = torch.cat([shared_x, expert_x], dim=-1)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = False, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor):
        batch_size, token_count, channel_count = x.shape
        qkv = self.qkv(x).reshape(batch_size, token_count, 3, self.num_heads, channel_count // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        attention = (query * self.scale) @ key.transpose(-2, -1)
        attention = attention.softmax(dim=-1)
        attention = self.attn_drop(attention)
        x = (attention @ value).transpose(1, 2).reshape(batch_size, token_count, channel_count)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        drop: float,
        attn_drop: float,
        drop_path: float,
        num_expert: int,
        part_features: int,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MoEMlp(
            num_expert=num_expert,
            in_features=dim,
            hidden_features=hidden_features,
            out_features=dim,
            part_features=part_features,
            drop=drop,
        )

    def forward(self, x: torch.Tensor, indices: torch.Tensor):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x), indices))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size: Tuple[int, int], patch_size: int, in_chans: int, embed_dim: int, ratio: int = 1):
        super().__init__()
        img_size = _to_2tuple(img_size)
        patch_size = _to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_shape = (int(img_size[0] // patch_size[0] * ratio), int(img_size[1] // patch_size[1] * ratio))
        self.num_patches = self.patch_shape[0] * self.patch_shape[1]
        padding = 4 + 2 * (ratio // 2 - 1)
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=(patch_size[0] // ratio),
            padding=padding,
        )

    def forward(self, x: torch.Tensor):
        x = self.proj(x)
        patch_h, patch_w = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        return x, (patch_h, patch_w)


class ViTMoE(nn.Module):
    def __init__(
        self,
        img_size: Tuple[int, int],
        patch_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        drop_path_rate: float,
        num_expert: int,
        part_features: int,
        ratio: int = 1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim, ratio=ratio)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=0.0,
                    attn_drop=0.0,
                    drop_path=dpr[idx],
                    num_expert=num_expert,
                    part_features=part_features,
                )
                for idx in range(depth)
            ]
        )
        self.last_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        _trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor, dataset_source: torch.Tensor):
        batch_size = x.shape[0]
        x, (patch_h, patch_w) = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1]
        for block in self.blocks:
            x = block(x, dataset_source)
        x = self.last_norm(x)
        x = x.permute(0, 2, 1).reshape(batch_size, -1, patch_h, patch_w).contiguous()
        return x


class TopdownHeatmapSimpleHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, deconv_filters: List[int], deconv_kernels: List[int]):
        super().__init__()
        layers: List[nn.Module] = []
        current_channels = in_channels
        for out_channel, kernel_size in zip(deconv_filters, deconv_kernels):
            padding = 1 if kernel_size == 4 else 0
            output_padding = 0 if kernel_size == 4 else 1
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=current_channels,
                    out_channels=out_channel,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(out_channel))
            layers.append(nn.ReLU(inplace=True))
            current_channels = out_channel
        self.deconv_layers = nn.Sequential(*layers)
        self.final_layer = nn.Conv2d(current_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor):
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        return x


class ViTPoseTopDownMoE(nn.Module):
    def __init__(self, backbone: ViTMoE, keypoint_head: TopdownHeatmapSimpleHead, associate_keypoint_heads: List[TopdownHeatmapSimpleHead]):
        super().__init__()
        self.backbone = backbone
        self.keypoint_head = keypoint_head
        self.associate_keypoint_heads = nn.ModuleList(associate_keypoint_heads)

    def forward(self, x: torch.Tensor, dataset_source: torch.Tensor):
        features = self.backbone(x, dataset_source)
        head_idx = int(dataset_source[0].item())
        if head_idx == 0:
            return self.keypoint_head(features)
        return self.associate_keypoint_heads[head_idx - 1](features)


@dataclass
class ViTPoseRepoConfig:
    input_size: Tuple[int, int] = (192, 256)
    dataset_index: int = 5
    num_heads: Optional[int] = None
    drop_path_rate: float = 0.0


def _load_overrides(config_path: Optional[str]) -> Dict:
    if not config_path:
        return {}
    namespace = runpy.run_path(config_path)
    for key in ("VITPOSE_REPO_MODEL", "VITPOSE_MODEL"):
        if key in namespace and isinstance(namespace[key], dict):
            return namespace[key]
    return {}


def _extract_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    else:
        state_dict = checkpoint
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            cleaned[key[7:]] = value
        else:
            cleaned[key] = value
    return cleaned


def _infer_num_heads(embed_dim: int, override: Optional[int]) -> int:
    if override is not None:
        return int(override)
    default_map = {384: 12, 768: 12, 1024: 16, 1280: 16}
    if embed_dim not in default_map:
        raise ValueError(f"Unable to infer ViTPose num_heads for embed_dim={embed_dim}. Provide it in pose_config.")
    return default_map[embed_dim]


def _infer_head_cfg(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict:
    conv_indices = []
    for key in state_dict.keys():
        if key.startswith(f"{prefix}.deconv_layers.") and key.endswith(".weight"):
            layer_idx = int(key.split(".")[-2])
            if layer_idx not in conv_indices and state_dict[key].ndim == 4:
                conv_indices.append(layer_idx)
    conv_indices = sorted(idx for idx in conv_indices if idx % 3 == 0)
    deconv_filters = []
    deconv_kernels = []
    in_channels = None
    for idx in conv_indices:
        weight = state_dict[f"{prefix}.deconv_layers.{idx}.weight"]
        if in_channels is None:
            in_channels = weight.shape[0]
        deconv_filters.append(weight.shape[1])
        deconv_kernels.append(weight.shape[2])
    out_channels = state_dict[f"{prefix}.final_layer.weight"].shape[0]
    if in_channels is None:
        in_channels = state_dict[f"{prefix}.final_layer.weight"].shape[1]
    return {
        "in_channels": int(in_channels),
        "out_channels": int(out_channels),
        "deconv_filters": [int(value) for value in deconv_filters],
        "deconv_kernels": [int(value) for value in deconv_kernels],
    }


def _build_model_from_state_dict(state_dict: Dict[str, torch.Tensor], config: ViTPoseRepoConfig) -> ViTPoseTopDownMoE:
    block_indices = sorted(
        {
            int(key.split(".")[2])
            for key in state_dict.keys()
            if key.startswith("backbone.blocks.") and key.split(".")[2].isdigit()
        }
    )
    depth = len(block_indices)
    embed_dim = int(state_dict["backbone.pos_embed"].shape[-1])
    patch_size = int(state_dict["backbone.patch_embed.proj.weight"].shape[-1])
    fc1_weight = state_dict["backbone.blocks.0.mlp.fc1.weight"]
    mlp_ratio = fc1_weight.shape[0] / fc1_weight.shape[1]
    expert_prefix = "backbone.blocks.0.mlp.experts."
    expert_indices = sorted(
        {
            int(key.split(".")[5])
            for key in state_dict.keys()
            if key.startswith(expert_prefix) and key.endswith(".weight")
        }
    )
    num_expert = len(expert_indices)
    part_features = int(state_dict[f"{expert_prefix}0.weight"].shape[0])
    num_heads = _infer_num_heads(embed_dim, config.num_heads)
    backbone = ViTMoE(
        img_size=(config.input_size[1], config.input_size[0]),
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias="backbone.blocks.0.attn.qkv.bias" in state_dict,
        drop_path_rate=float(config.drop_path_rate),
        num_expert=num_expert,
        part_features=part_features,
        ratio=1,
    )
    main_head_cfg = _infer_head_cfg(state_dict, "keypoint_head")
    keypoint_head = TopdownHeatmapSimpleHead(**main_head_cfg)
    associate_head_indices = sorted(
        {
            int(key.split(".")[1])
            for key in state_dict.keys()
            if key.startswith("associate_keypoint_heads.") and key.split(".")[1].isdigit()
        }
    )
    associate_heads = [
        TopdownHeatmapSimpleHead(**_infer_head_cfg(state_dict, f"associate_keypoint_heads.{idx}"))
        for idx in associate_head_indices
    ]
    model = ViTPoseTopDownMoE(backbone=backbone, keypoint_head=keypoint_head, associate_keypoint_heads=associate_heads)
    model.load_state_dict(state_dict, strict=True)
    return model


def _get_center_scale(box_xyxy: np.ndarray, aspect_ratio: float, padding: float = 1.25):
    x1, y1, x2, y2 = box_xyxy[:4].astype(np.float32)
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    center = np.array([x1 + width * 0.5, y1 + height * 0.5], dtype=np.float32)
    if width > aspect_ratio * height:
        height = width / aspect_ratio
    elif width < aspect_ratio * height:
        width = height * aspect_ratio
    scale = np.array([width / 200.0, height / 200.0], dtype=np.float32) * padding
    return center, scale


def _get_3rd_point(point_a: np.ndarray, point_b: np.ndarray):
    direction = point_a - point_b
    return point_b + np.array([-direction[1], direction[0]], dtype=np.float32)


def _get_affine_transform(center: np.ndarray, scale: np.ndarray, output_size: Tuple[int, int], inv: bool = False):
    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w, dst_h = output_size
    src_dir = np.array([0.0, -0.5 * src_w], dtype=np.float32)
    dst_dir = np.array([0.0, -0.5 * dst_w], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0] = center
    src[1] = center + src_dir
    src[2] = _get_3rd_point(src[0], src[1])
    dst[0] = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst[1] = dst[0] + dst_dir
    dst[2] = _get_3rd_point(dst[0], dst[1])

    if inv:
        return cv2.getAffineTransform(dst, src)
    return cv2.getAffineTransform(src, dst)


def _affine_transform(point: np.ndarray, transform: np.ndarray):
    return np.dot(transform, np.array([point[0], point[1], 1.0], dtype=np.float32))


def _sigmoid(array: np.ndarray):
    return 1.0 / (1.0 + np.exp(-array))


class ViTPoseRepoWholeBodyInferencer:
    def __init__(self, checkpoint_path: str, config_path: Optional[str] = None, device: str = "cuda:0"):
        if not checkpoint_path:
            raise ValueError("ViTPose checkpoint path is required.")
        state_dict = _extract_state_dict(checkpoint_path)
        overrides = _load_overrides(config_path)
        config = ViTPoseRepoConfig(
            input_size=tuple(overrides.get("input_size", (192, 256))),
            dataset_index=int(overrides.get("dataset_index", 5)),
            num_heads=overrides.get("num_heads"),
            drop_path_rate=float(overrides.get("drop_path_rate", 0.0)),
        )
        self.input_size = tuple(int(value) for value in config.input_size)
        self.dataset_index = config.dataset_index
        torch_device = torch.device(device if not device.startswith("cuda") or torch.cuda.is_available() else "cpu")
        self.device = torch_device
        self.model = _build_model_from_state_dict(state_dict, config)
        self.model.to(self.device)
        self.model.eval()
        self.aspect_ratio = self.input_size[0] / self.input_size[1]
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _prepare_batch(self, image: np.ndarray, bboxes: np.ndarray):
        inputs = []
        centers = []
        scales = []
        transforms = []
        for bbox in bboxes:
            center, scale = _get_center_scale(bbox[:4], aspect_ratio=self.aspect_ratio)
            transform = _get_affine_transform(center, scale, self.input_size)
            warped = cv2.warpAffine(
                image,
                transform,
                self.input_size,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
            warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            warped = (warped - self.mean) / self.std
            warped = warped.transpose(2, 0, 1)
            inputs.append(warped)
            centers.append(center)
            scales.append(scale)
            transforms.append(_get_affine_transform(center, scale, self.input_size, inv=True))
        batch = torch.from_numpy(np.stack(inputs, axis=0)).to(self.device, dtype=torch.float32)
        dataset_source = torch.full((batch.shape[0],), self.dataset_index, device=self.device, dtype=torch.long)
        return batch, np.stack(centers, axis=0), np.stack(scales, axis=0), np.stack(transforms, axis=0), dataset_source

    def _decode_heatmaps(self, heatmaps: np.ndarray, inverse_transforms: np.ndarray):
        batch_size, joint_count, heatmap_h, heatmap_w = heatmaps.shape
        flat = heatmaps.reshape(batch_size, joint_count, -1)
        indices = flat.argmax(axis=2)
        scores = flat.max(axis=2)
        coords = np.zeros((batch_size, joint_count, 2), dtype=np.float32)
        coords[:, :, 0] = indices % heatmap_w
        coords[:, :, 1] = indices // heatmap_w

        for batch_idx in range(batch_size):
            for joint_idx in range(joint_count):
                x_coord = int(coords[batch_idx, joint_idx, 0])
                y_coord = int(coords[batch_idx, joint_idx, 1])
                if 1 < x_coord < heatmap_w - 1 and 1 < y_coord < heatmap_h - 1:
                    diff_x = heatmaps[batch_idx, joint_idx, y_coord, x_coord + 1] - heatmaps[batch_idx, joint_idx, y_coord, x_coord - 1]
                    diff_y = heatmaps[batch_idx, joint_idx, y_coord + 1, x_coord] - heatmaps[batch_idx, joint_idx, y_coord - 1, x_coord]
                    coords[batch_idx, joint_idx, 0] += np.sign(diff_x) * 0.25
                    coords[batch_idx, joint_idx, 1] += np.sign(diff_y) * 0.25

        coords[:, :, 0] = coords[:, :, 0] * (self.input_size[0] / heatmap_w)
        coords[:, :, 1] = coords[:, :, 1] * (self.input_size[1] / heatmap_h)

        preds = np.zeros_like(coords)
        for batch_idx in range(batch_size):
            for joint_idx in range(joint_count):
                preds[batch_idx, joint_idx] = _affine_transform(coords[batch_idx, joint_idx], inverse_transforms[batch_idx])

        return np.concatenate([preds, _sigmoid(scores)[..., None]], axis=2).astype(np.float32)

    def infer(self, image: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
        if bboxes.shape[0] == 0:
            return np.zeros((0, 133, 3), dtype=np.float32)
        batch, _, _, inverse_transforms, dataset_source = self._prepare_batch(image, bboxes)
        with torch.no_grad():
            heatmaps = self.model(batch, dataset_source).detach().cpu().numpy()
        return self._decode_heatmaps(heatmaps, inverse_transforms)