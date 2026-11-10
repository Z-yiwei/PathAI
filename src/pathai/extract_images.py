"""Tissue sampling and frozen Phikon encoding / 组织采样与冻结 Phikon 编码。"""

import argparse
import json
from pathlib import Path

import numpy as np
import openslide
import torch
from PIL import Image
from transformers import ViTModel

from .common import write_json
from .data import read_rows


def tissue_mask(image):
    """Estimate tissue pixels from staining color / 根据染色颜色估计组织像素。

    Input / 输入:
        image: PIL image of a slide thumbnail or a candidate region, size (W, H).
        切片缩略图或候选组织窗口的 PIL 图像，尺寸为 (W, H)。
    Processing / 处理:
        Keep pixels with HSV saturation > 25 and mean RGB intensity in (20, 235).
        保留饱和度 > 25、RGB 均值在 (20, 235) 内的像素，过滤近白与过暗背景。
    Output / 输出:
        Boolean NumPy array [H, W]; True marks pixels passing the tissue rule.
        布尔数组 [H, W]，True 表示满足组织颜色规则，不表示肿瘤分类结果。
    """
    rgb = np.asarray(image.convert("RGB"))
    hsv = np.asarray(image.convert("HSV"))
    return (hsv[:, :, 1] > 25) & (rgb.mean(2) > 20) & (rgb.mean(2) < 235)


def region_image(slide, location, level, size, output=224):
    """Read and resize one slide region / 读取并缩放一个切片窗口。

    Inputs / 输入:
        slide: open OpenSlide object / 已打开的 OpenSlide 切片对象。
        location: top-left (x, y) in level-0 pixels / 原图 level 0 的左上角坐标。
        level: pyramid level to read / 读取的金字塔层级。
        size: square window width at that level / 所选层级上的方形窗口边长。
        output: resized width and height, default 224 / 输出边长，默认 224。
    Processing / 处理:
        Read the window, composite transparency onto white, convert to RGB,
        and resize with bilinear interpolation.
        读取窗口，将透明区域补白，转换为 RGB，再用双线性插值缩放。
    Output / 输出:
        RGB PIL image of size (output, output) / 尺寸为 (output, output) 的 RGB 图像。
    """
    rgba = slide.read_region(tuple(location), level, (size, size))
    canvas = Image.new("RGBA", rgba.size, "white")
    canvas.alpha_composite(rgba)
    return canvas.convert("RGB").resize((output, output), Image.Resampling.BILINEAR)


def sample_regions(slide, seed):
    """Select 256 tissue windows / 为一张切片选择 256 个组织窗口。

    Inputs / 输入:
        slide: OpenSlide object with MPP_X, dimensions, and pyramid metadata.
        带原始像素尺度 MPP_X、图像尺寸与金字塔信息的 OpenSlide 对象。
        seed: randomization setting for candidate order / 候选位置打乱的随机参数。
    Processing / 处理:
        Set the field of view to approximately 256 pixels at 0.5 micrometers/pixel.
        Find tissue centers on a thumbnail, shuffle up to 20,000 candidates,
        and map them to level-0 windows. Reject out-of-bounds windows, repeated
        grid cells, and windows with less than 50% tissue on a 256x256 preview.
        按约 0.5 微米/像素下 256 像素的视野确定窗口；在缩略图上找组织中心，
        打乱后最多检查 20,000 个位置，映射回原图；过滤越界、重复网格，
        并在 256x256 预览图上排除组织比例不足 50% 的窗口。
    Output / 输出:
        Dict with level, read_size (pixels at that level), coords_level0
        (256 top-left [x, y] pairs), and target_mpp (0.5).
        字典包含读取层级、该层窗口边长、256 个 level-0 左上角坐标和目标 MPP。
        Raises ValueError if fewer than 256 valid windows are found.
        合格窗口不足 256 个时抛出 ValueError。
    """
    target_mpp = 0.5
    downsample = target_mpp / float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
    level = slide.get_best_level_for_downsample(downsample)
    # 256 here sets the physical field of view (~128 µm), not the number of regions.
    # 此处 256 决定约 128 微米的窗口视野，与最终采样 256 块的数量独立。
    read_size = max(1, round(256 * downsample / slide.level_downsamples[level]))
    footprint = round(read_size * slide.level_downsamples[level])
    thumb = slide.get_thumbnail((1024, 1024)).convert("RGB")
    ys, xs = np.where(tissue_mask(thumb))
    rng = np.random.default_rng(seed)
    coords = []
    seen = set()
    for j in rng.permutation(len(xs))[:20000]:
        x = round(xs[j] / thumb.width * slide.dimensions[0] - footprint / 2)
        y = round(ys[j] / thumb.height * slide.dimensions[1] - footprint / 2)
        if (
            min(x, y) < 0
            or x + footprint > slide.dimensions[0]
            or y + footprint > slide.dimensions[1]
        ):
            continue
        # Grid deduplication reduces repeated sampling but permits window overlap.
        # 网格去重减少重复采样，不保证窗口之间完全不重叠。
        if (x // footprint, y // footprint) in seen:
            continue
        if tissue_mask(region_image(slide, (x, y), level, read_size, 256)).mean() < 0.5:
            continue
        coords.append([x, y])
        seen.add((x // footprint, y // footprint))
        if len(coords) == 256:
            break
    if len(coords) != 256:
        raise ValueError("Insufficient tissue regions")
    return {
        "level": level,
        "read_size": read_size,
        "coords_level0": coords,
        "target_mpp": target_mpp,
    }


def main():
    """Cache frozen Phikon patch features / 提取并缓存冻结 Phikon 的组织块特征。

    Inputs / 输入:
        CLI --out: new output directory, default artifacts/image_features.
        命令行 --out 指定新的输出目录，默认 artifacts/image_features。
        Read TCGA patient IDs and slide paths from artifacts/processed/patients.jsonl,
        local Phikon weights from artifacts/pretrained/phikon, and saved regions
        from configs/regions.json when available. Requires CUDA.
        读取患者表中的 TCGA ID 与切片路径、本地 Phikon 权重和已有采样坐标；需要 CUDA。
    Processing / 处理:
        Reuse saved coordinates or sample missing patients. Read 224x224 RGB patches,
        apply channel normalization, and encode batches of 8 with frozen Phikon.
        Collect one 768-dimensional CLS vector per patch.
        复用坐标或为缺少坐标的患者采样；读取 224x224 RGB 组织块并按通道归一化，
        冻结 Phikon 每批编码 8 块，每块保留一个 768 维 CLS 向量。
    Outputs / 输出:
        Returns None. Write one patient NPZ with features [256, 768] and level-0
        coordinates [256, 2], encoder.json metadata, and the updated regions.json.
        无返回值；保存患者 NPZ（特征 [256,768]、坐标 [256,2]）、编码器信息和采样配置。
        Slide mean pooling and L2 normalization occur in train.py; the trainable
        768-to-64 GPT token projection is in model.py.
        切片级均值汇聚与 L2 归一化在 train.py 中执行，768→64 投影在 model.py 中训练。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/image_features")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    model = (
        ViTModel.from_pretrained(
            "artifacts/pretrained/phikon",
            add_pooling_layer=False,
            local_files_only=True,
        )
        .eval()
        .requires_grad_(False)
        .cuda()
    )
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    region_path = Path("configs/regions.json")
    regions = json.loads(region_path.read_text()) if region_path.exists() else {}
    rows = sorted(
        [
            r
            for r in read_rows("artifacts/processed/patients.jsonl")
            if r["source"] == "tcga"
        ],
        key=lambda r: r["patient_id"],
    )
    for seed, row in enumerate(rows):
        pid = row["patient_id"]
        with openslide.OpenSlide(row["slide_path"]) as slide:
            meta = regions[pid] if pid in regions else sample_regions(slide, seed)
            regions[pid] = meta
            patches = []
            for coord in meta["coords_level0"]:
                rgb = region_image(slide, coord, meta["level"], meta["read_size"])
                patches.append(
                    (
                        torch.from_numpy(np.array(rgb)).permute(2, 0, 1).float() / 255
                        - mean
                    )
                    / std
                )
            # ViT produces [8,197,768]: 196 spatial tokens + CLS; index 0 selects CLS.
            # ViT 输出 [8,197,768]：196 个空间 token 加 CLS；索引 0 取整块图像的表示。
            with torch.inference_mode():
                features = torch.cat(
                    [
                        model(pixel_values=torch.stack(patches[i : i + 8]).cuda())
                        .last_hidden_state[:, 0]
                        .cpu()
                        for i in range(0, len(patches), 8)
                    ]
                )
        np.savez_compressed(
            out / f"{pid}.npz",
            features=features.numpy(),
            coords=np.array(meta["coords_level0"]),
        )
        print(pid, tuple(features.shape), flush=True)
    write_json(region_path, regions)
    write_json(
        out / "encoder.json",
        {"model": "owkin/phikon", "patches": 256, "dimension": 768, "frozen": True},
    )


if __name__ == "__main__":
    main()
