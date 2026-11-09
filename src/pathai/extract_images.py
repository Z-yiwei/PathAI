"""Read tissue regions from H&E SVS and cache frozen Phikon patch features."""

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
    rgb = np.asarray(image.convert("RGB"))
    hsv = np.asarray(image.convert("HSV"))
    return (hsv[:, :, 1] > 25) & (rgb.mean(2) > 20) & (rgb.mean(2) < 235)


def region_image(slide, location, level, size, output=224):
    rgba = slide.read_region(tuple(location), level, (size, size))
    canvas = Image.new("RGBA", rgba.size, "white")
    canvas.alpha_composite(rgba)
    return canvas.convert("RGB").resize((output, output), Image.Resampling.BILINEAR)


def sample_regions(slide, seed):
    target_mpp = 0.5
    downsample = target_mpp / float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
    level = slide.get_best_level_for_downsample(downsample)
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
