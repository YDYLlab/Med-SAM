#!/usr/bin/env python3
"""
Mask edge classifier using CLIP.

Usage example:
  python scripts/mask_edge_clip_classifier.py --mask mask.npy --image image.png

Supports mask as .npy or binary image (png). If `--image` provided, edge is rendered over crop; otherwise edge-only image is used.
"""
from pathlib import Path
import argparse
import numpy as np
from PIL import Image
import torch

try:
    from transformers import CLIPProcessor, CLIPModel
except Exception as e:
    raise RuntimeError("transformers and torch are required. See scripts/requirements-mask-clip.txt") from e


def load_mask(path):
    p = Path(path)
    if p.suffix == '.npy':
        m = np.load(str(p))
    else:
        im = Image.open(str(p)).convert('L')
        m = np.array(im)
        m = (m > 127).astype(np.uint8)
    if m.ndim == 3:
        m = m[..., 0]
    return (m > 0).astype(np.uint8)


def mask_to_edge(mask):
    mask = mask.astype(np.uint8)
    up = np.roll(mask, -1, axis=0)
    down = np.roll(mask, 1, axis=0)
    left = np.roll(mask, -1, axis=1)
    right = np.roll(mask, 1, axis=1)
    up[-1, :] = 0
    down[0, :] = 0
    left[:, -1] = 0
    right[:, 0] = 0
    edge = ((mask != up) | (mask != down) | (mask != left) | (mask != right)) & (mask == 1)
    return edge.astype(np.uint8)


def render_edge_image(edge, image=None, target_size=(224, 224), pad=8):
    # edge: 2D uint8 array
    h, w = edge.shape
    ys, xs = np.where(edge)
    if len(xs) == 0:
        # empty edge: return blank image
        img = Image.new('RGB', target_size, (0, 0, 0))
        return img

    minx, maxx = max(xs.min()-pad, 0), min(xs.max()+pad, w-1)
    miny, maxy = max(ys.min()-pad, 0), min(ys.max()+pad, h-1)

    crop = edge[miny:maxy+1, minx:maxx+1]
    if image is not None:
        img = Image.open(str(image)).convert('RGB')
        img = img.crop((minx, miny, maxx+1, maxy+1))
        # overlay white edge
        overlay = Image.new('RGB', img.size, (0, 0, 0))
        overlay_arr = np.array(overlay)
        edge_resized = Image.fromarray((crop * 255).astype(np.uint8))
        edge_resized = edge_resized.resize(img.size, Image.NEAREST)
        edge_mask = np.array(edge_resized) > 0
        overlay_arr[edge_mask] = [255, 255, 255]
        overlay = Image.fromarray(overlay_arr)
        blended = Image.blend(img, overlay, alpha=0.9)
        out = blended.resize(target_size, Image.BICUBIC)
        return out
    else:
        edge_img = Image.fromarray((crop * 255).astype(np.uint8)).convert('L')
        edge_rgb = Image.new('RGB', edge_img.size, (0, 0, 0))
        edge_arr = np.array(edge_img) > 0
        edge_rgb_arr = np.array(edge_rgb)
        edge_rgb_arr[edge_arr] = [255, 255, 255]
        edge_rgb = Image.fromarray(edge_rgb_arr)
        out = edge_rgb.resize(target_size, Image.BICUBIC)
        return out


def load_texts_from_file(file_path):
    p = Path(file_path)
    if not p.exists():
        return [
            'type_I: single nodule with distinct margin, usually round with complete tumour envelope',
            'type_II: single nodule with extranodular growth, no more than three extranodular points',
            'type_III: a unifocal lesion composed of confluent multiple nodules, distinct boundaries among the nodules',
            'type_IV: infiltrative nodule, with poor demarcated boundary and especially multiple extranodular points',
        ]
    txt = p.read_text(encoding='utf-8')
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    return lines


def classify_edge_with_clip(image_pil, texts, model_name='openai/clip-vit-base-patch32', device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    inputs = processor(text=texts, images=image_pil, return_tensors='pt', padding=True)
    # move tensors to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        image_features = model.get_image_features(inputs['pixel_values'])
        text_features = model.get_text_features(inputs['input_ids'], attention_mask=inputs['attention_mask'])

    # normalize
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    sims = (image_features @ text_features.T).cpu().numpy()[0]
    # return list of (text, score)
    return list(zip(texts, sims))
