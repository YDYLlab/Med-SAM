# Mask edge CLIP classifier

Usage:

```
python scripts/mask_edge_clip_classifier.py --mask path/to/mask.npy [--image path/to/image.png]
```

Notes:
- `--mask` accepts a `.npy` binary mask or a binary image (png) where foreground is non-zero.
- `--image` is optional; if provided the edge will be rendered over the original crop.
- The script uses the text descriptions from `GrossTypes.txt` by default.
- Install dependencies with `pip install -r scripts/requirements-mask-clip.txt`.
