import os
from matplotlib.image import imsave

# set-up
out_dir = './masks'
out_npy = os.path.join(out_dir, f'npy_{image_name}')
out_png = os.path.join(out_dir, f'png_{image_name}')
os.makedirs(out_npy, exist_ok=True)
os.makedirs(out_png, exist_ok=True)

# Save all masks and generate overlay image
for i, mask in enumerate(masks):
    seg = mask['segmentation'].astype(bool)
    # save the original bool mask
    np.save(os.path.join(out_npy, f'mask_{i:03d}.npy'), seg)

    # Generate an overlay visualization (red overlay).
    overlay = image.copy().astype(np.uint8)
    if seg.ndim != 2:
        seg = np.squeeze(seg)
    color = np.array([255, 0, 0], dtype=np.uint8)
    overlay[seg] = (overlay[seg] * 0.5 + color * 0.5).astype(np.uint8)

    # Save png
    imsave(os.path.join(out_png, f'mask_{i:03d}.png'), overlay)

print(f"Saved {len(masks)} masks to {out_dir}/ （npy + png）")
