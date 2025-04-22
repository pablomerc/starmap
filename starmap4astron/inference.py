# inference.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from trainers.lc2img_module import LC2ImgModule

def main(checkpoint, npz_path, out_image="reconstruction.png"):
    # 1) load LightningModule & model
    module = LC2ImgModule.load_from_checkpoint(checkpoint)
    module.eval()
    model = module.model.to(module.device)

    # 2) load one light-curve
    data = np.load(npz_path, allow_pickle=True)
    flux = data["flux"][0]
    lc   = torch.tensor(flux, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(module.device)

    # 3) forward pass
    with torch.no_grad():
        pred = model(lc)             # shape (1,1,H,W)

    # 4) extract & denormalize
    img = pred[0,0].cpu()           # → (H, W)
    img = (img * 0.5) + 0.5
    img = img.clamp(0,1).numpy()

    # 5) save with gray colormap
    plt.imsave(out_image, img, cmap="gray", origin="lower")
    print(f"Saved {out_image}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("npz_path")
    p.add_argument("--out", default="reconstruction.png")
    args = p.parse_args()
    main(args.checkpoint, args.npz_path, args.out)


#python inference.py "$CKPT" "$NPZ" --out recon.png
