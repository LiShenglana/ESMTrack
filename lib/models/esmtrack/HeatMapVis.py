import os

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_heatmap(image, x, H, W, i, alpha=0.6, title='Influence'):
    image = image[0].squeeze()
    image = image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    x = x[0].squeeze()
    x_map = x[0,:,:]
    x_map = x_map.max(dim=-1).values.reshape(16, 16)

    # diff_map = (x_after_map - x_before_map).abs()

    plt.figure(figsize=(12, 4))
    x_map = x_map.detach().cpu().numpy()
    attn_map = (x_map - x_map.min()) / (x_map.max() - x_map.min() + 1e-6)
    threshold = 0.3
    attn_map[attn_map < threshold] = 0
    attn_map = (255 * attn_map).astype(np.uint8)
    attn_map = cv2.resize(attn_map, (256, 256))
    heatmap = cv2.applyColorMap(attn_map, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)

    current_dir = os.getcwd()
    # save_dir = os.path.join(current_dir, title, str(seq_name))
    save_dir = os.path.join(current_dir, title)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{i}.jpg")
    cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # x_after_map = x_after_map.detach().cpu().numpy()
    # attn_map = (x_after_map - x_after_map.min()) / (x_after_map.max() - x_after_map.min() + 1e-6)
    # threshold = 0.3
    # attn_map[attn_map < threshold] = 0
    # attn_map = (255 * attn_map).astype(np.uint8)
    # attn_map = cv2.resize(attn_map, (256, 256))
    # heatmap = cv2.applyColorMap(attn_map, cv2.COLORMAP_JET)
    # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    # overlay = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
    # cv2.imwrite(f"after.jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    # plt.colorbar()

    # plt.subplot(1, 3, 2)
    # plt.title('After')
    # plt.imshow(x_after_map[i].detach().cpu(), cmap='jet')
    # plt.colorbar()
    #
    # plt.subplot(1, 3, 3)
    # plt.title('Difference')
    # plt.imshow(diff_map[i].detach().cpu(), cmap='hot')
    # plt.colorbar()
    # plt.suptitle(title)
    # plt.show()

def overlay_heatmap_on_image(image, attn_map, alpha=0.6):
    """
    image: numpy array (H, W, 3) RGB
    attn_map: torch.Tensor or numpy array (h, w)
    """
    if isinstance(attn_map, torch.Tensor):
        attn_map = attn_map.detach().cpu().numpy()
    image = image[:, :, :3]

    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-6)
    attn_map = np.uint8(255 * attn_map)

    attn_map = cv2.resize(attn_map, (image.shape[1], image.shape[0]))

    heatmap = cv2.applyColorMap(attn_map, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
    return overlay