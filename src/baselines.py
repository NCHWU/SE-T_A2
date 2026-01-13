import json
import os
import csv
import time
import numpy as np
import torch
from torchvision.utils import save_image
from torchvision.models import vgg16
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms
from PIL import Image

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent

# -----------------------------
# Utility: parse Imagenet prediction
# -----------------------------
def parse_prediction(output, categories):
    probs = torch.nn.functional.softmax(output[0], dim=0)
    top_prob, top_catid = torch.topk(probs, 1)
    return categories[top_catid], top_prob.item()


# ================================================================
# 1. Load JSON file with images + expected human label
# ================================================================
JSON_FILE = "data/image_labels.json"
IMAGE_DIR = "images/"

with open(JSON_FILE, "r") as f:
    items = json.load(f)

# ================================================================
# 2. Load ImageNet labels
# ================================================================
with open("data/imagenet_classes.txt", "r") as f:
    imagenet_labels = [s.strip() for s in f.readlines()]

label_to_index = {label: i for i, label in enumerate(imagenet_labels)}

# ================================================================
# 3. Model
# ================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
net = vgg16(weights="DEFAULT").to(device)
net.eval()

# ================================================================
# 4. Image preprocessing transform
# ================================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# ================================================================
# 5. Attack hyperparameters
# ================================================================
EPSILONS = [0.15, 0.30, 0.40, 0.50]
PGD_STEPS = 40
PGD_STEP_SIZE = 0.01

# ================================================================
# 6. Output directory
# ================================================================
OUTDIR = "attack_results"
os.makedirs(OUTDIR, exist_ok=True)

# Prepare CSV output containers
rows = []
headers = [
    "Image",
    *[f"FGM_{eps:.2f}" for eps in EPSILONS],
    *[f"PGD_{eps:.2f}" for eps in EPSILONS],
    "CleanCorrect",
    "FGM_FinalCorrect",
    "PGD_FinalCorrect",
]

# ================================================================
# 7. Run attacks for every image from the JSON file
# ================================================================
for entry in tqdm(items, desc="Running attacks"):
    image_file = entry["image"]
    human_label = entry["label"]  # e.g. "goldfish"

    # -----------------------------
    # Load + preprocess image
    # -----------------------------
    img_path = os.path.join(IMAGE_DIR, image_file)
    img_pil = Image.open(img_path).convert("RGB")
    x = transform(img_pil).unsqueeze(0).to(device)

    # -----------------------------
    # Ground truth index
    # -----------------------------
    if human_label in label_to_index:
        true_idx = label_to_index[human_label]
    else:
        true_idx = None
        print(f"⚠️ Warning: '{human_label}' not found in ImageNet labels.")

    # -----------------------------
    # Predict clean image
    # -----------------------------
    out_clean = net(x)
    pred_clean, prob_clean = parse_prediction(out_clean, imagenet_labels)

    # Save clean image
    save_image(x, os.path.join(OUTDIR, f"{image_file}_clean.png"))

    print(f"\nImage: {image_file}")
    print(f"Human label: {human_label}")
    print(f"Model prediction (clean): {pred_clean} ({prob_clean:.3f})")

    # =====================================================
    # FGM runtimes across epsilons
    # =====================================================
    fgm_times = []
    for eps in EPSILONS:
        t0 = time.perf_counter()
        _ = fast_gradient_method(net, x, eps, np.inf)
        fgm_times.append(time.perf_counter() - t0)

    # =====================================================
    # PGD runtimes across epsilons
    # =====================================================
    pgd_times = []
    for eps in EPSILONS:
        t0 = time.perf_counter()
        _ = projected_gradient_descent(net, x, eps, PGD_STEP_SIZE, PGD_STEPS, np.inf)
        pgd_times.append(time.perf_counter() - t0)

    # Correctness flags
    clean_correct = (pred_clean == human_label)
    max_eps = max(EPSILONS)
    x_fgm_end = fast_gradient_method(net, x, max_eps, np.inf)
    pred_fgm_end, _ = parse_prediction(net(x_fgm_end), imagenet_labels)
    fgm_final_correct = (pred_fgm_end == human_label)

    x_pgd_end = projected_gradient_descent(net, x, max_eps, PGD_STEP_SIZE, PGD_STEPS, np.inf)
    pred_pgd_end, _ = parse_prediction(net(x_pgd_end), imagenet_labels)
    pgd_final_correct = (pred_pgd_end == human_label)

    # Add row to CSV data
    rows.append({
        "Image": os.path.splitext(image_file)[0],
        **{f"FGM_{eps:.2f}": t for eps, t in zip(EPSILONS, fgm_times)},
        **{f"PGD_{eps:.2f}": t for eps, t in zip(EPSILONS, pgd_times)},
        "CleanCorrect": clean_correct,
        "FGM_FinalCorrect": fgm_final_correct,
        "PGD_FinalCorrect": pgd_final_correct,
    })

    print("FGM times:", ", ".join(f"{t:.4f}s" for t in fgm_times))
    print("PGD times:", ", ".join(f"{t:.4f}s" for t in pgd_times))
    print(f"Clean correct: {clean_correct} | FGM end correct: {fgm_final_correct} | PGD end correct: {pgd_final_correct}")
    print("------------------------------------------------------")

# Write CSV once after processing all images
csv_path = os.path.join(OUTDIR, "fgm_pgd_runtimes.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

print(f"Saved runtimes to {csv_path}")