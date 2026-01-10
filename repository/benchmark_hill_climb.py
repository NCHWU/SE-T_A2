import os
import csv
import time
import json
import numpy as np
from keras.applications import vgg16
from keras.applications.imagenet_utils import decode_predictions
from keras.utils import load_img, img_to_array

from hill_climbing import hill_climb

OUTDIR = "attack_results"
CSV_PATH = os.path.join(OUTDIR, "hill_climb_runtimes.csv")
JSON_FILE = "data/image_labels.json"
IMAGE_DIR = "images/"
EPSILONS = [0.15, 0.30, 0.40, 0.50]
ITERATIONS = 3000  # adjust for speed if needed

os.makedirs(OUTDIR, exist_ok=True)

# Load classifier once
model = vgg16.VGG16(weights="imagenet")

# Load dataset list
with open(JSON_FILE, "r") as f:
    image_list = json.load(f)

headers = [
    "Image",
    *[f"HC_{eps:.2f}" for eps in EPSILONS],
    "CleanCorrect",
    "HC_FinalCorrect",
]
rows = []

for item in image_list:
    image_path = os.path.join(IMAGE_DIR, item["image"]) 
    target_label = item["label"]

    img = load_img(image_path)
    img_array = img_to_array(img)
    seed = img_array.copy()

    # Clean correctness (top-1)
    preds = model.predict(np.expand_dims(seed, axis=0), verbose=0)
    clean_top1 = decode_predictions(preds, top=1)[0][0][1]
    clean_correct = (clean_top1 == target_label)

    # Measure hill climb runtimes for each epsilon
    hc_times = []
    for eps in EPSILONS:
        t0 = time.perf_counter()
        final_img, _ = hill_climb(
            initial_seed=seed,
            model=model,
            target_label=target_label,
            epsilon=eps,
            iterations=ITERATIONS,
        )
        hc_times.append(time.perf_counter() - t0)

    # Final correctness after strongest epsilon
    max_eps = max(EPSILONS)
    final_img, _ = hill_climb(
        initial_seed=seed,
        model=model,
        target_label=target_label,
        epsilon=max_eps,
        iterations=ITERATIONS,
    )
    final_preds = model.predict(np.expand_dims(final_img, axis=0), verbose=0)
    final_top1 = decode_predictions(final_preds, top=1)[0][0][1]
    hc_final_correct = (final_top1 == target_label)

    rows.append({
        "Image": os.path.splitext(item["image"])[0],
        **{f"HC_{eps:.2f}": t for eps, t in zip(EPSILONS, hc_times)},
        "CleanCorrect": clean_correct,
        "HC_FinalCorrect": hc_final_correct,
    })

    print(f"Image: {item['image']}")
    print("HC times:", ", ".join(f"{t:.4f}s" for t in hc_times))
    print(f"Clean correct: {clean_correct} | HC end correct: {hc_final_correct}")
    print("------------------------------------------------------")

# Write CSV
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

print(f"Saved hill climb runtimes to {CSV_PATH}")
