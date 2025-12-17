"""
Assignment 2 – Adversarial Image Attack via Hill Climbing

You MUST implement:
    - compute_fitness
    - mutate_seed
    - select_best
    - hill_climb

DO NOT change function signatures.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from keras.applications import vgg16
from keras.applications.imagenet_utils import decode_predictions
from keras.utils import array_to_img, load_img, img_to_array


# ============================================================
# 1. FITNESS FUNCTION
# ============================================================

def compute_fitness(
    image_array: np.ndarray,
    model,
    target_label: str
) -> float:
    """
    Compute fitness of an image for hill climbing.

    Fitness definition (LOWER is better):
        - If the model predicts target_label:
              fitness = probability(target_label)
        - Otherwise:
              fitness = -probability(predicted_label)
    """

    preds = model.predict(np.expand_dims(image_array, axis=0), verbose=0)
    predictions = []
    for cl in decode_predictions(preds, top=2)[0]:
        label, prob = cl[1], cl[2]
        predictions.append((label, prob))
    assert len(predictions) == 2
    if predictions[0][0] == target_label:
        fitness = predictions[0][1]
    else:
        fitness = -predictions[0][1]
    return fitness

# ============================================================
# 2. MUTATION FUNCTION
# ============================================================

def mutate_seed(
    seed: np.ndarray,
    epsilon: float
) -> List[np.ndarray]:
    """
    Produce ANY NUMBER of mutated neighbors.

    Students may implement ANY mutation strategy:
        - modify 1 pixel
        - modify multiple pixels
        - patch-based mutation
        - channel-based mutation
        - gaussian noise (clipped)
        - etc.

    BUT EVERY neighbor must satisfy the L∞ constraint:

        For all pixels i,j,c:
            |neighbor[i,j,c] - seed[i,j,c]| <= 255 * epsilon

    Requirements:
        ✓ Return a list of neighbors: [neighbor1, neighbor2, ..., neighborK]
        ✓ K can be ANY size ≥ 1
        ✓ Neighbors must be deep copies of seed
        ✓ Pixel values must remain in [0, 255]
        ✓ Must obey the L∞ bound exactly

    Args:
        seed (np.ndarray): input image
        epsilon (float): allowed perturbation budget

    Returns:
        List[np.ndarray]: mutated neighbors
    """

    range_limit = 255 * epsilon
    N_neighbors = 10 
    mutated_neighbors = []
    for i in range(N_neighbors):
        neighbor = seed.copy()
        # Randomly select 1 pixel
        x = np.random.randint(0, seed.shape[0])
        y = np.random.randint(0, seed.shape[1])
        c = np.random.randint(0, seed.shape[2])
        # Randomly perturb the pixel within the allowed range
        perturbation = np.random.uniform(-range_limit, range_limit)
        neighbor[x, y, c] = np.clip(neighbor[x, y, c] + perturbation, 0, 255)
        mutated_neighbors.append(neighbor)
    return mutated_neighbors



# ============================================================
# 3. SELECT BEST CANDIDATE
# ============================================================

def select_best(
    candidates: List[np.ndarray],
    model,
    target_label: str
) -> Tuple[np.ndarray, float]:
    """
    Evaluate fitness for all candidates and return the one with
    the LOWEST fitness score.

    Args:
        candidates (List[np.ndarray])
        model: classifier
        target_label (str)

    Returns:
        (best_image, best_fitness)
    """

    best_fitness = float('inf')
    best_image = None
    for image in candidates:
        fitness_score = compute_fitness(image, model, target_label)
        if fitness_score < best_fitness:
            best_fitness = fitness_score
            best_image = image
    return (best_image, best_fitness)


# ============================================================
# 4. HILL-CLIMBING ALGORITHM
# ============================================================

def hill_climb(
    initial_seed: np.ndarray,
    model,
    target_label: str,
    epsilon: float = 0.30,
    iterations: int = 300
) -> Tuple[np.ndarray, float]:
    """
    Main hill-climbing loop.

    Requirements:
        ✓ Start from initial_seed
        ✓ EACH iteration:
              - Generate ANY number of neighbors using mutate_seed()
              - Enforce the SAME L∞ bound relative to initial_seed
              - Add current image to candidates (elitism)
              - Use select_best() to pick the winner
        ✓ Accept new candidate only if fitness improves
        ✓ Stop if:
              - target class is broken confidently, OR
              - no improvement for multiple steps (optional)

    Returns:
        (final_image, final_fitness)
    """
    current_image = initial_seed.copy()
    current_fitness = compute_fitness(current_image, model, target_label)
    range_limit = 255 * epsilon
    lower_bound = np.clip(initial_seed - range_limit, 0, 255)
    upper_bound = np.clip(initial_seed + range_limit, 0, 255)
    no_improve_steps = 0
    max_no_improve = 20
    stop_reason = "reached max iterations"

    for i in range(iterations):
        candidates = mutate_seed(current_image, epsilon)
        for idx, candidate in enumerate(candidates):
            np.clip(candidate, lower_bound, upper_bound, out=candidate)
            candidates[idx] = candidate

        candidates.append(current_image)
        best_image, best_fitness = select_best(candidates, model, target_label)

        if best_fitness < current_fitness:
            current_image = best_image
            current_fitness = best_fitness
            no_improve_steps = 0
        else:
            no_improve_steps += 1
            if no_improve_steps >= max_no_improve:
                stop_reason = f"no improvement for {max_no_improve} steps"
                break

        preds = model.predict(np.expand_dims(current_image, axis=0), verbose=0)
        top1 = decode_predictions(preds, top=1)[0][0]
        print(f"iter {i + 1:03d}: model_prediction={top1[1]}  prob={top1[2]:.5f}")
        if top1[1] != target_label:
            stop_reason = "prediction is false"
            break

    preds = model.predict(np.expand_dims(current_image, axis=0), verbose=0)
    top1 = decode_predictions(preds, top=1)[0][0]
    print(f"Stop reason: {stop_reason}")
    print(f"Prediction at stop: {top1[1]}  prob={top1[2]:.5f}")
    changed_pixels = int(np.count_nonzero(np.any(initial_seed != current_image, axis=2)))
    print(f"Pixels changed: {changed_pixels}")
    output_dir = "attack_results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "hill_climb_result.png")
    array_to_img(current_image).save(output_path)
    print(f"Saved adversarial image: {output_path}")
    return current_image, current_fitness


# ============================================================
# 5. PROGRAM ENTRY POINT FOR RUNNING A SINGLE ATTACK
# ============================================================

if __name__ == "__main__":
    # Load classifier
    model = vgg16.VGG16(weights="imagenet")

    # Load JSON describing dataset
    with open("data/image_labels.json") as f:
        image_list = json.load(f)

    # Pick first entry
    item = image_list[0]
    image_path = "images/" + item["image"]
    target_label = item["label"]

    print(f"Loaded image: {image_path}")
    print(f"Target label: {target_label}")

    img = load_img(image_path)
    # plt.imshow(img)
    # plt.title("Original image")
    # plt.show()

    img_array = img_to_array(img)
    seed = img_array.copy()

    # Print baseline top-5 predictions
    print("\nBaseline predictions (top-5):")
    preds = model.predict(np.expand_dims(seed, axis=0))
    for cl in decode_predictions(preds, top=5)[0]:
        print(f"{cl[1]:20s}  prob={cl[2]:.5f}")

    # Run hill climbing attack
    final_img, final_fitness = hill_climb(
        initial_seed=seed,
        model=model,
        target_label=target_label,
        epsilon=0.30,
        iterations=300
    )

    print("\nFinal fitness:", final_fitness)

    plt.imshow(array_to_img(final_img))
    plt.title(f"Adversarial Result — fitness={final_fitness:.4f}")
    plt.show()

    # Print final predictions
    final_preds = model.predict(np.expand_dims(final_img, axis=0))
    print("\nFinal predictions:")
    for cl in decode_predictions(final_preds, top=5)[0]:
        print(cl)
