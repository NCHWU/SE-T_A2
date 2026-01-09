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

    # TODO (student)
    preds = model.predict(np.expand_dims(image_array, axis=0), verbose=0)
    predictions = decode_predictions(preds, top=2)[0]
    assert len(predictions) == 2

    top1_label, top1_prob = predictions[0][1], predictions[0][2]
    top2_prob = predictions[1][2]
    if top1_label == target_label:
        fitness = top1_prob - top2_prob
    else:
        fitness = -top1_prob
    return fitness

# ============================================================
# 2. MUTATION FUNCTION
# ============================================================

def mutate_seed(
    seed: np.ndarray,
    epsilon: float,
    step_scale : float
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

    import cv2
    delta = 255.0 * epsilon
    lower = np.clip(seed - delta, 0, 255)
    upper = np.clip(seed + delta, 0, 255)

    step = delta * step_scale

    K = 5
    h, w, c = seed.shape
    perturbation_probability = 0.75

    # Edge Detection -> gaussian "halo mask"
    gray = cv2.cvtColor(seed.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 40, 120) # 40-120 for smoother edges 

    # gaussian smoothing to create a "circle/halo" effect around edges
    # different halos for different textures "fine" vs "coarse" 
    halo1 = cv2.GaussianBlur(edges, (0, 0), sigmaX=2.0, sigmaY=2.0)
    halo2 = cv2.GaussianBlur(edges, (0, 0), sigmaX=5.0, sigmaY=5.0)
    edge_halo = np.maximum(halo1, halo2)

    # Normalize the mask to [0, 1] after gaussian blurring
    edge_halo = edge_halo / (edge_halo.max() + 1e-8)

    mutated_neighbors = []
    for _ in range(K):
        neighbor = seed.copy()

        # Threshold controls how wide/strong the halo extends to
        # Gaussian outputs 1.0 closer to the edges, and it "dissipates" further away
        # The mask essentially eliminates the gaussian's further away from the edges, which is how it controls
        # how strong the mask is
        # ex: 0.75 means more concentrated perturbations near edges
        # ex: 0.75 means we spread the halo our further
        halo_threshold = np.random.uniform(0.01, 0.25)
        halo_mask_2d = edge_halo > halo_threshold
        # Expand the channels from 2D -> 3D so we can mask the channels (h,w,c)
        halo_mask = np.repeat(halo_mask_2d[:, :, None], c, axis=2)

        # creates a low-frequnecy perturbation grid
        # upsamples it to "smooth" out the details
        # make changes to those upsampled regions 
        size_k = np.random.choice([7, 14, 28]) # filter sizes
        low_frequency_convolution = np.random.randn(size_k, size_k, c)
        upsampling = cv2.resize(low_frequency_convolution, (w, h), interpolation=cv2.INTER_CUBIC)

        # (h, w, c) for per-channel based perturbations
        # randomize image mask [0, 1] aka we only make changes to certain parts of the halo
        randomization_halo_mask = np.random.rand(h, w, c) < perturbation_probability
        mask = halo_mask & randomization_halo_mask
        
        # add delta noise in different local directions of upsampled smooth regions
        # dictated by the allow-able mask
        lowfreq_noise = step * np.sign(upsampling)

        # basically, how would it change when we add or remove the noise from the pixel?
        n1 = np.clip(neighbor + mask * lowfreq_noise, lower, upper)
        n2 = np.clip(neighbor - mask * lowfreq_noise, lower, upper)
        mutated_neighbors.append(n1)
        mutated_neighbors.append(n2)

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

    BROKEN_CONFIDENTLY_THRESHOLD = 0.9
    EARLY_STOPPING_CRITERIA = 9999 # Criteria for when no change after n steps
    iterations_without_improvement = 0

    # Enforce the SAME L∞ bound relative to initial_seed
    range_limit = 255.0 * epsilon
    lower = np.clip(initial_seed - range_limit, 0, 255)
    upper = np.clip(initial_seed + range_limit, 0, 255)
    current_seed = initial_seed.copy()
    current_fitness = compute_fitness(current_seed, model, target_label)
    for i in range(1, iterations):

        # We set a constraint to how "small" of changes we want to make 
        # at the beginning and it increases exponentially over iterations
        # this way we have more emphasis on smaller pixel changes
        scale_min = 0.1 # minimum allowable of delta perturbations
        scale_max = 1.0 # maximum allowable of delta perturbations 
        k = 50          # growth rate
        frac = i / max(1, iterations - 1) # 0 -> 1
        growth = (1 - np.exp(-k * frac))
        step_scale = scale_min + (scale_max - scale_min) * growth

        # Generate ANY number of neighbors using mutate_seed()
        neighbors = mutate_seed(current_seed, epsilon, step_scale)
        neighbors = [np.clip(n, lower, upper) for n in neighbors]
        
        # Add current image to candidates (elitism)
        neighbors.append(current_seed)
        best_iteration_neighbor, best_iteration_fitness = select_best(neighbors, model, target_label)

        # Condition on if adverserial fitness improves
        if best_iteration_fitness < current_fitness: 
            current_fitness = best_iteration_fitness
            iterations_without_improvement = 0

            # Accept new candidate only if fitness improves
            current_seed = best_iteration_neighbor
            # print(f"Iteration {i} Improvement: {current_fitness}")
        elif best_iteration_fitness == current_fitness:
            # Incremeent early-stopping count
            iterations_without_improvement += 1

        # Stop if target class is broken confidently, OR no improvement for multiple steps (optional)
        if (EARLY_STOPPING_CRITERIA == iterations_without_improvement or
            current_fitness < -BROKEN_CONFIDENTLY_THRESHOLD):
            break
        print(f"Iteration {i} w/ step: {step_scale}: {current_fitness}")
        
    # Returns the "best model" (final_image, final_fitness)
    return (current_seed, current_fitness)



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
    # hard: 2 (cup), 5 (bubble), 6 (goblet), 9 (viaduct)
    item = image_list[6]
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
        epsilon=0.15,
        iterations=3000
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