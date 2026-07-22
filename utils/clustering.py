import math
import os
import shutil

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from utils.common import find_original_image, reset_directory
from utils.vector_db import VectorDatabase


def run_dbscan_once(vectors, arcface_margin, eps_tolerance, min_samples):
    """Run DBSCAN once with an ArcFace-derived eps value."""
    base_eps = math.sqrt(2 - 2 * math.cos(arcface_margin))
    calculated_eps = base_eps * eps_tolerance
    clustering = DBSCAN(
        eps=calculated_eps,
        min_samples=min_samples,
        metric="euclidean",
        n_jobs=-1,
    )
    labels = clustering.fit_predict(vectors)
    return labels, calculated_eps


def choose_dbscan_labels(
    vectors,
    eps_tolerance_candidates=None,
    min_samples_candidates=None,
    arcface_margin=0.5,
    fallback_eps_tolerance=1.5,
    fallback_min_samples=2,
):
    if eps_tolerance_candidates is None:
        eps_tolerance_candidates = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    if min_samples_candidates is None:
        min_samples_candidates = [2]

    best_score = -1.0
    best_params = None
    best_labels = None
    best_eps = 0.0

    total_combinations = len(eps_tolerance_candidates) * len(min_samples_candidates)
    with tqdm(total=total_combinations, desc="Searching parameters", unit="combination") as progress:
        for tol in eps_tolerance_candidates:
            for min_s in min_samples_candidates:
                try:
                    labels, actual_eps = run_dbscan_once(
                        vectors,
                        arcface_margin=arcface_margin,
                        eps_tolerance=tol,
                        min_samples=min_s,
                    )
                    unique_labels = set(labels)
                    n_clusters = len(unique_labels - {-1})

                    if n_clusters < 2:
                        continue

                    score = silhouette_score(vectors, labels, metric="cosine")

                    if score > best_score:
                        best_score = score
                        best_params = (tol, min_s)
                        best_labels = labels.copy()
                        best_eps = actual_eps
                except Exception as e:
                    tqdm.write(f"   tol={tol}, min_samples={min_s} failed: {e}")
                finally:
                    progress.update(1)

    if best_params is None:
        print("Could not find suitable parameters. Running once with default parameters.")
        best_labels, best_eps = run_dbscan_once(
            vectors,
            arcface_margin=arcface_margin,
            eps_tolerance=fallback_eps_tolerance,
            min_samples=fallback_min_samples,
        )
        best_params = (fallback_eps_tolerance, fallback_min_samples)

    print(f"Best parameters: tol={best_params[0]}, min_samples={best_params[1]}.")
    print(f"Evaluation metrics: silhouette={best_score:.4f}, eps={best_eps:.4f}.")
    return best_labels, best_params, best_score, best_eps


def resolve_source_image(key, input_img_root, crop_root=None):
    original_path = find_original_image(key, input_img_root)
    if original_path:
        return original_path

    if crop_root:
        crop_path = os.path.join(crop_root, key)
        if os.path.exists(crop_path):
            return crop_path
    return None


def copy_image_to_folder(src_path, folder, output_name=None):
    os.makedirs(folder, exist_ok=True)
    if output_name is None:
        output_name = os.path.basename(src_path)
    dst_path = os.path.join(folder, output_name)
    shutil.copy2(src_path, dst_path)
    return dst_path


def augmented_mask(metadata, count):
    if metadata is None:
        return np.zeros(count, dtype=bool)
    return np.array([
        bool(item.get("is_augmented", False))
        for item in metadata
    ], dtype=bool)


def relabel_clusters_without_enough_originals(labels, is_augmented, min_original_samples):
    labels = np.asarray(labels).copy()
    original_mask = ~np.asarray(is_augmented, dtype=bool)
    for label in sorted(set(labels)):
        label = int(label)
        if label == -1:
            continue
        cluster_mask = labels == label
        original_count = int(np.sum(cluster_mask & original_mask))
        if original_count < min_original_samples:
            labels[cluster_mask] = -1
    return labels


def filter_original_items(keys, vectors, metadata):
    mask = ~augmented_mask(metadata, len(keys))
    indices = np.where(mask)[0]
    return (
        [keys[int(index)] for index in indices],
        vectors[indices],
        [metadata[int(index)] for index in indices],
        np.zeros(len(indices), dtype=bool),
    )


def source_key_for_item(key, metadata):
    if metadata is None:
        return key
    return metadata.get("source_key") or key


def build_augmented_bundles(keys, vectors, metadata):
    """Build one normalized clustering vector per original image source."""
    vectors = np.asarray(vectors, dtype=np.float32)
    groups = {}
    for index, key in enumerate(keys):
        item_metadata = metadata[index] if metadata is not None else None
        source_key = source_key_for_item(key, item_metadata)
        group = groups.setdefault(source_key, {
            "member_indices": [],
            "original_indices": [],
        })
        group["member_indices"].append(index)
        if not bool((item_metadata or {}).get("is_augmented", False)):
            group["original_indices"].append(index)

    bundle_keys = []
    bundle_vectors = []
    bundle_original_indices = []
    bundle_member_indices = []
    skipped_augmented_only = 0

    for source_key, group in groups.items():
        if not group["original_indices"]:
            skipped_augmented_only += 1
            continue

        member_indices = np.asarray(group["member_indices"], dtype=np.int64)
        bundle_vector = vectors[member_indices].mean(axis=0)
        bundle_vector = bundle_vector / (np.linalg.norm(bundle_vector) + 1e-8)
        original_index = int(group["original_indices"][0])

        bundle_keys.append(keys[original_index] if keys[original_index] else source_key)
        bundle_vectors.append(bundle_vector)
        bundle_original_indices.append(original_index)
        bundle_member_indices.append([int(index) for index in member_indices])

    if skipped_augmented_only:
        print(f"Skipped {skipped_augmented_only} augmented-only bundles without original vectors.")

    if bundle_vectors:
        bundle_vectors = np.asarray(bundle_vectors, dtype=np.float32)
    else:
        bundle_vectors = np.empty((0, vectors.shape[1] if vectors.ndim == 2 else 0), dtype=np.float32)

    return bundle_keys, bundle_vectors, bundle_original_indices, bundle_member_indices


def expand_bundle_labels(bundle_labels, item_count, bundle_member_indices):
    labels = np.full(item_count, -1, dtype=np.asarray(bundle_labels).dtype)
    for label, member_indices in zip(bundle_labels, bundle_member_indices):
        labels[np.asarray(member_indices, dtype=np.int64)] = label
    return labels


def find_cluster_centroids(keys, vectors, labels, is_augmented=None):
    if is_augmented is None:
        is_augmented = np.zeros(len(keys), dtype=bool)
    is_augmented = np.asarray(is_augmented, dtype=bool)
    centroid_indices = {}
    for label in sorted(set(labels)):
        if label == -1:
            continue
        cluster_indices = np.where((labels == label) & ~is_augmented)[0]
        if len(cluster_indices) == 0:
            continue
        cluster_vectors = vectors[cluster_indices]
        centroid = cluster_vectors.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        similarities = np.dot(cluster_vectors, centroid)
        best_local_index = int(np.argmax(similarities))
        centroid_indices[int(label)] = int(cluster_indices[best_local_index])
    return centroid_indices


def find_bundle_cluster_centroids(bundle_vectors, bundle_labels, bundle_original_indices):
    centroid_indices = {}
    bundle_labels = np.asarray(bundle_labels)
    bundle_original_indices = np.asarray(bundle_original_indices, dtype=np.int64)
    for label in sorted(set(bundle_labels)):
        if label == -1:
            continue
        cluster_indices = np.where(bundle_labels == label)[0]
        if len(cluster_indices) == 0:
            continue
        cluster_vectors = bundle_vectors[cluster_indices]
        centroid = cluster_vectors.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        similarities = np.dot(cluster_vectors, centroid)
        best_local_index = int(np.argmax(similarities))
        best_bundle_index = int(cluster_indices[best_local_index])
        centroid_indices[int(label)] = int(bundle_original_indices[best_bundle_index])
    return centroid_indices


def copy_cluster_outputs(keys, labels, input_img_root, output_root, crop_root=None, is_augmented=None):
    os.makedirs(output_root, exist_ok=True)
    copy_count = 0
    noise_count = 0

    if is_augmented is None:
        is_augmented = np.zeros(len(keys), dtype=bool)
    is_augmented = np.asarray(is_augmented, dtype=bool)

    for i in tqdm(range(len(keys)), desc="Organizing files", unit="image"):
        if is_augmented[i]:
            continue

        label = int(labels[i])
        if label == -1:
            cluster_folder = os.path.join(output_root, "special_noise")
            noise_count += 1
        else:
            cluster_folder = os.path.join(output_root, f"class_{label}")

        src_path = resolve_source_image(keys[i], input_img_root, crop_root=crop_root)
        if src_path:
            copy_image_to_folder(src_path, cluster_folder)
            copy_count += 1
        else:
            print(f"Source image not found for feature key: {keys[i]}")

    return copy_count, noise_count


def resolve_crop_image(key, crop_root):
    if not crop_root:
        return None
    crop_path = os.path.join(crop_root, key)
    if os.path.exists(crop_path):
        return crop_path
    return None


def copy_centroid_outputs(centroid_indices, keys, centroid_root, crop_root=None):
    os.makedirs(centroid_root, exist_ok=True)
    centroid_count = 0
    for label, index in centroid_indices.items():
        src_path = resolve_crop_image(keys[index], crop_root)
        if not src_path:
            print(f"Centroid crop image not found for feature key: {keys[index]}")
            continue
        extension = os.path.splitext(src_path)[1]
        copy_image_to_folder(src_path, centroid_root, output_name=f"class_{label}{extension}")
        centroid_count += 1
    return centroid_count


def cluster_feature_db(
    input_folder="buffer",
    db_path=None,
    db_name="features.fvdb",
    output_root=os.path.join("test_sets", "output"),
    input_img_root=os.path.join("test_sets", "input"),
    eps_tolerance_candidates=None,
    min_samples_candidates=None,
    clear_output=True,
    use_augmentation=True,
):
    """Cluster extracted features and organize original images by class."""
    if db_path is None:
        db = VectorDatabase.load_auto(input_folder, preferred_name=db_name)
    else:
        db = VectorDatabase.load(db_path)

    if len(db.keys) == 0:
        print("Feature database is empty.")
        return None

    keys = db.keys
    vectors = db.normalized_vectors()
    metadata = db.metadata
    is_augmented = db.is_augmented_mask()

    if use_augmentation:
        bundle_keys, bundle_vectors, bundle_original_indices, bundle_member_indices = build_augmented_bundles(
            keys,
            vectors,
            metadata,
        )
        if len(bundle_keys) == 0:
            print("Feature database has no original vectors to cluster.")
            return None

        bundle_labels, best_params, best_score, best_eps = choose_dbscan_labels(
            bundle_vectors,
            eps_tolerance_candidates=eps_tolerance_candidates,
            min_samples_candidates=min_samples_candidates,
        )
        labels = expand_bundle_labels(bundle_labels, len(keys), bundle_member_indices)
        centroid_indices = find_bundle_cluster_centroids(
            bundle_vectors,
            bundle_labels,
            bundle_original_indices,
        )
    else:
        keys, vectors, metadata, is_augmented = filter_original_items(keys, vectors, metadata)
        if len(keys) == 0:
            print("Feature database has no original vectors to cluster.")
            return None

        labels, best_params, best_score, best_eps = choose_dbscan_labels(
            vectors,
            eps_tolerance_candidates=eps_tolerance_candidates,
            min_samples_candidates=min_samples_candidates,
        )
        centroid_indices = find_cluster_centroids(keys, vectors, labels, is_augmented=is_augmented)

    if clear_output:
        reset_directory(output_root)
    else:
        os.makedirs(output_root, exist_ok=True)
    print(f"Organizing files into {output_root}.")
    copy_count, noise_count = copy_cluster_outputs(
        keys,
        labels,
        input_img_root,
        output_root,
        crop_root=input_folder,
        is_augmented=is_augmented,
    )

    centroid_root = os.path.join(output_root, "centroids")
    centroid_count = copy_centroid_outputs(
        centroid_indices,
        keys,
        centroid_root,
        crop_root=input_folder,
    )

    class_count = len(set(labels)) - (1 if -1 in labels else 0)
    print("Stage 3 complete.")
    print(f"   Valid classes found: {class_count}")
    print(f"   Noise points found: {noise_count}")
    print(f"   Images copied: {copy_count}")
    print(f"   Centroid images copied: {centroid_count}")
    print(f"   Centroid images saved to: {centroid_root}")

    return {
        "labels": labels,
        "best_params": best_params,
        "best_score": best_score,
        "best_eps": best_eps,
        "class_count": class_count,
        "noise_count": noise_count,
        "copy_count": copy_count,
        "centroid_count": centroid_count,
        "centroid_root": centroid_root,
    }
