import argparse
import os
import shutil

import numpy as np

from utils.clustering import build_augmented_bundles, choose_dbscan_labels
from utils.common import find_original_image, float_range, int_range, list_image_files, reset_directory
from utils.detection import crop_largest_furry_targets
from utils.embedding import extract_features_from_folder
from utils.vector_db import VectorDatabase, require_feature_db


DEFAULT_INPUT_FOLDER = os.path.join("input", "images")
DEFAULT_BUFFER_FOLDER = "buffer"
DEFAULT_TARGET_FOLDER = os.path.join("input", "id_targets")
DEFAULT_TARGET_BUFFER = os.path.join("buffer", "identify")
DEFAULT_OUTPUT_FOLDER = os.path.join("output", "identify")
DEFAULT_DB_NAME = "features.fvdb"


def parse_batch_size(value):
    if value is None or value == "auto":
        return value
    return int(value)


def build_clustering_candidates(args):
    eps_candidates = float_range(args.eps_start, args.eps_stop, args.eps_step)
    min_samples_candidates = int_range(args.min_samples_start, args.min_samples_stop, args.min_samples_step)
    return eps_candidates, min_samples_candidates


def normalize_vectors(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    return vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)


def crop_identity_targets(args):
    return crop_largest_furry_targets(
        input_folder=args.id_target_folder,
        output_folder=args.target_buffer_folder,
        model_path=args.model_path,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        clear_output=True,
    )


def build_target_source_lookup(args):
    target_sources = {}
    for img_name in list_image_files(args.id_target_folder):
        stem = os.path.splitext(img_name)[0]
        target_sources[stem] = os.path.join(args.id_target_folder, img_name)
    return target_sources


def target_stem_from_crop_key(crop_key):
    base_name = os.path.splitext(os.path.basename(crop_key))[0]
    return base_name.rsplit("_target", 1)[0]


def folder_name_from_reference(src_path):
    return os.path.splitext(os.path.basename(src_path))[0]


def unique_folder_name(used_names, folder_name):
    unique_name = folder_name
    suffix = 2
    while unique_name in used_names:
        unique_name = f"{folder_name}_{suffix}"
        suffix += 1
    used_names.add(unique_name)
    return unique_name


def build_target_label_folders(target_items, target_labels_all, target_sources):
    label_folders = {}
    used_names = set()
    for item, label in zip(target_items, target_labels_all):
        label = int(label)
        if label == -1 or label in label_folders:
            continue

        target_stem = target_stem_from_crop_key(item["key"])
        src_path = target_sources.get(target_stem)
        if not src_path:
            print(f"Reference image not found for target feature key: {item['key']}")
            continue

        folder_name = unique_folder_name(used_names, folder_name_from_reference(src_path))
        label_folders[label] = {
            "folder_name": folder_name,
            "source_path": src_path,
        }
    return label_folders


def extract_identity_target_features(args):
    feature_items = extract_features_from_folder(
        input_folder=args.target_buffer_folder,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_multi_gpu=args.use_multi_gpu,
    )
    if not feature_items:
        raise ValueError(f"No target features were extracted from {args.target_buffer_folder}.")
    return feature_items


def copy_identified_db_images(db_keys, labels, label_folders, input_folder, output_folder):
    reset_directory(output_folder)
    copied_sources = set()
    copied_count = 0

    for key, label in zip(db_keys, labels):
        label = int(label)
        if label not in label_folders:
            continue

        src_path = find_original_image(key, input_folder)
        if not src_path:
            print(f"Original image not found for feature key: {key}")
            continue

        copy_key = (label, os.path.abspath(src_path))
        if copy_key in copied_sources:
            continue

        class_folder = os.path.join(output_folder, label_folders[label]["folder_name"])
        os.makedirs(class_folder, exist_ok=True)
        dst_path = os.path.join(class_folder, os.path.basename(src_path))
        shutil.copy2(src_path, dst_path)
        copied_sources.add(copy_key)
        copied_count += 1

    return copied_count


def no_identity_result(args, target_count, best_params, best_score, best_eps):
    reset_directory(args.output_folder)
    print("No non-noise identity classes were found. Output folder was cleared.")
    return {
        "target_count": target_count,
        "identified_classes": 0,
        "copy_count": 0,
        "best_params": best_params,
        "best_score": best_score,
        "best_eps": best_eps,
    }


def identify(args):
    db_path = require_feature_db(args.buffer_folder, args.db_name)
    print(f"Using existing feature database: {db_path}")

    crop_paths = crop_identity_targets(args)
    print(f"Identity target crops produced: {len(crop_paths)}")
    target_sources = build_target_source_lookup(args)

    target_items = extract_identity_target_features(args)
    db = VectorDatabase.load_auto(args.buffer_folder, preferred_name=args.db_name)
    if len(db.keys) == 0:
        raise ValueError("Feature database is empty.")

    all_db_vectors = db.normalized_vectors()
    if args.use_augmentation:
        db_keys_for_clustering, db_vectors, _, _ = build_augmented_bundles(
            db.keys,
            all_db_vectors,
            db.metadata,
        )
        db_is_augmented = np.zeros(len(db_keys_for_clustering), dtype=bool)
    else:
        original_indices = np.where(~db.is_augmented_mask())[0]
        db_keys_for_clustering = [db.keys[int(index)] for index in original_indices]
        db_vectors = all_db_vectors[original_indices]
        db_is_augmented = np.zeros(len(db_keys_for_clustering), dtype=bool)

    if len(db_keys_for_clustering) == 0:
        raise ValueError("Feature database has no original vectors to identify.")

    target_vectors = normalize_vectors([item["vector"] for item in target_items])
    combined_vectors = np.vstack([db_vectors, target_vectors, target_vectors])

    eps_candidates, min_samples_candidates = build_clustering_candidates(args)
    labels, best_params, best_score, best_eps = choose_dbscan_labels(
        combined_vectors,
        eps_tolerance_candidates=eps_candidates,
        min_samples_candidates=min_samples_candidates,
    )

    db_count = len(db_keys_for_clustering)
    db_labels = labels[:db_count]
    target_labels_all = labels[db_count:]
    output_indices = np.where(~db_is_augmented)[0]
    output_db_keys = [db_keys_for_clustering[int(index)] for index in output_indices]
    output_db_labels = db_labels[output_indices]
    label_folders = build_target_label_folders(target_items, target_labels_all, target_sources)

    if not label_folders:
        return no_identity_result(args, len(target_items), best_params, best_score, best_eps)

    copied_count = copy_identified_db_images(
        output_db_keys,
        output_db_labels,
        label_folders,
        args.input_folder,
        args.output_folder,
    )

    if copied_count == 0:
        return no_identity_result(args, len(target_items), best_params, best_score, best_eps)

    print("Identity search complete.")
    print(f"   Targets processed: {len(target_items)}")
    print(f"   Identified classes: {len(label_folders)}")
    print(f"   DB images copied: {copied_count}")
    print(f"   Results saved to: {args.output_folder}")

    return {
        "target_count": len(target_items),
        "identified_classes": len(label_folders),
        "copy_count": copied_count,
        "best_params": best_params,
        "best_score": best_score,
        "best_eps": best_eps,
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Identify classes containing target animals using an existing DB.")
    parser.add_argument("--input-folder", default=DEFAULT_INPUT_FOLDER)
    parser.add_argument("--buffer-folder", default=DEFAULT_BUFFER_FOLDER)
    parser.add_argument("--db-name", default=DEFAULT_DB_NAME)
    parser.add_argument("--id-target-folder", default=DEFAULT_TARGET_FOLDER)
    parser.add_argument("--target-buffer-folder", default=DEFAULT_TARGET_BUFFER)
    parser.add_argument("--output-folder", default=DEFAULT_OUTPUT_FOLDER)

    parser.add_argument("--model-path", default=None)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=1280)

    parser.add_argument("--batch-size", type=parse_batch_size, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--no-multi-gpu", dest="use_multi_gpu", action="store_false")
    parser.add_argument("--no-augmentation", dest="use_augmentation", action="store_false")
    parser.set_defaults(use_multi_gpu=True, use_augmentation=True)

    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-stop", type=float, default=2.0)
    parser.add_argument("--eps-step", type=float, default=0.01)
    parser.add_argument("--min-samples-start", type=int, default=3)
    parser.add_argument("--min-samples-stop", type=int, default=3)
    parser.add_argument("--min-samples-step", type=int, default=1)
    return parser


def main():
    args = build_parser().parse_args()
    identify(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Program failed: {e}")
        raise
