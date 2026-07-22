import argparse
import os
import shutil

import numpy as np
import torch
from PIL import Image

from utils.common import find_original_image, get_device, get_model_assets_path, reset_directory
from utils.detection import crop_largest_furry_targets
from utils.embedding import load_stage2_model
from utils.vector_db import VectorDatabase, require_feature_db


DEFAULT_INPUT_FOLDER = os.path.join("input", "images")
DEFAULT_BUFFER_FOLDER = "buffer"
DEFAULT_TARGET_FOLDER = os.path.join("input", "sim_targets")
DEFAULT_TARGET_BUFFER = os.path.join("buffer", "similar")
DEFAULT_OUTPUT_FOLDER = os.path.join("output", "similar")
DEFAULT_DB_NAME = "features.fvdb"


def stage1_topk(
    ref_folder=DEFAULT_TARGET_FOLDER,
    output_folder=DEFAULT_TARGET_BUFFER,
    model_path=None,
    conf=0.5,
    iou=0.45,
    imgsz=1280,
):
    return crop_largest_furry_targets(
        input_folder=ref_folder,
        output_folder=output_folder,
        model_path=model_path,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        clear_output=True,
    )


def stage2_extract_feature(img_path, model, processor, device):
    print(f"Extracting reference image feature: {img_path}")

    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.inference_mode():
        embeddings = model(inputs.pixel_values)
        vec = embeddings[0].cpu().numpy()
        norm = np.linalg.norm(vec)
        if norm != 0:
            vec = vec / norm

    return vec


def extract_reference_items(crop_paths):
    print("Loading the custom embedding model for reference feature extraction.")
    local_model_path = get_model_assets_path()
    device, _ = get_device()
    torch_device = torch.device(device)
    model, processor = load_stage2_model(local_model_path, torch_device)
    return [
        {"crop_path": path, "vector": stage2_extract_feature(path, model, processor, torch_device)}
        for path in crop_paths
    ]


def similarity_percentage(similarity_score):
    return max(0.0, min(100.0, (float(similarity_score) + 1.0) * 50.0))


def query_stem_from_crop_path(crop_path):
    base_name = os.path.splitext(os.path.basename(crop_path))[0]
    return base_name.rsplit("_target", 1)[0]


def unique_folder_name(used_names, folder_name):
    unique_name = folder_name
    suffix = 2
    while unique_name in used_names:
        unique_name = f"{folder_name}_{suffix}"
        suffix += 1
    used_names.add(unique_name)
    return unique_name


def copy_topk_for_reference(db, ref_item, query_folder, input_root, k):
    if k <= 0:
        return 0

    include_mask = ~db.is_augmented_mask()
    candidate_count = int(np.count_nonzero(include_mask))
    top_k_indices, similarities = db.top_k(ref_item["vector"], candidate_count, include_mask=include_mask)
    copied_sources = set()
    copied_count = 0

    for idx, similarity_score in zip(top_k_indices, similarities):
        idx = int(idx)
        metadata = db.metadata[idx]
        key = metadata.get("source_key", db.keys[idx])
        src_path = find_original_image(key, input_root)
        if not src_path:
            print(f"Original image not found for feature key: {key} (score={similarity_score:.4f})")
            continue

        abs_src = os.path.abspath(src_path)
        if abs_src in copied_sources:
            continue

        _, ext = os.path.splitext(src_path)
        percent = similarity_percentage(similarity_score)
        rank = copied_count + 1
        dst_name = f"{rank:03d}_{percent:.2f}{ext.lower()}"
        dst_path = os.path.join(query_folder, dst_name)
        shutil.copy2(src_path, dst_path)
        copied_sources.add(abs_src)
        copied_count += 1
        if copied_count >= k:
            break

    if copied_count < k:
        print(f"Only copied {copied_count} unique matched images; requested {k}.")

    return copied_count


def stage3_topk(
    reference_items,
    buffer_folder=DEFAULT_BUFFER_FOLDER,
    db_name=DEFAULT_DB_NAME,
    input_root=DEFAULT_INPUT_FOLDER,
    output_folder=DEFAULT_OUTPUT_FOLDER,
    k=60,
):
    print(f"Starting stage 3: running per-query Top-{k} search.")

    db = VectorDatabase.load_auto(buffer_folder, preferred_name=db_name)
    reset_directory(output_folder)
    print(f"Organizing Top-{k} results into {output_folder}.")

    used_names = set()
    total_copied = 0
    for ref_item in reference_items:
        folder_name = unique_folder_name(used_names, query_stem_from_crop_path(ref_item["crop_path"]))
        query_folder = os.path.join(output_folder, folder_name)
        os.makedirs(query_folder, exist_ok=True)
        copied_count = copy_topk_for_reference(
            db,
            ref_item,
            query_folder,
            input_root,
            k,
        )
        total_copied += copied_count
        print(f"   {folder_name}: copied {copied_count} matched images")

    print("Top-K search complete.")
    print(f"   Queries processed: {len(reference_items)}")
    print(f"   Matched images copied: {total_copied}")
    print(f"   Results saved to: {output_folder}")


def build_parser():
    parser = argparse.ArgumentParser(description="Find top-k images similar to one or more target images using an existing DB.")
    parser.add_argument("--input-folder", default=DEFAULT_INPUT_FOLDER)
    parser.add_argument("--buffer-folder", default=DEFAULT_BUFFER_FOLDER)
    parser.add_argument("--db-name", default=DEFAULT_DB_NAME)
    parser.add_argument("--sim-target-folder", default=DEFAULT_TARGET_FOLDER)
    parser.add_argument("--target-buffer-folder", default=DEFAULT_TARGET_BUFFER)
    parser.add_argument("--output-folder", default=DEFAULT_OUTPUT_FOLDER)
    parser.add_argument("--k", type=int, default=1)

    parser.add_argument("--model-path", default=None)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=1280)
    return parser


def main():
    args = build_parser().parse_args()
    db_path = require_feature_db(args.buffer_folder, args.db_name)
    print(f"Using existing feature database: {db_path}")

    crop_paths = stage1_topk(
        ref_folder=args.sim_target_folder,
        output_folder=args.target_buffer_folder,
        model_path=args.model_path,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
    )
    reference_items = extract_reference_items(crop_paths)
    stage3_topk(
        reference_items=reference_items,
        buffer_folder=args.buffer_folder,
        db_name=args.db_name,
        input_root=args.input_folder,
        output_folder=args.output_folder,
        k=args.k,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Program failed: {e}")
        raise
