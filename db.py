import argparse
import os
import shutil
import tempfile

from tqdm import tqdm

from utils.augmentation import DEFAULT_AUGMENTATION_COUNT
from utils.common import IMAGE_EXTENSIONS, ensure_directory, list_image_files, original_name_from_crop
from utils.detection import crop_furry_detections
from utils.embedding import append_features_to_db, extract_features_to_db
from utils.vector_db import VectorDatabase, require_feature_db


DEFAULT_INPUT_FOLDER = os.path.join("input", "images")
DEFAULT_BUFFER_FOLDER = "buffer"
DEFAULT_DB_NAME = "features.fvdb"


def parse_batch_size(value):
    if value is None or value == "auto":
        return value
    return int(value)


def parse_augmentation_count(value):
    count = int(value)
    if count < 1:
        raise argparse.ArgumentTypeError("augmentation-count must be at least 1")
    return count


def validate_image_folder(folder, role):
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"{role} folder not found: {folder}")

    image_files = list_image_files(folder)
    if not image_files:
        raise ValueError(f"No supported image files found in {role} folder: {folder}")
    return image_files


def same_folder(left, right):
    return os.path.realpath(left) == os.path.realpath(right)


def is_path_inside(path, folder):
    path = os.path.realpath(path)
    folder = os.path.realpath(folder)
    try:
        return os.path.commonpath([path, folder]) == folder
    except ValueError:
        return False


def list_image_paths_recursive(folder, exclude_folder=None):
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"input folder not found: {folder}")

    excluded_root = os.path.realpath(exclude_folder) if exclude_folder else None
    image_paths = []
    for root, dirs, files in os.walk(folder):
        if excluded_root and is_path_inside(root, excluded_root):
            dirs[:] = []
            continue

        if excluded_root:
            dirs[:] = [
                dirname for dirname in dirs
                if not is_path_inside(os.path.join(root, dirname), excluded_root)
            ]

        for filename in files:
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                image_paths.append(os.path.join(root, filename))

    image_paths.sort(key=lambda path: os.path.relpath(path, folder))
    return image_paths


def copy_images_to_input_folder(source_folder, input_folder):
    image_paths = list_image_paths_recursive(source_folder, exclude_folder=input_folder)
    if not image_paths:
        raise ValueError(f"No supported image files found in input folder: {source_folder}")

    ensure_directory(input_folder)
    reserved_names = set()
    copied_count = 0
    for source_path in image_paths:
        final_name = unique_destination_name(input_folder, os.path.basename(source_path), reserved_names)
        destination_path = os.path.join(input_folder, final_name)
        if os.path.realpath(source_path) == os.path.realpath(destination_path):
            continue
        shutil.copy2(source_path, destination_path)
        copied_count += 1
    return copied_count


def prepare_cold_input_folder(input_folder):
    if same_folder(input_folder, DEFAULT_INPUT_FOLDER):
        validate_image_folder(input_folder, "input")
        return input_folder

    copied_count = copy_images_to_input_folder(input_folder, DEFAULT_INPUT_FOLDER)
    tqdm.write(f"Copied {copied_count} input images into {DEFAULT_INPUT_FOLDER}.")
    validate_image_folder(DEFAULT_INPUT_FOLDER, "input")
    return DEFAULT_INPUT_FOLDER


def validate_append_folder(append_folder, input_folder):
    if append_folder is None:
        return validate_image_folder(input_folder, "input")

    if same_folder(append_folder, input_folder):
        return validate_image_folder(input_folder, "input")

    return validate_image_folder(append_folder, "append")


def existing_original_names_from_db(buffer_folder, db_name):
    db = VectorDatabase.load_auto(buffer_folder, preferred_name=db_name)
    names = set()
    for key, metadata in zip(db.keys, db.metadata):
        if metadata.get("is_augmented", False):
            continue
        source_key = metadata.get("source_key", key)
        names.add(original_name_from_crop(source_key))
    return names


def filter_new_input_images(input_folder, image_files, buffer_folder, db_name):
    existing_original_names = existing_original_names_from_db(buffer_folder, db_name)
    return [
        image_name for image_name in image_files
        if os.path.splitext(image_name)[0] not in existing_original_names
    ]


def unique_destination_name(destination_folder, filename, reserved_names):
    stem, ext = os.path.splitext(filename)
    candidate = filename
    suffix = 2
    while candidate in reserved_names or os.path.exists(os.path.join(destination_folder, candidate)):
        candidate = f"{stem}_{suffix}{ext}"
        suffix += 1
    reserved_names.add(candidate)
    return candidate


def build_append_copy_plan(append_folder, input_folder, image_files, copy_to_input=True):
    reserved_names = set()
    copy_plan = []
    for image_name in image_files:
        final_name = image_name
        if copy_to_input:
            final_name = unique_destination_name(input_folder, image_name, reserved_names)
        copy_plan.append({
            "source_path": os.path.join(append_folder, image_name),
            "final_name": final_name,
        })
    return copy_plan


def create_append_staging_folder(copy_plan):
    staging_folder = tempfile.mkdtemp(prefix="fursee_append_")
    for item in copy_plan:
        shutil.copy2(item["source_path"], os.path.join(staging_folder, item["final_name"]))
    return staging_folder


def copy_staged_images_to_input(staging_folder, input_folder, copy_plan):
    ensure_directory(input_folder)
    copied_count = 0
    for item in copy_plan:
        final_name = item["final_name"]
        shutil.copy2(os.path.join(staging_folder, final_name), os.path.join(input_folder, final_name))
        copied_count += 1
    return copied_count


def cold_build(args):
    input_folder = prepare_cold_input_folder(args.input_folder)

    tqdm.write("Stage 1: detecting and cropping images")
    crop_furry_detections(
        input_folder=input_folder,
        output_folder=args.buffer_folder,
        model_path=args.model_path,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        clear_output=True,
    )

    tqdm.write("Stage 2: extracting features")
    db_path = extract_features_to_db(
        input_folder=args.buffer_folder,
        db_name=args.db_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_multi_gpu=args.use_multi_gpu,
        augmentation_count=args.augmentation_count,
    )
    print(f"Feature database ready: {db_path}")
    return db_path


def append_build(args):
    require_feature_db(args.buffer_folder, args.db_name)
    append_folder = args.append_folder or args.input_folder
    append_from_input = same_folder(append_folder, args.input_folder)

    image_files = validate_append_folder(args.append_folder, args.input_folder)
    if append_from_input:
        image_files = filter_new_input_images(args.input_folder, image_files, args.buffer_folder, args.db_name)
        if not image_files:
            db_path = require_feature_db(args.buffer_folder, args.db_name)
            print(f"No new append images found in {args.input_folder}.")
            print(f"Feature database ready: {db_path}")
            return db_path
        print(f"Found {len(image_files)} new append images in {args.input_folder}.")

    copy_plan = build_append_copy_plan(
        append_folder,
        args.input_folder,
        image_files,
        copy_to_input=not append_from_input,
    )
    staging_folder = create_append_staging_folder(copy_plan)
    try:
        tqdm.write("Stage 1: detecting and cropping append images")
        crop_furry_detections(
            input_folder=staging_folder,
            output_folder=args.buffer_folder,
            model_path=args.model_path,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            clear_output=False,
        )

        tqdm.write("Stage 2: appending features")
        db_path = append_features_to_db(
            input_folder=args.buffer_folder,
            db_name=args.db_name,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_multi_gpu=args.use_multi_gpu,
            augmentation_count=args.augmentation_count,
        )

        if append_from_input:
            print(f"Processed {len(image_files)} new images from {args.input_folder}.")
        else:
            copied_count = copy_staged_images_to_input(staging_folder, args.input_folder, copy_plan)
            print(f"Copied {copied_count} append images into {args.input_folder}.")
        print(f"Feature database ready: {db_path}")
        return db_path
    finally:
        shutil.rmtree(staging_folder, ignore_errors=True)


def add_common_arguments(parser):
    parser.add_argument("--input-folder", default=DEFAULT_INPUT_FOLDER)
    parser.add_argument("--buffer-folder", default=DEFAULT_BUFFER_FOLDER)
    parser.add_argument("--db-name", default=DEFAULT_DB_NAME)

    parser.add_argument("--model-path", default=None)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=1280)

    parser.add_argument("--batch-size", type=parse_batch_size, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--augmentation-count", type=parse_augmentation_count, default=DEFAULT_AUGMENTATION_COUNT)
    parser.add_argument("--no-multi-gpu", dest="use_multi_gpu", action="store_false")
    parser.set_defaults(use_multi_gpu=True)


def build_parser():
    parser = argparse.ArgumentParser(description="Build or append the FurSee feature database.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    cold_parser = subparsers.add_parser("cold", help="Rebuild the feature database from input/images.")
    add_common_arguments(cold_parser)
    cold_parser.set_defaults(func=cold_build)

    append_parser = subparsers.add_parser("append", help="Append new photos from a folder or from input/images.")
    add_common_arguments(append_parser)
    append_parser.add_argument("--append-folder", default=None)
    append_parser.set_defaults(func=append_build)

    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Program failed: {e}")
        raise
