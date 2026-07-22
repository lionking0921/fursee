import argparse
import os

from tqdm import tqdm

from utils.clustering import cluster_feature_db
from utils.common import float_range, int_range
from utils.vector_db import require_feature_db


DEFAULT_INPUT_FOLDER = os.path.join("input", "images")
DEFAULT_BUFFER_FOLDER = "buffer"
DEFAULT_OUTPUT_FOLDER = os.path.join("output", "classify")
DEFAULT_DB_NAME = "features.fvdb"


def build_clustering_candidates(args):
    eps_candidates = float_range(args.eps_start, args.eps_stop, args.eps_step)
    min_samples_candidates = int_range(args.min_samples_start, args.min_samples_stop, args.min_samples_step)
    return eps_candidates, min_samples_candidates


def stage3(
    input_folder=None,
    db_name=DEFAULT_DB_NAME,
    output_root=DEFAULT_OUTPUT_FOLDER,
    input_img_root=DEFAULT_INPUT_FOLDER,
    eps_tolerance_candidates=None,
    min_samples_candidates=None,
    use_augmentation=True,
):
    if input_folder is None:
        input_folder = DEFAULT_BUFFER_FOLDER
    return cluster_feature_db(
        input_folder=input_folder,
        db_name=db_name,
        output_root=output_root,
        input_img_root=input_img_root,
        eps_tolerance_candidates=eps_tolerance_candidates,
        min_samples_candidates=min_samples_candidates,
        clear_output=True,
        use_augmentation=use_augmentation,
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Cluster images using an existing FurSee feature DB.")
    parser.add_argument("--input-folder", default=DEFAULT_INPUT_FOLDER)
    parser.add_argument("--buffer-folder", default=DEFAULT_BUFFER_FOLDER)
    parser.add_argument("--db-name", default=DEFAULT_DB_NAME)
    parser.add_argument("--output-folder", default=DEFAULT_OUTPUT_FOLDER)

    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-stop", type=float, default=2.0)
    parser.add_argument("--eps-step", type=float, default=0.01)
    parser.add_argument("--min-samples-start", type=int, default=2)
    parser.add_argument("--min-samples-stop", type=int, default=2)
    parser.add_argument("--min-samples-step", type=int, default=1)
    parser.add_argument("--no-augmentation", dest="use_augmentation", action="store_false")
    parser.set_defaults(use_augmentation=True)
    return parser


def main():
    args = build_parser().parse_args()
    db_path = require_feature_db(args.buffer_folder, args.db_name)
    print(f"Using existing feature database: {db_path}")

    eps_candidates, min_samples_candidates = build_clustering_candidates(args)
    tqdm.write("Clustering images")
    stage3(
        input_folder=args.buffer_folder,
        db_name=args.db_name,
        output_root=args.output_folder,
        input_img_root=args.input_folder,
        eps_tolerance_candidates=eps_candidates,
        min_samples_candidates=min_samples_candidates,
        use_augmentation=args.use_augmentation,
    )


if __name__ == "__main__":
    main()
