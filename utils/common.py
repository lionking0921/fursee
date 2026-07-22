import glob
import os
import shutil

import torch


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def project_path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)


def get_device():
    """Detect the inference device and whether FP16 should be used."""
    if torch.cuda.is_available():
        device = "cuda:0"
        half = True
        print("Detected GPU. Using CUDA.") 
    else:
        device = "cpu"
        half = False
        print("No GPU detected. Using CPU inference.") 
    return device, half


def get_model_assets_path():
    local_model_path = project_path("fursee_models")
    if os.path.exists(local_model_path):
        return local_model_path
    return "./fursee_models"


def get_cut_model_path():
    return os.path.join(get_model_assets_path(), "cut.pt")


def reset_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def ensure_directory(path):
    os.makedirs(path, exist_ok=True)
    return path


def list_image_files(folder, extensions=IMAGE_EXTENSIONS):
    return sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith(extensions)
    )


def float_range(start, stop, step):
    if step <= 0:
        raise ValueError("step must be greater than 0")
    if stop < start:
        raise ValueError("stop must be greater than or equal to start")

    values = []
    current = float(start)
    stop = float(stop)
    epsilon = abs(step) / 1_000_000
    while current <= stop + epsilon:
        values.append(round(current, 10))
        current += step
    return values


def int_range(start, stop, step):
    if step <= 0:
        raise ValueError("step must be greater than 0")
    if stop < start:
        raise ValueError("stop must be greater than or equal to start")
    return list(range(start, stop + 1, step))


def original_name_from_crop(crop_name):
    base_name = os.path.splitext(os.path.basename(crop_name))[0]
    return base_name.rsplit("_", 2)[0]


def find_original_image(crop_name, input_root):
    original_name = original_name_from_crop(crop_name)
    matched_files = glob.glob(os.path.join(input_root, f"{original_name}.*"))
    if not matched_files:
        return None
    return matched_files[0]
