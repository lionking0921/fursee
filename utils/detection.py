import os

import cv2
from tqdm import tqdm
from ultralytics import YOLO

from utils.common import get_cut_model_path, get_device, list_image_files, reset_directory


def crop_furry_detections(
    input_folder="test_sets/input",
    output_folder="buffer",
    model_path=None,
    conf=0.5,
    iou=0.45,
    imgsz=1280,
    clear_output=True,
):
    """Detect furry targets with YOLO and save cropped regions."""
    if model_path is None:
        model_path = get_cut_model_path()

    model = YOLO(model_path)
    device, half = get_device()
    if clear_output:
        reset_directory(output_folder)
    else:
        os.makedirs(output_folder, exist_ok=True)

    image_paths = [
        os.path.join(input_folder, f)
        for f in list_image_files(input_folder)
    ]

    for img_path in tqdm(image_paths, desc="Processing images", unit="image"):
        results = model.predict(
            source=img_path,
            device=device,
            quantize=16 if half else None,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            verbose=False,
        )

        result = results[0]
        boxes = result.boxes
        if len(boxes) == 0:
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"Skipping unreadable image: {img_path}")
            continue
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            cls_name = result.names[cls_id]
            if cls_name != "furry":
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            h, w, _ = img.shape
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            face_crop = img[y1:y2, x1:x2]
            if face_crop.size == 0:
                print(f"Skipping empty crop from image: {img_path}")
                continue

            conf_score = float(box.conf[0])
            save_name = f"{img_name}_{i}_{conf_score:.2f}.jpg"
            save_path = os.path.join(output_folder, save_name)
            if not clear_output and os.path.exists(save_path):
                print(f"Skipping existing crop in append mode: {save_path}")
                continue
            cv2.imwrite(save_path, face_crop)

    print("Stage 1 complete.")
    return output_folder


def _find_largest_furry_box(model, image_path, device, half, conf, iou, imgsz):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to read image: {image_path}")

    h, w, _ = img.shape
    results = model.predict(
        source=image_path,
        device=device,
        quantize=16 if half else None,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        verbose=False,
    )

    result = results[0]
    boxes = result.boxes
    if len(boxes) == 0:
        raise ValueError(f"No target was detected in image: {image_path}")

    max_area = 0
    best_box = None
    best_conf = 0.0
    for box in boxes:
        cls_id = int(box.cls[0])
        cls_name = result.names[cls_id]
        if cls_name != "furry":
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        area = (x2 - x1) * (y2 - y1)
        if area > max_area:
            max_area = area
            best_box = (x1, y1, x2, y2)
            best_conf = float(box.conf[0])

    if best_box is None:
        raise ValueError(f"No furry target was detected in image: {image_path}")

    return img, best_box, best_conf


def _save_largest_furry_crop(model, image_path, output_path, device, half, conf, iou, imgsz):
    img, best_box, _ = _find_largest_furry_box(model, image_path, device, half, conf, iou, imgsz)
    x1, y1, x2, y2 = best_box
    face_crop = img[y1:y2, x1:x2]
    if face_crop.size == 0:
        raise ValueError(f"Largest target crop is empty: {image_path}")
    cv2.imwrite(output_path, face_crop)
    return output_path


def crop_largest_furry_target(
    ref_image_path="topk_test/ref.png",
    output_folder="topk_test",
    output_name="temp.png",
    model_path=None,
    conf=0.5,
    iou=0.45,
    imgsz=1280,
):
    """Detect and crop the largest furry target in a reference image."""
    print("Starting stage 1: detecting and cropping the largest target.")
    if model_path is None:
        model_path = get_cut_model_path()

    model = YOLO(model_path)
    device, half = get_device()
    os.makedirs(output_folder, exist_ok=True)

    if not os.path.exists(ref_image_path):
        raise FileNotFoundError(f"Reference image not found: {ref_image_path}")

    temp_path = os.path.join(output_folder, output_name)
    _save_largest_furry_crop(model, ref_image_path, temp_path, device, half, conf, iou, imgsz)

    print(f"Largest target crop saved to {temp_path}.")
    return temp_path


def crop_largest_furry_targets(
    input_folder,
    output_folder,
    model_path=None,
    conf=0.5,
    iou=0.45,
    imgsz=1280,
    clear_output=True,
):
    """Crop the largest furry target from each image in a folder."""
    print(f"Starting stage 1: detecting largest targets in {input_folder}.")
    if model_path is None:
        model_path = get_cut_model_path()

    if clear_output:
        reset_directory(output_folder)
    else:
        os.makedirs(output_folder, exist_ok=True)

    image_files = list_image_files(input_folder)
    if not image_files:
        raise ValueError(f"No supported image files found in {input_folder}.")

    model = YOLO(model_path)
    device, half = get_device()
    crop_paths = []

    for index, img_name in enumerate(tqdm(image_files, desc="Processing target images", unit="image")):
        img_path = os.path.join(input_folder, img_name)
        stem = os.path.splitext(img_name)[0]
        output_name = f"{stem}_target{index}_1.00.jpg"
        output_path = os.path.join(output_folder, output_name)
        try:
            _save_largest_furry_crop(model, img_path, output_path, device, half, conf, iou, imgsz)
            crop_paths.append(output_path)
        except ValueError as e:
            print(f"Skipping target image {img_path}: {e}")

    if not crop_paths:
        raise ValueError(f"No furry target crops were produced from {input_folder}.")

    print(f"Largest target crops saved to {output_folder}.")
    return crop_paths
