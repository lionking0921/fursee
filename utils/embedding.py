import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import torch.multiprocessing as mp
from PIL import Image
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from utils.augmentation import DEFAULT_AUGMENTATION_COUNT, augment_image
from utils.common import IMAGE_EXTENSIONS, get_device, get_model_assets_path
from utils.fursee_models import FurseeModel
from utils.vector_db import VectorDatabase, require_feature_db


STAGE2_IMAGE_EXTENSIONS = IMAGE_EXTENSIONS


class Stage2ImageDataset(Dataset):
    def __init__(self, input_folder, indexed_image_files, include_augmentations=False, augmentation_count=DEFAULT_AUGMENTATION_COUNT):
        self.input_folder = input_folder
        self.samples = self._build_samples(indexed_image_files, include_augmentations, augmentation_count)

    def _build_samples(self, indexed_image_files, include_augmentations, augmentation_count):
        samples = []
        per_image_count = 1 + augmentation_count if include_augmentations else 1
        for original_index, img_name in indexed_image_files:
            base_index = original_index * per_image_count
            samples.append({
                "index": base_index,
                "image_name": img_name,
                "source_name": img_name,
                "metadata": {
                    "is_augmented": False,
                    "source_key": img_name,
                    "augment_index": None,
                },
            })
            if include_augmentations:
                for augment_offset in range(augmentation_count):
                    augment_index = augment_offset + 1
                    samples.append({
                        "index": base_index + augment_index,
                        "image_name": f"{img_name}::aug{augment_index}",
                        "source_name": img_name,
                        "metadata": {
                            "is_augmented": True,
                            "source_key": img_name,
                            "augment_index": augment_index,
                        },
                    })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        img_path = os.path.join(self.input_folder, sample["source_name"])
        try:
            with Image.open(img_path) as image:
                image = image.convert("RGB")
                if sample["metadata"].get("is_augmented", False):
                    image = augment_image(image, sample["metadata"]["augment_index"] - 1)
            return {
                "index": sample["index"],
                "image_name": sample["image_name"],
                "metadata": sample["metadata"],
                "image": image,
            }
        except Exception as e:
            return {
                "index": sample["index"],
                "image_name": sample["image_name"],
                "metadata": sample["metadata"],
                "image": None,
                "error": str(e),
            }


class Stage2Collator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, samples):
        valid_samples = []
        for sample in samples:
            if sample.get("image") is None:
                print(f"Skipping unreadable image {sample['image_name']}: {sample.get('error', 'unknown error')}")
                continue
            valid_samples.append(sample)

        if not valid_samples:
            return None

        inputs = self.processor(images=[sample["image"] for sample in valid_samples], return_tensors="pt")
        return {
            "pixel_values": inputs["pixel_values"],
            "indices": [sample["index"] for sample in valid_samples],
            "image_names": [sample["image_name"] for sample in valid_samples],
            "metadata": [sample["metadata"] for sample in valid_samples],
        }


def load_stage2_model(local_model_path, torch_device):

    import logging
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.ERROR)

    processor = AutoImageProcessor.from_pretrained(local_model_path)
    dummy_backbone = AutoModel.from_pretrained(local_model_path, trust_remote_code=True)
    model = FurseeModel(
        backbone=dummy_backbone,
        input_dim=1024,
        embedding_dim=512,
        dropout=0.1,
    )

    weights_path = os.path.join(local_model_path, "model.safetensors")
    if os.path.exists(weights_path):
        state_dict = load_file(weights_path, device=str(torch_device))
    else:
        state_dict = torch.load(os.path.join(local_model_path, "pytorch_model.bin"), map_location=torch_device)

    model.load_state_dict(state_dict, strict=False)
    model.to(torch_device)
    model.eval()
    return model, processor


def choose_stage2_batch_size(torch_device):
    if torch_device.type != "cuda":
        return 1

    props = torch.cuda.get_device_properties(torch_device)
    total_gb = props.total_memory / (1024 ** 3)
    if total_gb < 6:
        return 2
    if total_gb < 10:
        return 4
    if total_gb < 16:
        return 8
    if total_gb < 24:
        return 16
    return 32


def choose_stage2_num_workers(torch_device):
    cpu_count = os.cpu_count() or 1
    if torch_device.type == "cuda":
        return min(4, cpu_count)
    return min(2, cpu_count)


def build_stage2_loader(
    input_folder,
    indexed_image_files,
    processor,
    batch_size,
    num_workers,
    torch_device,
    include_augmentations=False,
    augmentation_count=DEFAULT_AUGMENTATION_COUNT,
):
    dataset = Stage2ImageDataset(
        input_folder,
        indexed_image_files,
        include_augmentations=include_augmentations,
        augmentation_count=augmentation_count,
    )
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "collate_fn": Stage2Collator(processor),
        "pin_memory": torch_device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **loader_kwargs)


def extract_stage2_features_batched(
    input_folder,
    indexed_image_files,
    model,
    processor,
    torch_device,
    batch_size,
    num_workers,
    include_augmentations=False,
    augmentation_count=DEFAULT_AUGMENTATION_COUNT,
):
    loader = build_stage2_loader(
        input_folder,
        indexed_image_files,
        processor,
        batch_size,
        num_workers,
        torch_device,
        include_augmentations=include_augmentations,
        augmentation_count=augmentation_count,
    )
    feature_items = []
    for batch in tqdm(loader, desc=f"Extracting features ({torch_device})", unit="batch"):
        if batch is None:
            continue

        pixel_values = batch["pixel_values"].to(device=torch_device, non_blocking=torch_device.type == "cuda")
        with torch.inference_mode():
            embeddings = model(pixel_values)

        vectors = embeddings.detach().cpu().tolist()
        for original_index, img_name, metadata, vec in zip(batch["indices"], batch["image_names"], batch["metadata"], vectors):
            feature_items.append((original_index, {"key": img_name, "vector": vec, "metadata": metadata}))
    return feature_items


def extract_stage2_features_with_retry(
    input_folder,
    indexed_image_files,
    model,
    processor,
    torch_device,
    batch_size,
    num_workers,
    include_augmentations=False,
    augmentation_count=DEFAULT_AUGMENTATION_COUNT,
):
    current_batch_size = max(1, batch_size)
    while True:
        try:
            return extract_stage2_features_batched(
                input_folder,
                indexed_image_files,
                model,
                processor,
                torch_device,
                current_batch_size,
                num_workers,
                include_augmentations=include_augmentations,
                augmentation_count=augmentation_count,
            )
        except torch.cuda.OutOfMemoryError:
            if torch_device.type == "cuda":
                torch.cuda.empty_cache()
            if current_batch_size == 1:
                raise RuntimeError("CUDA ran out of memory with batch_size=1. Stage 2 cannot finish.")
            current_batch_size = max(1, current_batch_size // 2)
            print(f"CUDA ran out of memory. Retrying with batch_size={current_batch_size}.")


def get_stage2_cuda_devices():
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


def shard_stage2_images(indexed_image_files, num_shards):
    shards = [[] for _ in range(num_shards)]
    for i, item in enumerate(indexed_image_files):
        shards[i % num_shards].append(item)
    return [shard for shard in shards if shard]


def extract_stage2_features_on_device(args):
    (
        input_folder,
        indexed_image_files,
        local_model_path,
        device_id,
        batch_size,
        num_workers,
        include_augmentations,
        augmentation_count,
    ) = args
    torch.cuda.set_device(device_id)
    torch_device = torch.device(f"cuda:{device_id}")
    model, processor = load_stage2_model(local_model_path, torch_device)
    worker_batch_size = batch_size if batch_size is not None else choose_stage2_batch_size(torch_device)
    worker_num_workers = num_workers if num_workers is not None else choose_stage2_num_workers(torch_device)
    feature_count = len(indexed_image_files) * (1 + augmentation_count if include_augmentations else 1)
    print(
        f"GPU {device_id} processing {len(indexed_image_files)} images "
        f"({feature_count} feature samples) with batch_size={worker_batch_size}, workers={worker_num_workers}."
    )
    return extract_stage2_features_with_retry(
        input_folder,
        indexed_image_files,
        model,
        processor,
        torch_device,
        worker_batch_size,
        worker_num_workers,
        include_augmentations=include_augmentations,
        augmentation_count=augmentation_count,
    )


def extract_features_from_folder(
    input_folder="buffer",
    batch_size=None,
    num_workers=None,
    use_multi_gpu=True,
    include_augmentations=False,
    augmentation_count=DEFAULT_AUGMENTATION_COUNT,
):
    """Extract image embeddings from a folder and return vector database items."""
    logging.getLogger("transformers").setLevel(logging.ERROR)

    local_model_path = get_model_assets_path()
    image_files = sorted([
        f for f in os.listdir(input_folder)
        if f.lower().endswith(STAGE2_IMAGE_EXTENSIONS)
    ])
    indexed_image_files = list(enumerate(image_files))

    feature_sample_count = len(image_files) * (1 + augmentation_count if include_augmentations else 1)
    print(f"Extracting features for {len(image_files)} images ({feature_sample_count} feature samples).")
    if len(image_files) == 0:
        return []

    if batch_size == "auto":
        batch_size = None

    cuda_devices = get_stage2_cuda_devices()
    use_sharded_multi_gpu = use_multi_gpu and len(cuda_devices) > 1 and len(image_files) > 1

    print("Starting stage 2: loading the custom embedding model.")
    if use_sharded_multi_gpu:
        active_devices = cuda_devices[:min(len(cuda_devices), len(image_files))]
        shards = shard_stage2_images(indexed_image_files, len(active_devices))
        worker_args = [
            (input_folder, shard, local_model_path, device_id, batch_size, num_workers, include_augmentations, augmentation_count)
            for device_id, shard in zip(active_devices, shards)
        ]
        print(f"Detected {len(active_devices)} GPUs. Using multiprocessing sharded extraction.")

        feature_items = []
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(worker_args), mp_context=context) as executor:
            futures = [executor.submit(extract_stage2_features_on_device, args) for args in worker_args]
            for future in as_completed(futures):
                feature_items.extend(future.result())
    else:
        device = get_device()[0]
        torch_device = torch.device(device)
        actual_batch_size = batch_size if batch_size is not None else choose_stage2_batch_size(torch_device)
        actual_num_workers = num_workers if num_workers is not None else choose_stage2_num_workers(torch_device)
        print(f"Using device {torch_device}, batch_size={actual_batch_size}, workers={actual_num_workers}.")

        model, processor = load_stage2_model(local_model_path, torch_device)
        print("Custom embedding model loaded.")
        feature_items = extract_stage2_features_with_retry(
            input_folder,
            indexed_image_files,
            model,
            processor,
            torch_device,
            actual_batch_size,
            actual_num_workers,
            include_augmentations=include_augmentations,
            augmentation_count=augmentation_count,
        )

    feature_items.sort(key=lambda item: item[0])
    return [item for _, item in feature_items]


def extract_features_to_db(
    input_folder="buffer",
    db_name="features.fvdb",
    batch_size=None,
    num_workers=None,
    use_multi_gpu=True,
    augmentation_count=DEFAULT_AUGMENTATION_COUNT,
):
    """Extract image embeddings and save them in a vector database."""
    db_path = os.path.join(input_folder, db_name)
    feature_items = extract_features_from_folder(
        input_folder=input_folder,
        batch_size=batch_size,
        num_workers=num_workers,
        use_multi_gpu=use_multi_gpu,
        include_augmentations=True,
        augmentation_count=augmentation_count,
    )

    db = VectorDatabase()
    db.add_many(feature_items)
    db.save(db_path)

    if not feature_items:
        print("No processable images found. Created an empty vector database.")
    print(f"Vector database saved to {db_path}.")
    print("Stage 2 complete.")
    return db_path


def append_features_to_db(
    input_folder="buffer",
    db_name="features.fvdb",
    batch_size=None,
    num_workers=None,
    use_multi_gpu=True,
    augmentation_count=DEFAULT_AUGMENTATION_COUNT,
):
    """Append new image embeddings to an existing vector database, skipping duplicate keys."""
    require_feature_db(input_folder, preferred_name=db_name)
    db_path = os.path.join(input_folder, db_name)
    db = VectorDatabase.load_auto(input_folder, preferred_name=db_name)
    existing_keys = set(db.keys)
    feature_items = extract_features_from_folder(
        input_folder=input_folder,
        batch_size=batch_size,
        num_workers=num_workers,
        use_multi_gpu=use_multi_gpu,
        include_augmentations=True,
        augmentation_count=augmentation_count,
    )
    new_items = [item for item in feature_items if item["key"] not in existing_keys]
    skipped_count = len(feature_items) - len(new_items)

    db.add_many(new_items)
    db.save(db_path)

    print(f"Skipped {skipped_count} existing feature keys.")
    print(f"Appended {len(new_items)} new feature keys.")
    print(f"Vector database saved to {db_path}.")
    print("Stage 2 complete.")
    return db_path
