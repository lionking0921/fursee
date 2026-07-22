import random

import numpy as np
from PIL import Image, ImageEnhance


DEFAULT_AUGMENTATION_COUNT = 4
OP_PROBABILITY = 0.8


def ensure_rgb(image):
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        return background
    return image.convert("RGB")


def random_rotate(image):
    angle = random.uniform(-45, 45)
    return image.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(0, 0, 0))


def mirror_flip(image):
    if random.randint(0, 1) == 0:
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    return image


def shear_transform(image):
    width, height = image.size
    shear_x = random.uniform(-0.3, 0.3)
    shear_y = random.uniform(-0.3, 0.3)

    new_width = int(width + abs(shear_x * height) + abs(shear_y * width))
    new_height = int(height + abs(shear_y * width) + abs(shear_x * height))

    offset_x = (width - new_width) / 2.0 - shear_x * (new_height / 2.0)
    offset_y = (height - new_height) / 2.0 - shear_y * (new_width / 2.0)
    matrix = (1, shear_x, offset_x, shear_y, 1, offset_y)

    return image.transform(
        (new_width, new_height),
        Image.AFFINE,
        matrix,
        resample=Image.BICUBIC,
        fillcolor=0,
    )


def color_distortion(image):
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.6, 1.4))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.6, 1.4))
    image = ImageEnhance.Color(image).enhance(random.uniform(0.6, 1.4))
    return image


def add_random_blocks(image):
    width, height = image.size
    total_area = width * height
    max_block_area = total_area * 0.20

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    current_area = 0
    block_count = 0

    min_side = min(width, height)
    block_min_size = max(10, int(min_side * 0.05))
    block_max_size = max(20, int(min_side * 0.25))

    while current_area < max_block_area:
        block_width = random.randint(block_min_size, block_max_size)
        block_height = random.randint(block_min_size, block_max_size)
        block_area = block_width * block_height
        if current_area + block_area > max_block_area:
            break
        if block_width >= width or block_height >= height:
            break

        x = random.randint(0, width - block_width)
        y = random.randint(0, height - block_height)
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
            255,
        )
        angle = random.uniform(0, 360)
        temp_rect = Image.new("RGBA", (block_width, block_height), color)
        rotated_rect = temp_rect.rotate(angle, expand=True, resample=Image.BICUBIC)
        overlay.paste(rotated_rect, (x, y), rotated_rect)

        current_area += block_area
        block_count += 1
        if block_count > 50:
            break

    if image.mode != "RGBA":
        image = image.convert("RGBA")
    return Image.alpha_composite(image, overlay)


def lens_effect(image):
    image = ensure_rgb(image)
    image_array = np.array(image)
    height, width = image_array.shape[:2]

    center_x = random.uniform(width * 0.2, width * 0.8)
    center_y = random.uniform(height * 0.2, height * 0.8)
    min_side = min(width, height)
    radius = random.uniform(min_side * 0.15, min_side * 0.5)

    strength = random.uniform(0.5, 1.2)
    if random.random() > 0.5:
        strength = -strength

    x, y = np.meshgrid(np.arange(width), np.arange(height))
    dist_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    mask = dist_from_center < radius

    if np.any(mask):
        normalized_dist = dist_from_center[mask] / radius
        mapping_factor = 1 + strength * (normalized_dist ** 2 - 1)

        new_x = center_x + (x[mask] - center_x) * mapping_factor
        new_y = center_y + (y[mask] - center_y) * mapping_factor

        new_x = np.clip(new_x, 0, width - 1).astype(np.int32)
        new_y = np.clip(new_y, 0, height - 1).astype(np.int32)

        result_array = image_array.copy()
        result_array[mask] = image_array[new_y, new_x]
        return Image.fromarray(result_array)

    return image


def augment_image(original_image, seq_num):
    image = original_image.copy()

    if seq_num == 0:
        try:
            image = add_random_blocks(image)
        except Exception as e:
            print(f"  [warning] augmentation step add_random_blocks failed: {e}")
    else:
        operations = [
            (random_rotate, OP_PROBABILITY),
            (shear_transform, OP_PROBABILITY * 0.5),
            (lens_effect, OP_PROBABILITY * 0.6),
            (mirror_flip, OP_PROBABILITY),
            (color_distortion, min(1.0, OP_PROBABILITY * 1.2)),
        ]
        random.shuffle(operations)

        for op_func, probability in operations:
            if random.random() < probability:
                try:
                    image = op_func(image)
                except Exception as e:
                    print(f"  [warning] augmentation step {op_func.__name__} failed: {e}")

    return ensure_rgb(image)


def generate_augmented_images(original_image, count=DEFAULT_AUGMENTATION_COUNT):
    return [augment_image(original_image, index) for index in range(count)]
