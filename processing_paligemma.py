from typing import Dict, List, Optional, Union, Tuple, Iterable
import numpy as np
from PIL import Image
import torch

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]


def add_image_tokens_to_prompt(
    prefix_prompt,
    bos_token,
    image_seq_length,
    image_token,
):
    return f"{image_token * image_seq_length}{bos_token}{prefix_prompt}\n"



def resize(
    image: Image.Image,
    size: Tuple[int, int],
    resample: Image.Resampling = None,
    reducing_gap: int = None
) -> np.ndarray:
    height, width = size
    return image.resize((width, height), resample=resample, reducing_gap=reducing_gap)


def rescale(
    image: np.ndarray,
    scale: float,
    dtype: np.dtype = np.float32
) -> np.ndarray:
    return (image * scale).astype(dtype)


def normalize(
    image: np.ndarray,
    mean: List[float],
    std: List[float],
    dtype: np.dtype = np.float32
) -> np.ndarray:
    mean = np.array(mean, dtype=dtype)
    std = np.array(std, dtype=dtype)
    return (image - mean) / std


def process_images(
    images: List[Image.Image],
    size: Tuple[int, int] = None,
    resample: int = None,
    rescale_factor: float = None,
    image_mean: List[float] = None,
    image_std: List[float] = None,
) -> List[np.ndarray]:
    height, width = size
    
    images = [
        resize(image=image, size=(height, width), resample=resample) for image in images
    ]

    # convert to np arrays
    images = [np.array(image) for image in images]

    # rescale the pixel values
    images = [rescale(image, scale=rescale_factor) for image in images]

    # normalize the pixel values
    images = [normalize(image, mean=image_mean, std=image_std) for image in images]

    # move the channel dimension to the first dimension, The model expects images in shape [channels, height, width]
    images = [image.transpose(2, 0, 1) for image in images]

    return images


class PaliGemmaProcessor:
    
    IMAGE_TOKEN = "<image>"
    
    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):
        # super().__init__()

        self.image_seq_length = num_image_tokens
        self.image_size = image_size

        # we will use gemma model tokenizer, and this tokenizer wasn't created with the special tokens
        # needed for vision, so we will add them manually
        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)

        EXTRA_TOKENS = [
            f"<loc{i:04d}>" for i in range(1024)
        ]  # these tokens are used for object detection (boudning boxes)

        EXTRA_TOKENS += [
            f"<seg{i:03d}>" for i in range(128)
        ]  # these tokens are used for object segemtation (masks)
        tokenizer.add_tokens(EXTRA_TOKENS)

        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)

        # we will add the BOS and EOS tokens ourselves
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer


    def __call__(
        self, 
        text: List[str],
        images: List[Image.Image],
        padding: str = "longest",
        truncation: bool = True,
        ) -> dict:
        # we just care about working with a single image and prompt at a time now
        assert len(images) == 1 and len(text) == 1, f"received {len(images)} images for {len(text)} prompts"

        pixel_values = process_images(
            images,
            size=(self.image_size, self.image_size),
            resample=Image.Resampling.BICUBIC,
            rescale_factor= 1/255.0,
            image_mean=IMAGENET_STANDARD_MEAN,
            image_std=IMAGENET_STANDARD_STD,
        )

        # convert the list of numpy arrays to a single tensor with shape [batch_size, channels, height, width]
        pixel_values = torch.tensor(np.stack(pixel_values, axis=0))

        # prepend a `self.image_seq_length` number of image tokens to the prompt
        inputs_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                image_seq_length=self.image_seq_length,
                image_token=self.IMAGE_TOKEN,
            )
            for prompt in text
        ]

        inputs = self.tokenizer(
            inputs_strings,
            return_tensors="pt",
            padding=padding,
            truncation=truncation,
        )

        return {"pixles_values": pixel_values, **inputs}