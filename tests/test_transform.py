import pytest
import torch
from PIL import Image
from torchvision import transforms
from .transforms import (pixelbert_transform, pixelbert_transform_randaug, imagenet_transform,
                         imagenet_transform_randaug, vit_transform, vit_transform_randaug,
                         clip_transform, clip_transform_randaug)

@pytest.fixture
def dummy_image():
    # Create a dummy image (RGB) of size 800x800
    return Image.new("RGB", (800, 800), color=(255, 0, 0))

def test_pixelbert_transform(dummy_image):
    transform = pixelbert_transform(size=800)
    transformed_image = transform(dummy_image)
    
    assert isinstance(transformed_image, torch.Tensor), "Output should be a tensor."
    assert transformed_image.shape == (3, 800, 800), f"Expected shape (3, 800, 800), but got {transformed_image.shape}"

def test_pixelbert_transform_randaug(dummy_image):
    transform = pixelbert_transform_randaug(size=800)
    transformed_image = transform(dummy_image)
    
    assert isinstance(transformed_image, torch.Tensor), "Output should be a tensor."
    assert transformed_image.shape == (3, 800, 800), f"Expected shape (3, 800, 800), but got {transformed_image.shape}"

def test_imagenet_transform(dummy_image):
    transform = imagenet_transform(size=800)
    transformed_image = transform(dummy_image)
    
    assert isinstance(transformed_image, torch.Tensor), "Output should be a tensor."
    assert transformed_image.shape == (3, 800, 800), f"Expected shape (3, 800, 800), but got {transformed_image.shape}"

def test_imagenet_transform_randaug(dummy_image):
    transform = imagenet_transform_randaug(size=800)
    transformed_image = transform(dummy_image)
    
    assert isinstance(transformed_image, torch.Tensor), "Output should be a tensor."
    assert transformed_image.shape == (3, 800, 800), f"Expected shape (3, 800, 800), but got {transformed_image.shape}"

def test_vit_transform(dummy_image):
    transform = vit_transform(size=800)
    transformed_image = transform(dummy_image)
    
    assert isinstance(transformed_image, torch.Tensor), "Output should be a tensor."
    assert transformed_image.shape == (3, 800, 800), f"Expected shape (3, 800, 800), but got {transformed_image.shape}"

def test_vit_transform_randaug(dummy_image):
    transform = vit_transform_randaug(size=800)
    transformed_image = transform(dummy_image)
    
    assert isinstance(transformed_image, torch.Tensor), "Output should be a tensor."
    assert transformed_image.shape == (3, 800, 800), f"Expected shape (3, 800, 800), but got {transformed_image.shape}"

def test_clip_transform(dummy_image):
    transform = clip_transform(size=800)
    transformed_image = transform(dummy_image)
    
    assert isinstance(transformed_image, torch.Tensor), "Output should be a tensor."
    assert transformed_image.shape == (3, 800, 800), f"Expected shape (3, 800, 800), but got {transformed_image.shape}"

def test_clip_transform_randaug(dummy_image):
    transform = clip_transform_randaug(size=800)
    transformed_image = transform(dummy_image)
    
    assert isinstance(transformed_image, torch.Tensor), "Output should be a tensor."
    assert transformed_image.shape == (3, 800, 800), f"Expected shape (3, 800, 800), but got {transformed_image.shape}"
