from .utils import (
    inception_normalize,      # Inception model-specific normalization
    imagenet_normalize,      # ImageNet dataset-specific normalization
    MinMaxResize,            # Resizing images while maintaining aspect ratio
)
from PIL import Image
from torchvision import transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from .randaug import RandAugment  # RandAugment for data augmentation

def pixelbert_transform(size=800):
    """
    PixelBERT-specific transformation pipeline.
    Resizes images to a fixed size, converts them to tensors, and applies Inception normalization.
    
    Args:
        size (int): The shorter dimension size to which the image should be resized. Default is 800.
        
    Returns:
        Compose: A composed transformation for PixelBERT.
    """
    longer = int((1333 / 800) * size)  # Maintain aspect ratio for resizing
    return transforms.Compose(
        [
            MinMaxResize(shorter=size, longer=longer),  # Resizing with aspect ratio
            transforms.ToTensor(),  # Convert image to tensor
            inception_normalize,  # Apply Inception model normalization
        ]
    )

def pixelbert_transform_randaug(size=800):
    """
    PixelBERT transformation with RandAugment for additional data augmentation.
    
    Args:
        size (int): The shorter dimension size to which the image should be resized. Default is 800.
        
    Returns:
        Compose: A composed transformation with RandAugment for PixelBERT.
    """
    longer = int((1333 / 800) * size)  # Maintain aspect ratio for resizing
    trs = transforms.Compose(
        [
            MinMaxResize(shorter=size, longer=longer),  # Resizing with aspect ratio
            transforms.ToTensor(),  # Convert image to tensor
            inception_normalize,  # Apply Inception model normalization
        ]
    )
    trs.transforms.insert(0, RandAugment(2, 9))  # Apply RandAugment for more variations
    return trs

def imagenet_transform(size=800):
    """
    ImageNet-specific transformation pipeline.
    Resizes the image, applies center crop, and normalizes for ImageNet models.
    
    Args:
        size (int): The size for resizing and cropping. Default is 800.
        
    Returns:
        Compose: A composed transformation for ImageNet.
    """
    return transforms.Compose(
        [
            Resize(size, interpolation=Image.BICUBIC),  # Resize image with bicubic interpolation
            CenterCrop(size),  # Crop the center of the image
            transforms.ToTensor(),  # Convert image to tensor
            imagenet_normalize,  # Normalize image for ImageNet
        ]
    )

def imagenet_transform_randaug(size=800):
    """
    ImageNet transformation with RandAugment for enhanced data augmentation.
    
    Args:
        size (int): The size for resizing and cropping. Default is 800.
        
    Returns:
        Compose: A composed transformation with RandAugment for ImageNet.
    """
    trs = transforms.Compose(
        [
            Resize(size, interpolation=Image.BICUBIC),  # Resize image with bicubic interpolation
            CenterCrop(size),  # Crop the center of the image
            transforms.ToTensor(),  # Convert image to tensor
            imagenet_normalize,  # Normalize image for ImageNet
        ]
    )
    trs.transforms.insert(0, RandAugment(2, 9))  # Apply RandAugment for additional data augmentation
    return trs

def vit_transform(size=800):
    """
    ViT (Vision Transformer) specific transformation pipeline.
    Similar to ImageNet transform, but uses Inception normalization.

    Args:
        size (int): The size for resizing and cropping. Default is 800.
        
    Returns:
        Compose: A composed transformation for ViT.
    """
    return transforms.Compose(
        [
            Resize(size, interpolation=Image.BICUBIC),  # Resize image with bicubic interpolation
            CenterCrop(size),  # Crop the center of the image
            lambda image: image.convert("RGB"),  # Convert to RGB
            transforms.ToTensor(),  # Convert image to tensor
            inception_normalize,  # Apply Inception normalization
        ]
    )

def vit_transform_randaug(size=800):
    """
    ViT transformation with RandAugment for enhanced data augmentation.
    
    Args:
        size (int): The size for resizing and cropping. Default is 800.
        
    Returns:
        Compose: A composed transformation with RandAugment for ViT.
    """
    trs = transforms.Compose(
        [
            Resize(size, interpolation=Image.BICUBIC),  # Resize image with bicubic interpolation
            CenterCrop(size),  # Crop the center of the image
            lambda image: image.convert("RGB"),  # Convert to RGB
            transforms.ToTensor(),  # Convert image to tensor
            inception_normalize,  # Apply Inception normalization
        ]
    )
    trs.transforms.insert(0, lambda image: image.convert('RGBA'))  # Apply RGBA conversion for RandAugment
    trs.transforms.insert(0, RandAugment(2, 9))  # Apply RandAugment for additional variations
    trs.transforms.insert(0, lambda image: image.convert('RGB'))  # Convert back to RGB after RandAugment
    return trs

def clip_transform(size):
    """
    CLIP-specific transformation pipeline.
    Resizes the image, crops it, and applies CLIP normalization.
    
    Args:
        size (int): The size for resizing and cropping.
        
    Returns:
        Compose: A composed transformation for CLIP.
    """
    return Compose([
        Resize(size, interpolation=Image.BICUBIC),  # Resize image with bicubic interpolation
        CenterCrop(size),  # Crop the center of the image
        lambda image: image.convert("RGB"),  # Convert to RGB
        ToTensor(),  # Convert image to tensor
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),  # Normalize for CLIP
    ])

def clip_transform_randaug(size):
    """
    CLIP transformation with RandAugment for enhanced data augmentation.
    
    Args:
        size (int): The size for resizing and cropping.
        
    Returns:
        Compose: A composed transformation with RandAugment for CLIP.
    """
    trs = Compose([
        Resize(size, interpolation=Image.BICUBIC),  # Resize image with bicubic interpolation
        CenterCrop(size),  # Crop the center of the image
        lambda image: image.convert("RGB"),  # Convert to RGB
        ToTensor(),  # Convert image to tensor
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),  # Normalize for CLIP
    ])
    trs.transforms.insert(0, lambda image: image.convert('RGBA'))  # Apply RGBA conversion for RandAugment
    trs.transforms.insert(0, RandAugment(2, 9))  # Apply RandAugment for additional variations
    trs.transforms.insert(0, lambda image: image.convert('RGB'))  # Convert back to RGB after RandAugment
    return trs
