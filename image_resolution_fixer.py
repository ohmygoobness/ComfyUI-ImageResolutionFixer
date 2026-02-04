"""
ComfyUI Image Resolution Fixer Node
Resizes images to compatible resolutions for models that require specific dimension constraints
"""

import torch
import numpy as np
from PIL import Image
import math
import cv2

class ImageResolutionFixer:
    """
    A ComfyUI node that takes an image and outputs it with a fixed/compatible resolution.
    Handles various resizing methods and ensures dimensions are divisible by specified values.
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "fit": (["smart_fill", "letterbox", "crop", "fill"], {
                    "default": "smart_fill"
                }),
                "method": (["lanczos", "bicubic", "hamming", "bilinear", "box", "nearest"],),
                "round_to_multiple": ([2, 4, 8, 14, 16, 28, 32, 64, 128, 256, 512], {
                    "default": 16
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "resize_image"
    CATEGORY = "image/transform"
    
    def get_resampling_method(self, method_name):
        """Convert method name to PIL resampling filter"""
        method_map = {
            "lanczos": Image.Resampling.LANCZOS,
            "bicubic": Image.Resampling.BICUBIC,
            "hamming": Image.Resampling.HAMMING,
            "bilinear": Image.Resampling.BILINEAR,
            "box": Image.Resampling.BOX,
            "nearest": Image.Resampling.NEAREST,
        }
        return method_map.get(method_name, Image.Resampling.LANCZOS)
    
    def round_to_multiple(self, value, multiple):
        """Round a value to the nearest multiple"""
        return int(math.ceil(value / multiple) * multiple)
    
    def calculate_target_dimensions(self, orig_width, orig_height, round_multiple):
        """Calculate target dimensions by rounding to nearest multiple"""
        new_width = self.round_to_multiple(orig_width, round_multiple)
        new_height = self.round_to_multiple(orig_height, round_multiple)
        return new_width, new_height
    
    def resize_letterbox(self, pil_image, target_width, target_height, resampling):
        """Resize with letterboxing (maintain aspect ratio, add padding)"""
        orig_width, orig_height = pil_image.size
        
        # Calculate scaling to fit within target dimensions
        scale = min(target_width / orig_width, target_height / orig_height)
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)
        
        # Resize image
        resized = pil_image.resize((new_width, new_height), resampling)
        
        # Create new image with target dimensions (black background)
        result = Image.new('RGB', (target_width, target_height), (0, 0, 0))
        
        # Paste resized image centered
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2
        result.paste(resized, (paste_x, paste_y))
        
        return result
    
    def resize_crop(self, pil_image, target_width, target_height, resampling):
        """Resize with center crop (fill target, crop excess)"""
        orig_width, orig_height = pil_image.size
        
        # Calculate scaling to cover target dimensions
        scale = max(target_width / orig_width, target_height / orig_height)
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)
        
        # Resize image
        resized = pil_image.resize((new_width, new_height), resampling)
        
        # Calculate crop coordinates (center crop)
        left = (new_width - target_width) // 2
        top = (new_height - target_height) // 2
        right = left + target_width
        bottom = top + target_height
        
        return resized.crop((left, top, right, bottom))
    
    def resize_fill(self, pil_image, target_width, target_height, resampling):
        """Resize to fill (stretch to fit, may distort aspect ratio)"""
        return pil_image.resize((target_width, target_height), resampling)
    
    def resize_smart_fill(self, pil_image, target_width, target_height, resampling):
        """
        Resize with smart fill using OpenCV Border Reflection.
        Instead of stretching pixels (which causes smearing), this mirrors the 
        image content at the edges. This is fast, algorithmic, and looks natural.
        """
        orig_width, orig_height = pil_image.size
        
        # 1. Resize the image first to fit within the target bounds while maintaining aspect ratio
        scale = min(target_width / orig_width, target_height / orig_height)
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)
        
        resized_pil = pil_image.resize((new_width, new_height), resampling)
        
        # 2. Convert to NumPy for OpenCV operations
        # PIL (RGB) -> NumPy (RGB)
        img_np = np.array(resized_pil)
        
        # 3. Calculate padding requirements
        pad_w = target_width - new_width
        pad_h = target_height - new_height
        
        # Calculate padding for each side (center the image)
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        
        # 4. Apply Border Reflection
        # BORDER_REFLECT_101 mirrors pixels: gfedcb|abcdefgh|gfedcba
        # This avoids repeating the edge pixel itself and creates a smooth texture transition
        result_np = cv2.copyMakeBorder(
            img_np, 
            top, bottom, left, right, 
            cv2.BORDER_REFLECT_101
        )
        
        # 5. Convert back to PIL
        return Image.fromarray(result_np)
    
    def resize_image(self, image, fit, method, round_to_multiple):
        """Main function to resize image with specified parameters"""
        
        # Convert ComfyUI image tensor to PIL Image
        # ComfyUI images are in format: [batch, height, width, channels]
        batch_size = image.shape[0]
        results = []
        
        for i in range(batch_size):
            # Get single image from batch
            img_tensor = image[i]
            
            # Convert to numpy and then PIL
            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(img_np)
            
            orig_width, orig_height = pil_image.size
            
            # Calculate target dimensions (just round to nearest multiple)
            target_width, target_height = self.calculate_target_dimensions(
                orig_width, orig_height, round_to_multiple
            )
            
            # Get resampling method
            resampling = self.get_resampling_method(method)
            
            # Apply resize based on fit mode
            if fit == "letterbox":
                result_image = self.resize_letterbox(pil_image, target_width, target_height, resampling)
            elif fit == "crop":
                result_image = self.resize_crop(pil_image, target_width, target_height, resampling)
            elif fit == "fill":
                result_image = self.resize_fill(pil_image, target_width, target_height, resampling)
            elif fit == "smart_fill":
                result_image = self.resize_smart_fill(pil_image, target_width, target_height, resampling)
            
            # Convert back to tensor
            result_np = np.array(result_image).astype(np.float32) / 255.0
            result_tensor = torch.from_numpy(result_np)
            results.append(result_tensor)
        
        # Stack batch back together
        output_tensor = torch.stack(results, dim=0)
        
        # Return image tensor and final dimensions
        final_width = target_width
        final_height = target_height
        
        return (output_tensor, final_width, final_height)


# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "ImageResolutionFixer": ImageResolutionFixer
}

# Display name mappings for ComfyUI
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageResolutionFixer": "Image Resolution Fixer"
}
