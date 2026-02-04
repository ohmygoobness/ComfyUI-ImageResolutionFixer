"""
Image Resolution Fixer - ComfyUI Custom Node
Fixes image resolutions to be compatible with models requiring specific dimension constraints
"""

from .image_resolution_fixer import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
