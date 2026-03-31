from dataclasses import dataclass
import torch

@dataclass
class Intrinsics:
    """Camera intrinsic parameters.
    
    Attributes:
        fx (torch.Tensor): Focal length in x direction
        fy (torch.Tensor): Focal length in y direction
        cx (torch.Tensor): Principal point x-coordinate
        cy (torch.Tensor): Principal point y-coordinate
        image_width (int): Width of the camera image
        image_height (int): Height of the camera image"""
        
    # intrinsics: torch.Tensor  # (fx, fy, cx, cy)
    fx: torch.Tensor
    fy: torch.Tensor
    cx: torch.Tensor
    cy: torch.Tensor
    image_width: int
    image_height: int
    