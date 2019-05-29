import numpy as np
import torch
from torchvision import transforms as T
from pathlib import Path
from typing import Union

# Type for str and Path inputs
pathlike = Union[str, Path]


def convert_to_tensor(path: pathlike) -> None:
    """
    Function for converting saved ndarrays (.npy files) to torch tensors.
    Tensors will be saved in a subdirectory called 'tensor_files'.
    INPUT: path to saved ndarrays
    """
    path = Path(path)
    save_dir = path/"tensor_files"
    for array in path.glob("*.npy").iterdir():
        np.load(str(array))



def convert_to_grayscale(path: pathlike) -> None:
    """
    Converts RGB image samples in torch tensors to grayscale.
    Tensors should have the dimensions (N, H, W, 3).
    Saved tensors will have the form (N, H, W, 1).
    INPUT: path to saved torch tensors
    """
    pass

