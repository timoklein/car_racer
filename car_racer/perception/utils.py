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
    Array shapes: (N,H,W,3)
    Tensor shapes: (N,1,H,W)
    INPUT: path to saved ndarrays
    """
    path = Path(path)
    # convert arrays in target folder to tensors and save
    for array in path.glob("*.npy"):
        image_batch = np.load(str(array))

        # leave the first 40 frames out as they contain zoomed images
        converted = torch.from_numpy(image_batch[40:, ...])
        # transpose tensor: (n_samples, height, width, # channels) -> (# samples, # channels, h, w)
        converted = torch.einsum("nhwc -> nchw", converted)
        fp = path/array.stem
        torch.save(converted, fp.with_suffix(".pt"))


def convert_to_grayscale(path: pathlike) -> None:
    """
    Converts RGB image samples in torch tensors to grayscale.
    Tensors should have the dimensions (N, H, W, 3).
    Saved tensors will have the form (N, 1, H, W).
    INPUT: path to saved torch tensors
    """
    # grayscaling pipeline using torch transforms
    grayscaler = T.Compose([T.ToPILImage(),
                        T.Grayscale(num_output_channels=3),
                        T.ToTensor()])

    path = Path(path)
    savedir = path/"grayscale"

    for batch in path.glob("*.pt"):
        tensor = torch.load(str(batch))
        for row in tensor:
            converted = grayscaler(row)


def main():
    path = "/home/timo/DataSets/carracer_images/"
    convert_to_tensor(path)


if __name__ == '__main__':
        main()