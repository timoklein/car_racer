import numpy as np
import torch
from torchvision import transforms as T
from torch import Tensor
from pathlib import Path
from typing import Union, Callable, Sequence, Tuple

# Type for str and Path inputs
PathOrStr = Union[str, Path]


def convert_to_tensor(path: PathOrStr) -> None:
    """
    Converts saved ndarrays (.npy files) to torch tensors.
    Tensors will be saved in a subdirectory called 'tensor_files'.
    Note: The first 40 samples of each tensor will be left out.
    Array shapes: (N,H,W,3)
    Tensor shapes: (N-40,3,H,W)
    INPUT: 
        path: str or Path       path to saved ndarrays
    """
    path = Path(path)
    savedir = path/"tensor_dataset"
    savedir.mkdir(exist_ok=True)

    # convert arrays in target folder to tensor and save
    for array in path.glob("*.npy"):
        image_batch = np.load(array)

        # leave the first 40 frames out as they contain zoomed images
        converted = torch.from_numpy(image_batch[40:, ...])
        # transpose tensor: (n_samples, height, width, # channels) -> (# samples, # channels, h, w)
        converted = torch.einsum("nhwc -> nchw", converted)

        fp = savedir/array.stem
        torch.save(converted, fp.with_suffix(".pt"))
        

def apply(func: Callable[[Tensor], Tensor], M: Tensor, d: int = 0) -> Tensor:
    """
    Applies an operation to a tensor along the a specified dimension.
    INPUTS:
        func: Callable      Function to be applied
        M: Tensor           Input Tensor
        d: int              Dimension along which the function is applied
    OUTPUTS:
        res: Tensor         Processed tensor
    """
    tList = [func(m) for m in torch.unbind(M, dim=d) ]
    res = torch.stack(tList, dim=d)

    return res 


def convert_to_grayscale(path: PathOrStr) -> None:
    """
    Converts RGB image samples in torch tensors to grayscale.
    Tensors should have the dimensions (N, H, W, 3).
    Saved tensors will have the form (N, 3, H, W) and be of type FloatTensor.
    INPUT: 
        path: str or Path    path to saved torch tensors
    """
    path = Path(path)
    savedir = path/"grayscale_dataset"
    savedir.mkdir(exist_ok=True)

    # grayscaling pipeline using torch transforms
    grayscaler = T.Compose([T.ToPILImage(),
                        T.Grayscale(num_output_channels=3),
                        T.ToTensor()])
    
    for file in path.glob("*.pt"):
        batch = torch.load(file)
        converted = apply(grayscaler, batch)

        fp = savedir/file.stem
        torch.save(converted, fp.with_suffix(".pt"))


def cat_tensors(path: PathOrStr, d: int = 0) -> None:
    """
    Script to look for saved tensors in a folder and concatenate them along a defined axis.
    INPUT:
        path: str or Path   path to saved tensors
        d: int              concatenation dimension
    """
    path = Path(path)

    tlist = []
    for file in path.glob("*.pt"):
        batch = torch.load(file)
        tlist.append(batch)

    res = torch.cat(tlist, dim=d)
    fp = path/"cat_data.pt"
    torch.save(res, fp)


def convert_to_float_tensor(path: PathOrStr) -> None:
    """
    Script converting saved Tensors into FloatTensors ready for model consumption.
    INPUT:
    * __path__(PathOrStr):     Path to saved tensors 
    """
    path = Path(path)
    savedir = path/"float_dataset"
    savedir.mkdir(exist_ok=True)

    for file in path.glob("*.pt"):
        batch = torch.load(file)
        converted = batch.type("torch.FloatTensor")
        fp = savedir/file.stem
        torch.save(converted, fp.with_suffix(".pt"))


def find_mean_std(fp: PathOrStr) -> Tuple[Sequence[float], Sequence[float]]:
    """
    Finds per channel mean and standard deviations for an RGB image dataset.
    Data should be in the form N,C,H,W. Tensor Type FloatTensor.
    INPUT:
    * __fp__(PathOrStr):    Path to saved tensor of images
    OUTPUT:
    * __means, stds__:      Tuple of sequences containing per channel means and standard deviations
    """
    data = torch.load(fp)
    means = [data[:,0,...].mean(), data[:,1,...].mean(), data[:,2,...].mean()]
    stds = [data[:,0,...].std(), data[:,1,...].std(), data[:,2,...].std()]

    return means, stds


def main():
    color_path = "/home/jupyter/tutorials/praktikum_ml/color_dataset"
    cat_tensors(color_path)
    gray_path = "/home/jupyter/tutorials/praktikum_ml/grayscale_dataset"
    cat_tensors(gray_path)


if __name__ == '__main__':
        main()