import numpy as np
import torch
from numpy import ndarray
from torchvision import transforms as T
from torch import Tensor, FloatTensor
from pathlib import Path
from typing import Union, Callable, Sequence, Tuple
from perception.autoencoders import ConvAE, ConvBetaVAE

# Type for str and Path inputs
PathOrStr = Union[str, Path]
"""Custom data type for pathlike input."""

AutoEncoder = Union[ConvAE, ConvBetaVAE]
"""Custom data type for autoencoder models."""

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""Set the device globally if a GPU is available."""


def convert_to_tensor(path: PathOrStr) -> None:
    """
    Converts saved ndarrays (.npy files) to torch tensors.
    Tensors will be saved in path/tensor_dataset.  
    Note: The first 40 samples of each tensor will be left out.  

    ## Parameters:  

    - **path** *(PathOrStr)*: Path to folder containing saved numpy arrays.     
    
    ## Input:  

    - **Saved numpy arrays** *(N - 40: samples, H: height, W: width, 3: # of channels)*:  
        Data batches saved as numpy arrays.  

    ## Output:  

    - **Tensors** *(N - 40: samples, 3: # of channels, H: height, W: width)*:  
        Datas batches in Tensor format with reorded dimensions.
    """
    path = Path(path)
    savedir = path/"tensor_dataset"
    savedir.mkdir(exist_ok=True)

    cropper = T.Compose([T.ToPILImage(),
                            T.CenterCrop((64,64)),
                            T.ToTensor()])

    # convert arrays in target folder to tensor and save
    for array in path.glob("*.npy"):
        image_batch = np.load(array)

        # leave the first 40 frames out as they contain zoomed images
        converted = torch.from_numpy(image_batch[40:, ...])
        # transpose tensor: (n_samples, height, width, # channels) -> (# samples, # channels, h, w)
        converted = torch.einsum("nhwc -> nchw", converted)
        converted = apply(cropper, converted)

        fp = savedir/array.stem
        torch.save(converted, fp.with_suffix(".pt"))
        

def apply(func: Callable[[Tensor], Tensor], M: Tensor, d: int = 0) -> Tensor:
    """
    Applies an operation to a tensor along the a specified dimension.  

    ## Parameters:  

    - **func** *(Callable[[Tensor], Tensor])*: Function with tensor input and tensor output.
    - **M** *(Tensor)*: Input tensor to be processed.  
    - **d** *(int)*: Dimension along which *func* is applied.     
    
    ## Input:  

    - **Tensor**: Any torch tensor.  

    ## Output:  

    - **Tensor**: Tensor with *func* applied to dimension *d*.  
    """
    tList = [func(m) for m in torch.unbind(M, dim=d) ]
    res = torch.stack(tList, dim=d)

    return res 


def convert_to_grayscale(path: PathOrStr) -> None:
    """
    Converts RGB image samples in torch tensors to grayscale using PyTorch transforms.
    The pipeline applied is [ToPILImage, Grayscale, ToTensor].  
    Grayscale tensors will be saved in path/grayscale_dataset.   

    ## Parameters:  

    - **path** *(PathOrStr)*: Path to folder containing images saved as tensors.     
    
    ## Input:  

    - **Tensors** *(N: # samples, 3: # of channels, H: height, W: width)*:  
        Colored image batches.  

    ## Output:  

    - **FloatTensors** *(N: # samples, 3: # of channels, H: height, W: width)*:  
        Image batches as grayscale tensors.
    """
    path = Path(path)
    savedir = path/"grayscale_dataset"
    savedir.mkdir(exist_ok=True)

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
    Script to concatenate tensors along a specified dimension.
    The concatenated data will be saved in the file path/cat_data.pt.  
    
    ## Parameters:  

    - **path** *(PathOrStr)*: Path to folder containing saved tensors.   
    - **d** *(int)*: Dimension for concatenation.  
    
    ## Input:  

    - **Tensors**: Tensors of any shape. 

    ## Output:  

    - **Tensor**:  A single torch tensor.
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
    Files will be saved in the folder path/float_dataset.   

    ## Parameters:  

    - **path** *(PathOrStr)*: Path to folder containing saved tensors.    
    
    ## Input:  

    - **Tensors**: Tensors (CarRacer observations contain integers). 

    ## Output:  

    - **Tensors**:  FloatTensors.
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

    ## Parameters:  

    - **fp** *(PathOrStr)*: Path to a single tensor containing image data.    
    
    ## Input:  

    - **Tensor** *(N: # samples, 3: # of channels, H: height, W: width)*:    
        Tensors containing image data.  

    **Output**:  

    - **Tuple** *(Tuple[Sequence[float], Sequence[float]])*: Tuple containing per channel means and standard deviations.  
    """
    data = torch.load(fp)
    means = [data[:,i,...].mean().item() for i in range(data.shape[1])]
    stds = [data[:,i,...].std().item() for i in range(data.shape[1])]

    return means, stds


def process_observation(obs: ndarray) -> FloatTensor:
    """
    Converts a single CarRacing observation into a tensor ready for autoencoder consumption.
    The transformation pipeline applied is [ToPILImage, CenterCrop(64, 64), ToTensor].     

    ## Parameters:  

    - **obs** *(ndarray)*: Observation from the CarRacing enironvment.     
    
    ## Input:  

    - **ndarray** *(H: height, W: width, 3: # of channels)*:  
        Gym CarRacer-v0 observation.  

    ## Output:  

    - **FloatTensor** *(1: # samples, 3: # of channels, 64: height, 64: width)*:  
        Reshaped image tensor ready for model consumption.
    """
    cropper = T.Compose([T.ToPILImage(),
                            T.CenterCrop((64,64)),
                            T.ToTensor()])
    converted = torch.from_numpy(obs.copy())
    converted.unsqueeze_(0)
    converted = torch.einsum("nhwc -> nchw", converted)
    return apply(cropper, converted).to(DEVICE)


def load_model(path_to_weights: PathOrStr, vae: bool = False) -> AutoEncoder:
    """
    Loads a ConvAE model with pretrained weights. If available the model is loaded to the GPU.  

    ## Parameters:  

    - **path_to_weights** *(PathOrStr)*: Path to trained autoencoder weights.     
    
    ## Input:  

    - **file_name.pt** *(state_dict)*: Weights of the trained convolutional autoencoder model.  

    ## Output:  

    - **ae** *(ConvAE, ConvBetaVAE)*:  Loaded ConvAE model specified in autoencoders.
    """
    if vae:
        ae = ConvBetaVAE()
        ae.load_state_dict(torch.load(path_to_weights, map_location=DEVICE))
    else:
        ae = ConvAE()
        ae.load_state_dict(torch.load(path_to_weights, map_location=DEVICE))

    return ae


if __name__ == '__main__':
    # numpy_path = "/home/timo/DataSets/carracer_images/numpy_dataset"
    # convert_to_tensor(numpy_path)
    # tensor_dataset = "/home/timo/DataSets/carracer_images/color_dataset"
    # convert_to_grayscale(tensor_dataset)

    # Note: the code below needs 16GB RAM
    # color_path = "/home/timo/DataSets/carracer_images/color_dataset"
    gray_path = "/home/timo/DataSets/carracer_images/grayscale_dataset"
    # cat_tensors(color_path)
    cat_tensors(gray_path)