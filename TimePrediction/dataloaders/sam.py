import os 
import re 
from torchvision import transforms
from PIL import Image 
import io 
import torch 
import webdataset as wds
import json
from torch.utils.data import DataLoader
from functools import partial

import torch_utils.distributed as dist
from .sampler import InfiniteSampler

pre_processor = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

def filter_dataset(item, propmt_path):
    prompt_exist = os.path.exists(os.path.join(propmt_path, item["__key__"][2:] + ".json"))
    if "jpg" not in item or not prompt_exist:
        return False
    
    return True

def preprocess_dataset(item, propmt_path):
    output_dict = {}
    orig_image = Image.open(io.BytesIO(item["jpg"])).convert("RGB")
    image = pre_processor(orig_image)

    output_dict["image"] = image
    # output_dict["prompt"] = "test"
    with open(os.path.join(propmt_path, item["__key__"][2:] + ".json"), "r") as f:
        output_dict["prompt"] = json.load(f)['<DETAILED_CAPTION>']

    return output_dict

def collate_fn(batch):
    images = torch.stack([item["image"] for item in batch], dim=0)
    prompts = [item["prompt"] for item in batch]
    return {
        "images": images,
        "prompts": prompts
    }


def load_sam_with_wds(
    data_path: str,
    propmt_path: str,
    batch_size: int,
    num_workers: int = 4
):
    pattern = r".*\.tar$"
    if isinstance(data_path, list):
        tar_files = []
        for path in data_path:
            tar_files.extend([f for f in os.listdir(path) if re.match(pattern, f)])
            tar_files = [os.path.join(path, f) for f in tar_files]
    else:
        tar_files = [f for f in os.listdir(data_path) if re.match(pattern, f)]
        tar_files = [os.path.join(data_path, f) for f in tar_files]
        
    dataset = wds.WebDataset(
            tar_files,
            resampled=False,
            nodesplitter=wds.split_by_node
        )\
            .select(partial(filter_dataset, propmt_path=propmt_path))\
            .map(partial(preprocess_dataset, propmt_path=propmt_path))\
            .batched(batch_size, collation_fn=collate_fn, partial=False)
            
    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader
