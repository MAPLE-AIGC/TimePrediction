import torch 
import os 
import json
import torch 
import random
from torch.utils.data import DataLoader, Dataset
import torch_utils.distributed as dist
from .sampler import InfiniteSampler
import torchvision.io as tvio
from torchvision import transforms

class MJHQ30K(Dataset):
    
    def __init__(self, meta, root_dir):
        super().__init__()
        with open(meta, 'r') as f:
            self.meta = json.load(f)
        self.meta_keys = list(self.meta.keys())
        self.root_dir = root_dir
        self.tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256)
        ])
    
    def __len__(self):
        return len(self.meta_keys)
    
    def __getitem__(self, index):
        fn = self.meta_keys[index]
        prompt = self.meta[fn]['prompt']
        subdir = self.meta[fn]['category']
        image = self.tf(tvio.read_image(os.path.join(self.root_dir, subdir, fn + '.jpg'), tvio.ImageReadMode.RGB).float() / 255)
        image = (image - 0.5) / .5
        return prompt, image 

    @staticmethod
    def collate_fn(batch):
        prompts = [item[0] for item in batch]
        images = torch.stack([item[1] for item in batch])
        
        return {
            'prompts': prompts, 
            'images': images
        }


def load_mjhq30k(
    data_dir: os.PathLike,
    meta: os.PathLike,
    batch_size: int,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    pin_memory: bool = True,
):
    dataset = MJHQ30K(
        meta=meta, root_dir=data_dir
    )
    
    ddp_sampler = InfiniteSampler(
        dataset, 
        rank=dist.get_rank(), 
        num_replicas=dist.get_world_size(),
        shuffle=False
    )
    
    dl = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=ddp_sampler,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        collate_fn=dataset.collate_fn
    )
    
    return dl