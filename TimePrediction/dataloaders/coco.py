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

class COCOCaption(Dataset):
    
    def __init__(self, json_file, root_dir, img_size=256, unconditional=False):
        super().__init__()
        self.unconditional = unconditional
        with open(json_file, 'r') as f:
            self.data_json = json.load(f)
        self.root_dir = root_dir
        self.image_to_caption = {}
        for item in self.data_json['annotations']:
            image_id = item['image_id']
            caption = item['caption']
            if image_id not in self.image_to_caption:
                self.image_to_caption[image_id] = []
            self.image_to_caption[image_id].append(caption)
        self.tf = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size)
        ])
    
    def __len__(self):
        return len(self.data_json['images'])
    
    def __getitem__(self, index):
        image_item = self.data_json['images'][index]
        image_id = image_item['id']
        image_fn = image_item['file_name']
        image = self.tf(tvio.read_image(os.path.join(self.root_dir, image_fn), tvio.ImageReadMode.RGB).float() / 255)
        image = (image - 0.5) / .5
        captions = self.image_to_caption[image_id]
        # random choose a caption
        caption = random.choice(captions) if not self.unconditional else ''
        return caption, image 

    @staticmethod
    def collate_fn(batch):
        prompts = [item[0] for item in batch]
        images = torch.stack([item[1] for item in batch])
        
        return {
            'prompts': prompts, 
            'images': images
        }


def load_coco_caption(
    image_dir: os.PathLike,
    json_file: os.PathLike,
    batch_size: int,
    img_size: int = 256,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    pin_memory: bool = True,
    unconditional: bool = False,
):
    dataset = COCOCaption(
        json_file=json_file, root_dir=image_dir, img_size=img_size, unconditional=unconditional,
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


if __name__ == "__main__":
    dl = load_coco_caption(
        image_dir='/mnt/data/coco/images/train2017',
        json_file='/mnt/data/coco/annotations/captions_train2017.json',
        batch_size=2
    )
    for batch in dl:
        print(batch['prompts'])
        print(batch['images'].shape)
        break