import os
import torch
import io
import json
import webdataset as wds
from PIL import Image
from functools import partial

import PIL
from torchvision import transforms as T


def preprocess_dataset(item, image_transform, class_mapping):
    output = {}
    image_data = item['jpeg']
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    image_tensor = image_transform(image)
    output["image"] = image_tensor
    output["label"] = class_mapping[item["__key__"].split("_")[0]][0] # [0] for id, [1] for name
    output["key"] = item["__key__"]

    return output

def collate_fn(batch):
    images = torch.stack([item["image"] for item in batch], dim=0)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    
    return {
        "images": images,
        "prompts": labels,
    }


def load_imagenet_with_wds(
    data_dir: os.PathLike,
    batch_size: int,
    image_size: int = 512,
    num_workers: int = 8,
):
    shards = []
    image_transform = T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
    ])
    
    with open(os.path.join(data_dir, "meta.json"), "r") as f:
        class_mapping = json.load(f)

    for fn in os.listdir(data_dir):
        path = os.path.join(data_dir, fn)
        if path.endswith(".tar"): 
            shards.append(path)

    pd = partial(preprocess_dataset, image_transform=image_transform, class_mapping=class_mapping)
    
    dataset = wds.WebDataset(
        urls=shards, shardshuffle=True, nodesplitter=wds.split_by_node, resampled=True, 
    ).map(pd)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
    ).shuffle(10000).batched(batch_size, collation_fn=collate_fn)


    return dataloader



if __name__ == '__main__':
    data_dir = "/huangzemin/niche/datasets/imagenet1k"
    loader = load_imagenet_with_wds(
        data_dir=data_dir,
        batch_size=20, 
        image_size=512,
    )
    
    for item in loader:
        p = item['prompts']
        p.sort()
        print(p)
        break