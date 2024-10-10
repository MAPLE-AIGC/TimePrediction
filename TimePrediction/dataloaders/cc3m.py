import os
import torch
import io
import webdataset as wds
from PIL import Image
from functools import partial

from torchvision import transforms as T


def preprocess_dataset(item, image_transform):
    output = {}

    image_data = item['image']

    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    image_tensor = image_transform(image)
    output["image"] = image_tensor
    output["prompt"] = item['blip_caption'].decode() # [0] for id, [1] for name
    output["key"] = item["__key__"]

    return output

def collate_fn(batch):
    images = torch.stack([item["image"] for item in batch], dim=0)
    prompts = [item["prompt"] for item in batch]
    
    return {
        "images": images,
        "prompts": prompts,
    }


def load_cc3m_with_wds(
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

    for fn in os.listdir(data_dir):
        path = os.path.join(data_dir, fn)
        if path.endswith(".tar"): 
            shards.append(path)

    pd = partial(preprocess_dataset, image_transform=image_transform)
    
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
    data_dir = "/storage/huangzemin/CC3M_subset"
    loader = load_cc3m_with_wds(
        data_dir=data_dir,
        batch_size=20, 
        image_size=256,
    )
    
    for item in loader:
        p = item['prompts']
        p.sort()
        print(p)
        break