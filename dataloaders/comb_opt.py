import os
import torch
import io
import webdataset as wds
from PIL import Image
import random
import json
import torch.nn.functional as F
from functools import partial

from transformers import AutoTokenizer
from torchvision import transforms as T


def filter_dataset(item):
    # filter out cases without image
    if "jpg" not in item and "image" not in item:
        return False
    if "jpg" in item and 'json' not in item:
        return False
    if "jpg" in item:
        # filter out the images with low original resolution
        file_items = json.loads(item['json'].decode('utf-8'))
        original_width, original_height = file_items['original_width'], file_items['original_height']
        if original_height < 128 or original_width < 128:
            return False

        try:
            item['pil'] = Image.open(io.BytesIO(item['jpg'])).convert("RGB")
        except Exception as e:
            return False 
    
    return True


def preprocess_dataset(item, image_transform, tokenizer, max_len=256):
    output = {}
    if "image" in item:
        image_data = item['image']

        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image_tensor = image_transform(image)
        output["image"] = image_tensor
        output["prompt"] = item['blip_caption'].decode() # [0] for id, [1] for name
        output["key"] = item["__key__"]
    else:
        image = item["pil"]
        image_tensor = image_transform(image)
        output["image"] = image_tensor
        file_items = json.loads(item['json'].decode('utf-8'))
        txt_key = '<DETAILED_CAPTION>'
        if '<DETAILED_CAPTION>' not in file_items:
            txt_key = 'caption'
        txt_content = file_items[txt_key].strip()
        output["prompt"] = txt_content
        output['key'] = item['__key__']

    tokens = tokenizer(
            output["prompt"],
            max_length=max_len,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )['input_ids']
    output['token_count'] = tokens.size(1)
    output['tokens'] = F.pad(
        tokens, pad=(max_len - tokens.size(1), 0), mode='constant', value=0
    )
    output['padding_mask'] = torch.zeros(max_len, dtype=torch.bool)
    output['padding_mask'][:max_len - tokens.size(1)] = True
    output['text_id'] = F.pad(torch.stack(
        [torch.arange(0, tokens.size(1)), torch.arange(0, tokens.size(1))],
        dim=-1),
        pad=(0, 0, max_len - tokens.size(1), 0), mode='constant', value=0
    )
    
    return output

def collate_fn(batch):
    images = torch.stack([item["image"] for item in batch], dim=0)
    prompts = [item["prompt"] for item in batch]
    tokens = torch.concat([item["tokens"] for item in batch], dim=0)
    token_counts = torch.tensor([item["token_count"] for item in batch], dtype=torch.long)
    padding_masks = torch.stack([item["padding_mask"] for item in batch], dim=0)
    text_ids = torch.stack([item["text_id"] for item in batch], dim=0)
    max_batch_seq_len = token_counts.max().item()
    
    # truncate tokens
    tokens = tokens[:, -max_batch_seq_len:]
    padding_masks = padding_masks[:, -max_batch_seq_len:]
    text_ids = text_ids[:, -max_batch_seq_len:]
    
    return {
        "images": images,
        "prompts": prompts,
        "tokens": tokens,
        'token_counts': token_counts,
        "padding_masks": padding_masks,
        'text_ids': text_ids,
    }


def load_comb_with_wds(
    data_dir: os.PathLike,
    data_dir2: os.PathLike,
    tokenizer,
    batch_size: int,
    max_token_len: int = 256,
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
    
    for fn in os.listdir(data_dir2):
        path = os.path.join(data_dir2, fn)
        if path.endswith(".tar"): 
            shards.append(path)        
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    
    # shuffle shards
    random.shuffle(shards)
    pd = partial(
        preprocess_dataset, 
        image_transform=image_transform, 
        tokenizer=tokenizer, 
        max_len=max_token_len
    )
    
    dataset = wds.WebDataset(
        urls=shards, shardshuffle=True, nodesplitter=wds.split_by_node, resampled=True, 
    ).select(filter_dataset).map(pd)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
    ).shuffle(10000).batched(batch_size, collation_fn=collate_fn)

    return dataloader