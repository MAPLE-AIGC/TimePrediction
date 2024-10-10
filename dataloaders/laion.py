import torch
from torchvision import transforms
import webdataset as wds
import os
import io
import re
from PIL import Image
import json
import tarfile


def find_tar_files(folder, pattern):
    tar_files = []
    for root, dirs, files in os.walk(folder):
        tar_files.extend([os.path.join(root, f) for f in files if re.match(pattern, f)])
    return tar_files


def read_caption_from_tar(tar_filename, basename, target_json_key):
    try:
        with tarfile.open(tar_filename, 'r') as tar:
            for member in tar.getmembers():
                if member.name.startswith(basename) and member.name.endswith('.json'):
                    json_file = tar.extractfile(member)
                    json_data = json.load(json_file)
                    if json_data is None or target_json_key not in json_data:
                        print(f'Reading caption json error: {tar_filename}, {basename}.json, {target_json_key}')
                    return json_data[target_json_key]
    except (tarfile.TarError, json.JSONDecodeError) as e:
        raise ValueError(f'Reading caption error: {tar_filename}, {basename}.json, {target_json_key}: {e}')


output_vis_processor = transforms.Compose([
    transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])


def filter_dataset(item):
    if "jpg" not in item or "json" not in item:
        return False

    file_items = json.loads(item['json'].decode('utf-8'))
    original_width, original_height = file_items.get('original_width', 0), file_items.get('original_height', 0)
    if original_height < 128 or original_width < 128:
        return False

    try:
        item['pil'] = Image.open(io.BytesIO(item['jpg'])).convert("RGB")
    except Exception:
        return False 

    return True


def preprocess_dataset(item):
    output_dict = {}
    orig_image = item["pil"]

    file_items = json.loads(item['json'].decode('utf-8'))
    txt_key = '<DETAILED_CAPTION>' if '<DETAILED_CAPTION>' in file_items else 'caption'
    txt_content = file_items[txt_key].strip()
    
    max_len = 128
    txt_split = txt_content.split(" ")
    if len(txt_split) > max_len:
        print(f"text too long: {len(txt_split)}")
    txt_content = " ".join(txt_split[:max_len])
    
    if not txt_content:
        raise ValueError("Caption is empty after processing.")

    tensor_image = output_vis_processor(orig_image)

    output_dict["pixel_values"] = tensor_image
    output_dict["text"] = txt_content
    output_dict['url'] = item['__url__']
    output_dict['key'] = item['__key__']

    return output_dict


def collate_fn(batch):
    if not batch:
        return {"images": torch.empty(0), "prompts": [], "urls": [], "keys": []}

    pixel_values = torch.stack([item["pixel_values"] for item in batch], dim=0)
    urls = [item["url"] for item in batch]
    texts = [item['text'] for item in batch] 
    keys = [item["key"] for item in batch]
    
    return {
        "images": pixel_values, 
        "prompts": texts,
        "urls": urls,
        "keys": keys
    }


def load_laion_with_wds(
    data_dir,
    batch_size,
    num_workers: int = 4,
    target_json_key: str = "<DETAILED_CAPTION>",
):
    assert isinstance(data_dir, (str, list)), "data_dir must be a string or a list of strings."
    pattern = r".*\.tar$"
    
    if isinstance(data_dir, list):
        tar_files = []
        for path in data_dir:
            tar_files.extend(find_tar_files(path, pattern))
    else:
        tar_files = find_tar_files(data_dir, pattern)

    tar_files = sorted(tar_files)

    dataset = wds.DataPipeline(
        wds.SimpleShardList(tar_files),
        wds.tarfile_to_samples(),
        wds.shuffle(10000),
        wds.select(filter_dataset),
        wds.split_by_node,
        wds.map(preprocess_dataset),
        wds.batched(batch_size, collation_fn=collate_fn),
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
    )

    return dataloader
