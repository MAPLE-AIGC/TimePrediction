import os
import tarfile
import json
from tqdm import tqdm
import io

def update_json_with_new_caption(old_json, new_json):
    old_data = json.loads(old_json)
    new_data = json.loads(new_json)
    
    if "<DETAILED_CAPTION>" in new_data:
        old_data["<DETAILED_CAPTION>"] = new_data["<DETAILED_CAPTION>"]
    
    return json.dumps(old_data, indent=4)

def process_tar_files(a_tar_path, b_tar_path, output_tar_path):
    with tarfile.open(a_tar_path, 'r') as a_tar, tarfile.open(b_tar_path, 'r') as b_tar:
        b_jsons = {os.path.basename(member.name): b_tar.extractfile(member).read().decode('utf-8') 
                   for member in b_tar if member.name.endswith('.json')}

        with tarfile.open(output_tar_path, 'w') as out_tar:
            for member in a_tar.getmembers():
                extracted_file = a_tar.extractfile(member)
                if member.name.endswith('.json') and os.path.basename(member.name) in b_jsons:
                    old_json = extracted_file.read().decode('utf-8')
                    new_json = b_jsons[os.path.basename(member.name)]
                    updated_json = update_json_with_new_caption(old_json, new_json)
                    info = tarfile.TarInfo(name=member.name)
                    info.size = len(updated_json)
                    out_tar.addfile(tarinfo=info, fileobj=io.BytesIO(updated_json.encode('utf-8')))
                else:
                    out_tar.addfile(member, fileobj=extracted_file)

def main(a_dir, b_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    a_tar_files = [f for f in os.listdir(a_dir) if f.endswith('.tar')]
    b_tar_files = [f for f in os.listdir(b_dir) if f.endswith('.tar')]

    for a_tar_file in tqdm(a_tar_files, desc="Processing tar files"):
        a_tar_path = os.path.join(a_dir, a_tar_file)
        b_tar_path = os.path.join(b_dir, a_tar_file)
        output_tar_path = os.path.join(output_dir, a_tar_file)
        
        if os.path.exists(b_tar_path):
            if os.path.exists(output_tar_path):
                print(f"skip: {output_tar_path}")
            else:
                try:
                    process_tar_files(a_tar_path, b_tar_path, output_tar_path)
                except Exception as e:
                    print(f"Error processing {a_tar_path}: {e}")
        else:
            with tarfile.open(a_tar_path, 'r') as a_tar, tarfile.open(output_tar_path, 'w') as out_tar:
                for member in a_tar.getmembers():
                    out_tar.addfile(member, fileobj=a_tar.extractfile(member))

if __name__ == "__main__":
    a_dir = "/qiguojun/laion-aesthetic"  # 替换为A文件夹的路径
    b_dir = "/qiguojun/dataset/recaption_florence/laion_aes/256_tar"  # 替换为B文件夹的路径
    output_dir = "/qiguojun/home/Dataset/laion_aes_recap_full"  # 替换为输出文件夹的路径

    main(a_dir, b_dir, output_dir)
