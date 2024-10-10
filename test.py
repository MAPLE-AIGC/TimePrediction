from dataloaders.laion import load_laion_with_wds
from tqdm import tqdm 
import pickle 

dl = load_laion_with_wds(
    data_folder='/storage/qiguojunLab/qiguojun/laion_aes_recap_full',
    sample_size=16,
    batch_size=64,
    num_workers=4
)
unique_prompt = set()
pbar = tqdm(total=280_0000)
data_num = 0
unique_num = 0
for batch in dl:
    unique_prompt.update(batch['prompts'])
    pbar.update(len(batch['prompts']))
    unique_num = len(unique_prompt)
    data_num += len(batch['prompts'])
    pbar.set_description(f'{unique_num}/{data_num}')
    if unique_num > 150_0000:
        break

# with open('/storage/qiguojunLab/huangzem/codes/LLaMADiffusion/prompts.pkl', 'wb') as f:
#     pickle.dump(unique_prompt, f)


# import pickle 

# with open('/storage/qiguojunLab/huangzem/codes/LLaMADiffusion/prompts.pkl', 'rb') as f:
#     unique_prompt = pickle.load(f)

with open('/storage/qiguojunLab/huangzem/codes/LLaMADiffusion/aesthetic-recap.txt', 'w') as f:
    f.write("\n".join(unique_prompt))