from load_data.audio_dataloader import AudioDataLoader
import torch as T
import torch.distributed as dist
import os

use_dataloader = True

dist.init_process_group('nccl')
cuID = int(os.environ['LOCAL_RANK'])
T.cuda.set_device(cuID)
rank = dist.get_rank()
world_size = dist.get_world_size()
local_ngpus = T.cuda.device_count()
print(f'Rank {rank} of {world_size}.')

T.backends.cuda.matmul.allow_tf32 = True
T.set_float32_matmul_precision('high')
T.backends.cuda.enable_flash_sdp(True)
T.manual_seed(42)

checkpoint_dir = "ckpt"

loader_obj = AudioDataLoader(
        rank=rank,
        world_size=world_size,
        slice_size=1000,
        sample_rate=16_000,
        #file_fetch_batch_size=4_000,
        num_workers=16,
        log_folder=checkpoint_dir,
    )

dataloader = loader_obj.load()

def add_noise(tensor, noise_factor=0.00001):
    noise = T.randn_like(tensor) * noise_factor
    return tensor + noise

# for tensor in dataloader:
#     print(tensor)
#     break

for tensor in dataloader:
 
    tensor = tensor.to(T.device(f"cuda:{cuID}" if T.cuda.is_available() else "cpu"))
    noisy_tensor = add_noise(tensor)
 
    print("Original tensor (first 10 elements):", tensor.flatten()[:100])
    print("Noisy tensor (first 10 elements):", noisy_tensor.flatten()[:100])
    
    break  

dist.destroy_process_group()
