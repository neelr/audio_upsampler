from comet_ml import Experiment
import torch as t
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from noised_dataloader import NoisyAudioDataLoader
from model import Upsampler
import os
import time
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
clog = True and rank == 0
checkpoint_dir = 'ckpt/_tmp'

if clog:
    experiment = Experiment(
        api_key=os.environ['COMET_API_KEY'],
        project_name='upsampler',
        workspace='standard-log',
    )
    checkpoint_dir = 'ckpt/' + experiment.get_name()
# Training function
def train(model, dataloader, optimizer, criterion, device, scaler, epochs=10, batch_size=16):
    model.train()
    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0
        batch_idx = 0
        for (noisy, original) in dataloader:
            batch_idx += 1
            noisy, original = noisy.to(device), original.to(device)

            noisy = noisy.reshape(batch_size, 1, 4096*2)
            original = original.reshape(batch_size, 1, 4096*2)
            
            optimizer.zero_grad()
            
            with autocast():
                output = model(noisy)
                # cut off the output to match the original size
                original = original[:, :, :output.shape[2]]
                loss = T.sqrt(criterion(output, original))
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            experiment.log_metric('loss', loss.item(), step=batch_idx)
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss
        end_time = time.time()
        print(f'Epoch {epoch+1}/{epochs} completed. Average Loss: {avg_loss:.4f}, Time: {end_time-start_time:.2f}s')
        
        # Save checkpoint
        t.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, f'{checkpoint_dir}/checkpoint_epoch_{epoch+1}.pt')

# Main execution
if __name__ == "__main__":
    # Set up CUDA
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    learning_rate = 0.001
    epochs = 10
    checkpoint_dir = "ckpt"
    batch_size = 128
    # Create dataloader
    loader_obj = NoisyAudioDataLoader(
        rank=rank,
        world_size=world_size,
        slice_size=4096*2*batch_size,
        sample_rate=16_000,
        #file_fetch_batch_size=4_000,
        num_workers=16,
        log_folder=checkpoint_dir,
    )
    dataloader = loader_obj.load()

    # Create model, optimizer, and loss function
    model = Upsampler(noised=True).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[cuID],  find_unused_parameters=False)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Create GradScaler for mixed precision training
    scaler = GradScaler()

    # Train the model
    train(model, dataloader, optimizer, criterion, device, scaler, epochs, batch_size)

    # Clean up
    t.distributed.destroy_process_group()