import torch as T
from torch.utils.data import IterableDataset, DataLoader

class NoisyAudioDataset(IterableDataset):
    def __init__(self, audio_dataloader, noise_factor=0.3):
        self.audio_dataloader = audio_dataloader
        self.noise_factor = noise_factor

    def __iter__(self):
        for tensor in self.audio_dataloader:
            noisy_tensor = self.add_noise(tensor)
            yield noisy_tensor.unsqueeze(0), tensor.unsqueeze(0)

    def add_noise(self, tensor,
                    sp_ratio=0.1): 
        # get max value of tensor
        max_val = T.max(tensor)
        # get min value of tensor
        min_val = T.min(tensor)
        
        # Create salt-and-pepper noise
        salt_pepper_noise = T.rand(tensor.shape)
        
        # Add salt (1s) and pepper (0s) noise
        salt_pepper_noise = T.where(salt_pepper_noise < sp_ratio / 2, 
                                    T.zeros_like(tensor), salt_pepper_noise)
        salt_pepper_noise = T.where(salt_pepper_noise > 1 - sp_ratio / 2, 
                                    T.ones_like(tensor), salt_pepper_noise)
        
        # get range of tensor
        tensor_range = max_val - min_val
        # get noise
        
        # Create Gaussian noise
        gaussian_noise = T.randn_like(tensor) * self.noise_factor * 2

        gaussian_noise-= T.mean(gaussian_noise)
        # noise = T.randn_like(tensor) * self.noise_factor * tensor_range
        
        # Combine salt-and-pepper noise with Gaussian noise
        combined_noise = salt_pepper_noise * gaussian_noise
        
        return tensor + combined_noise
        # # center noise around 0
        # noise -= T.mean(noise)
        # return tensor + noise

class NoisyAudioDataLoader:
    def __init__(self, rank, world_size, slice_size, sample_rate, num_workers, log_folder, noise_factor=0.00001):
        self.rank = rank
        self.world_size = world_size
        self.slice_size = slice_size
        self.sample_rate = sample_rate
        self.num_workers = num_workers
        self.log_folder = log_folder
        self.noise_factor = noise_factor

    def load(self):
        from load_data.audio_dataloader import AudioDataLoader
        loader_obj = AudioDataLoader(
            rank=self.rank,
            world_size=self.world_size,
            slice_size=self.slice_size,
            sample_rate=self.sample_rate,
            num_workers=self.num_workers,
            log_folder=self.log_folder,
        )

        base_dataloader = loader_obj.load()
        noisy_dataset = NoisyAudioDataset(base_dataloader, self.noise_factor)

        return DataLoader(noisy_dataset, batch_size=None, num_workers=0)