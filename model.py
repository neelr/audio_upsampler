import torch as t
import torch.nn as nn

class Upsampler(nn.Module):
    def __init__(self, input_channels=1, noised=False):
        super(Upsampler, self).__init__()

        self.noised = noised # true/false 
        
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=16, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(16, 32, kernel_size=16, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(32, 64, kernel_size=16, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(64, 128, kernel_size=16, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(128, 256, kernel_size=16, stride=2, padding=1),
            nn.LeakyReLU(0.1),
        )  

        self.middle = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 256),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=16, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose1d(128, 64, kernel_size=16, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose1d(64, 32, kernel_size=16, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose1d(32, 16, kernel_size=16, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose1d(16, input_channels, kernel_size=16, stride=2, padding=1),
            nn.LeakyReLU(0.1),
        )  
            
    def forward(self, x):
        # Ensure input is 3D: (batch_size, channels, time)
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Encoder
        x = self.encoder(x)
        
        # Middle (operating on each time step independently)
        batch_size, channels, time_steps = x.shape
        x = x.permute(0, 2, 1)  # (batch_size, time_steps, channels)
        x = self.middle(x)
        x = x.permute(0, 2, 1)  # (batch_size, channels, time_steps)
        
        # Decoder
        x = self.decoder(x)
        
        return x