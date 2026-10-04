from src.training import train
from src.training.train import Trainer
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

training = Trainer()
training.train(200)
