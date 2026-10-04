from src.training import train
from src.training.train import Trainer

training = Trainer()
training.train(200)
training.val(50)
