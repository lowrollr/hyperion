from train import mp_train
from test import mp_selfplay
import torch
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperion Chess Engine Training')
    parser.add_argument('--epoch_games', type=int, default=100, required=False)
    parser.add_argument('--testing_games', type=int, default=100, required=False)
    parser.add_argument('--testing_depth', type=int, default=200, required=False)
    parser.add_argument('--training_depth', type=int, default=200, required=False)
    parser.add_argument('--training_processes', type=int, default=2, required=False)
    parser.add_argument('--testing_processes', type=int, default=4, required=False)
    parser.add_argument('--num_gpus', type=int, default=1, required=False)
    args = parser.parse_args()
    torch.backends.cudnn.benchmark = True
    
    devices = [torch.device('cpu')]
    if torch.cuda.is_available():
        devices = []
        for i in range(torch.cuda.device_count()):
            devices.append(torch.device(f"cuda:{i}"))
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type(torch.FloatTensor)
    torch.set_default_dtype(torch.float)
    torch.multiprocessing.set_start_method('spawn')

    while True:
        trained_model = mp_train(devices, args.epoch_games, args.training_depth, args.training_processes)
        mp_selfplay(trained_model, devices, args.testing_games, args.testing_depth, args.testing_processes)