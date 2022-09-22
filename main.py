from train import mp_train
from test import mp_selfplay
import torch
import argparse

def memstats(devices):
    for device in devices:
        print(device, torch.cuda.memory_stats(device))

def training_loop(devices):
    while True:
        print("Starting training...")
        memstats(devices)
        trained_model, (avg_loss, avg_moves, avg_time) = mp_train(devices, args.epoch_games, args.training_depth, args.training_processes, args.num_epochs)
        print('Finished training epoch, beginning versus play...')
        memstats(devices)
        (new_wins, old_wins, draws, t_avg_moves) = mp_selfplay(trained_model, devices, args.testing_games, args.testing_depth, args.testing_processes)
        
        print('Finished versus play...')
        torch.cuda.empty_cache()
        memstats(devices)
        with open('resutls.csv', mode='a') as f:
            f.write(','.join([str(x) for x in [avg_loss, avg_moves, avg_time, new_wins, old_wins, draws, t_avg_moves]]) + '\n')



            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperion Chess Engine Training')
    parser.add_argument('--epoch_games', type=int, default=100, required=False)
    parser.add_argument('--testing_games', type=int, default=100, required=False)
    parser.add_argument('--testing_depth', type=int, default=200, required=False)
    parser.add_argument('--training_depth', type=int, default=200, required=False)
    parser.add_argument('--training_processes', type=int, default=2, required=False)
    parser.add_argument('--testing_processes', type=int, default=4, required=False)
    parser.add_argument('--num_gpus', type=int, default=1, required=False)
    parser.add_argument('--num_epochs', type=int, default=3, required=False)
    args = parser.parse_args()
    torch.backends.cudnn.benchmark = True
    
    devices = [torch.device('cpu')]
    if torch.cuda.is_available():
        devices = []
        for i in range(args.num_gpus):
            devices.append(torch.device(f"cuda:{i}"))
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type(torch.FloatTensor)
    torch.set_default_dtype(torch.float)
    torch.multiprocessing.set_start_method('spawn')
    training_loop(devices)
    



