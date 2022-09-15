from train import mp_train
from test import mp_selfplay
import torch


if __name__ == '__main__':

    torch.backends.cudnn.benchmark = True
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type(torch.FloatTensor)
    torch.set_default_dtype(torch.float)
    torch.multiprocessing.set_start_method('spawn')

    trained_model = mp_train([device])
    mp_selfplay(trained_model, [device])

        