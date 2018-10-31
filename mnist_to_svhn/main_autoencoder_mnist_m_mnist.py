import argparse
import os
from torch.backends import cudnn

from solver_autoencoder_mnist_m_mnist import Solver
from data_loader_mnist_m_mnist import get_loader


def str2bool(v):
    return v.lower() in ('true')


def main(config):
    mnist_m_loader, mnist_loader, mnist_m_test_loader, mnist_test_loader = get_loader(config)

    solver = Solver(config, mnist_m_loader, mnist_loader)
    cudnn.benchmark = True

    # create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.sample_path):
        os.makedirs(config.sample_path)

    if config.mode == 'train':
        solver.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--g_conv_dim', type=int, default=64)
    parser.add_argument('--d_conv_dim', type=int, default=64)
    parser.add_argument('--num_classes', type=int, default=10)

    # training hyper-parameters
    parser.add_argument('--train_iters', type=int, default=15000)
    parser.add_argument('--mnist_batch_size', type=int, default=64)
    parser.add_argument('--mnist_m_batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--kl_lambda', type=float, default=0.1)

    # misc
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_path', type=str, default='./models_autoencoder_mnist_m_mnist')
    parser.add_argument('--sample_path', type=str, default='./samples_autoencoder_mnist_m_mnist')
    parser.add_argument('--mnist_path', type=str, default='./mnist')
    parser.add_argument('--mnist_m_path', type=str, default='./mnist_m')
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=500)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--use_augmentation', required=True, type=str2bool)

    config = parser.parse_args()
    print(config)
    main(config)
