import torch
from torchvision import datasets
from torchvision import transforms
from PIL import Image
import numpy as np
import copy
import os


AUGMENTATION_TRANSFORM_SIZE = 2


class CustomMNIST(datasets.MNIST):
    def __init__(self, root, use_augmentation=True, train=True, transform=None, target_transform=None, download=False):
        super(CustomMNIST, self).__init__(root, train, transform, target_transform, download)
        self.use_augmentation = use_augmentation
        self.transform = copy.deepcopy(transform)

        if self.transform is not None:
            if self.use_augmentation:
                for _ in range(AUGMENTATION_TRANSFORM_SIZE):
                    self.transform.transforms.pop(0)

    def __getitem__(self, index):
        img, target = super(CustomMNIST, self).__getitem__(index)

        if self.train:
            orig_img = self.train_data[index]
        else:
            orig_img = self.test_data[index]

        orig_img = Image.fromarray(orig_img.numpy(), mode='L')

        if self.transform is not None:
            orig_img = self.transform(orig_img)

        return orig_img, img, target


def get_loader(config):
    """Builds and returns Dataloader for MNIST and SVHN dataset."""

    transform_list = []

    if config.use_augmentation:
        transform_list.append(transforms.RandomHorizontalFlip())
        transform_list.append(transforms.RandomRotation(0.1))

    AUGMENTATION_TRANSFORM_SIZE = len(transform_list)

    transform_list.append(transforms.Scale(config.image_size))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    transform_test = transforms.Compose([
        transforms.Scale(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform_train = transforms.Compose(transform_list)

    mnist = CustomMNIST(root=config.mnist_path, use_augmentation=config.use_augmentation, download=True,
                        transform=transform_train, train=True)

    # TODO: I should custom this dataset also to get the orig image before augmentation
    mnist_m = datasets.ImageFolder(root=os.path.join(config.mnist_m_path, "train"), transform=transform_train)

    mnist_test = datasets.MNIST(root=config.mnist_path, download=True, transform=transform_test, train=False)
    mnist_m_test = datasets.ImageFolder(root=os.path.join(config.mnist_m_path, "test"), transform=transform_test)

    mnist_loader = torch.utils.data.DataLoader(dataset=mnist,
                                               batch_size=config.mnist_batch_size,
                                               shuffle=config.shuffle,
                                               num_workers=config.num_workers)

    mnist_m_loader = torch.utils.data.DataLoader(dataset=mnist_m,
                                                 batch_size=config.mnist_m_batch_size,
                                                 shuffle=config.shuffle,
                                                 num_workers=config.num_workers)

    mnist_test_loader = torch.utils.data.DataLoader(dataset=mnist_test,
                                                    batch_size=config.mnist_batch_size,
                                                    shuffle=False,
                                                    num_workers=config.num_workers)

    mnist_m_test_loader = torch.utils.data.DataLoader(dataset=mnist_m_test,
                                                      batch_size=config.mnist_m_batch_size,
                                                      shuffle=False,
                                                      num_workers=config.num_workers)

    return mnist_m_loader, mnist_loader, mnist_m_test_loader, mnist_test_loader
