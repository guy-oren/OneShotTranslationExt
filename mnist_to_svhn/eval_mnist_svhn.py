import torch
import os
import argparse
import sys
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision import transforms
from utee import misc
from svhn import model as svhn_model
from mnist import model as mnist_model
import torch.nn.functional as F
from PIL import Image
from skimage.io import imread
from tqdm import tqdm


MODELS_BASE_PATH = r"/home/goren/.torch/models/"


def mnist_loader(path):
    img = imread(path, as_grey=True)
    # to be consistent with PIL images
    img = Image.fromarray(img, mode="L")
    return img


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--svhn",  help="eval mnist to svhn", action='store_true')
    parser.add_argument("--mnist", help="eval svhn to mnist", action='store_true')
    parser.add_argument("--eval_folder", type=str, help="folder path to evaluate", required=True)

    args = parser.parse_args()

    if args.svhn and args.mnist:
        print("Run program only with one flag on")
        sys.exit(-1)

    if args.mnist:
        img_loader = mnist_loader
        model = mnist_model.mnist(pretrained=True)
        image_size = 28
    elif args.svhn:
        img_loader = default_loader
        model = svhn_model.svhn(n_channel=32, pretrained=True)
        image_size = 32
    else:
        print("Run program with --svhn OR --mnist flags")
        sys.exit(-1)

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    eval_dataset = ImageFolder(args.eval_folder, transform=transform, loader=img_loader)
    eval_dataloader = DataLoader(eval_dataset)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)

    correct = 0.
    for img, target in tqdm(eval_dataloader, total=len(eval_dataloader)):
        img = img.to(device)
        target = target.to(device)

        output = model(img)
        correct += torch.sum(torch.argmax(F.softmax(output, dim=0)) == target).item()

    print("Accuracy: ", correct / len(eval_dataloader) * 100)


if __name__ == "__main__":
    main()


