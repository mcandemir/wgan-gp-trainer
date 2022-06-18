import torch
import torchvision
from models.architectures.DeepConv_GAN_64 import Generator
import yaml
import matplotlib.pyplot as plt
import argparse

# setup configurations
with open('train.config.yaml', 'r') as cfg:
    cfg = yaml.safe_load(cfg)
    IMAGE_SIZE = cfg['IMAGE_SIZE']
    CHANNELS_IMG = cfg['CHANNELS_IMG']
    Z_DIM = cfg['Z_DIM']
    FEATURES_GEN = cfg['FEATURES_GEN']


def load_checkpoint_model(PATH):
    gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN)

    torch.load(PATH, map_location=torch.device('cpu'))
    gen.load_state_dict(torch.load(PATH, map_location=torch.device('cpu'))['state_dict'])
    gen.eval()
    return gen


def run(*args):
    while True:
        gen = load_checkpoint_model(args[0].path)

        # create data and test model
        z = torch.randn(1, Z_DIM, 1, 1)

        # create a data distribution
        images = gen(z)

        # convert to image
        img = torchvision.utils.make_grid(
            images[0], normalize=True
        )

        img = img.permute(1, 2, 0)
        img = img.detach().numpy()

        # position of the figure
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(950, 540, args[0].figsize, args[0].figsize)

        # plt.rcParams["figure.figsize"] = (args[0].figsize, args[0].figsize)
        plt.imshow(img)
        plt.show()



def arg_parser():
    """
    :return parsed arguments:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='model path')
    parser.add_argument('--figsize', type=int,
                        default=220, help='size of the figures (as pixels)')
    return parser.parse_args()


if __name__ == "__main__":
    opts = arg_parser()
    run(opts)
