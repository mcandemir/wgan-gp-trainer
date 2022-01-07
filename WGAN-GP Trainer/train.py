import os
import pathlib
import argparse
import torch
import yaml
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from utils import increment_trains, gradient_penalty
from models.architectures.LSUN_WGAN_64 import Discriminator, Generator, initialize_weights


ROOT = pathlib.Path(__file__).resolve().parents[0]
ROOT = pathlib.Path(os.path.relpath(ROOT, pathlib.Path.cwd()))  # relative path


# setup configurations
with open('train.config.yaml', 'r') as cfg:
    cfg = yaml.safe_load(cfg)
    NUM_EPOCHS = cfg['NUM_EPOCHS']
    LEARNING_RATE = float(cfg['LEARNING_RATE'])
    BATCH_SIZE = cfg['BATCH_SIZE']
    IMAGE_SIZE = cfg['IMAGE_SIZE']
    CHANNELS_IMG = cfg['CHANNELS_IMG']
    Z_DIM = cfg['Z_DIM']
    FEATURES_CRITIC = cfg['FEATURES_CRITIC']
    FEATURES_GEN = cfg['FEATURES_GEN']
    CRITIC_ITERATIONS = cfg['CRITIC_ITERATIONS']
    LAMBDA_GP = cfg['LAMBDA_GP']
    DATA_NAME = cfg['DATA_NAME']
    DEVICE = cfg['DEVICE']


def prepare_loader():
    """
    Do given transformations on the determined dataset
    and then return
    :return loader (DataLoader):
    """
    transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
            ),
    ])
    data = datasets.ImageFolder(root=os.path.join('data', DATA_NAME), transform=transform)

    loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
    return loader


def arg_parser():
    """
    :return parsed arguments:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--hyp', type=str, default=ROOT / 'train.config.yaml', help='hyperparameters path')
    parser.add_argument('--name', default='train', help='save to project/name')
    return parser.parse_args()


def train(loader, train_dir):

    gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(DEVICE)
    critic = Discriminator(CHANNELS_IMG, FEATURES_CRITIC).to(DEVICE)

    initialize_weights(gen)
    initialize_weights(critic)

    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
    opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

    fixed_noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1).to(DEVICE)

    # todo: might change this summary thing a bit
    writer_real = SummaryWriter(f'{train_dir}/logs/real')
    writer_fake_fixed = SummaryWriter(f'{train_dir}/logs/fake_fixed')
    writer_fake_different = SummaryWriter(f'{train_dir}/logs/fake_different')
    writer_fake_singular = SummaryWriter(f'{train_dir}/logs/fake_singular')
    writer_critic = SummaryWriter(f'{train_dir}/logs/original_criticLoss')
    writer_original_critic = SummaryWriter(f'{train_dir}/logs/criticLoss')
    writer_gen = SummaryWriter(f'{train_dir}/logs/genLoss')

    step = 0

    gen.train()
    critic.train()

    for epoch in range(NUM_EPOCHS):
        for batch_idx, (real, _) in enumerate(loader):
            real = real.to(DEVICE)
            cur_batch_size = real.shape[0]

            # train disc / critic
            for _ in range(CRITIC_ITERATIONS):
                noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(DEVICE)
                fake = gen(noise)

                critic_real = critic(real).reshape(-1)
                critic_fake = critic(fake).reshape(-1)

                # calculate gradient penalty ---------------------------------------
                gp = gradient_penalty(critic, real, fake, DEVICE)

                # calculate original loss and than apply gradient penalty
                original_loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake))
                )

                # write loss ----------------------------------------------------------------------
                writer_original_critic.add_scalar('CriticLoss/Epochs', original_loss_critic, global_step=step)

                # apply gradient penalty
                loss_critic = original_loss_critic + LAMBDA_GP * gp

                # write loss ----------------------------------------------------------------------
                writer_critic.add_scalar('CriticLoss_GP/Epochs', loss_critic, global_step=step)

                # set gradients to 0
                critic.zero_grad()
                loss_critic.backward(retain_graph=True)

                # update grads
                opt_critic.step()

                # Train gen ====================================================================
                output = critic(fake).reshape(-1)
                loss_gen = -torch.mean(output)

                # write loss ----------------------------------------------------------------------
                writer_gen.add_scalar('GeneratorLoss/Epochs', loss_gen, global_step=step)

                gen.zero_grad()
                loss_gen.backward()
                opt_gen.step()

                # print on tensorboard
                # Print on tensorboard
                if batch_idx % 100 == 0:
                    print(f'Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)}'
                          f'Critic Loss: {loss_critic}')

                    with torch.no_grad():
                        fake_fixed = gen(fixed_noise)
                        fake_different = gen(noise)

                        # take out (up to) 32 examples
                        GRID_SIZE = 32 if BATCH_SIZE > 32 else BATCH_SIZE

                        img_grid_real = torchvision.utils.make_grid(
                            real[:GRID_SIZE], normalize=True
                        )
                        img_grid_fake_fixed = torchvision.utils.make_grid(
                            fake_fixed[:GRID_SIZE], normalize=True
                        )
                        img_grid_fake_different = torchvision.utils.make_grid(
                            fake_different[:GRID_SIZE], normalize=True
                        )
                        img_grid_fake_singular = torchvision.utils.make_grid(
                            fake_different[0], normalize=True
                        )

                        writer_real.add_image('Real', img_grid_real, global_step=step)
                        writer_fake_fixed.add_image('Fake_fixed', img_grid_fake_fixed, global_step=step)
                        writer_fake_different.add_image('Fake_different', img_grid_fake_different, global_step=step)
                        writer_fake_singular.add_image('Fake_singular', img_grid_fake_singular, global_step=step)

                        save_image(img_grid_fake_fixed,
                                   f'{train_dir}/generated_grid_images_fixed/generated_img{epoch}_{batch_idx}.png')
                        save_image(img_grid_fake_different,
                                   f'{train_dir}/generated_grid_images/generated_img{epoch}_{batch_idx}.png')
                        save_image(img_grid_fake_singular,
                                   f'{train_dir}/generated_images_fixed/generated_img{epoch}_{batch_idx}.png')

                    step += 1

    torch.save(gen.state_dict(), f'{train_dir}/model/model.pth.tar')


def main(*args):
    """
    :param args:
    """
    train_dir = increment_trains(config=args[0].hyp, name=args[0].name)
    loader = prepare_loader()
    train(loader, train_dir=train_dir)


if __name__ == '__main__':
    opts = arg_parser()
    main(opts)










