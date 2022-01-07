import os
import yaml
import torch
import torch.nn as nn
import torchinfo


def increment_trains(config, name='train'):
    """
    creates files for (x)th train:
    -> trains/{name}(x)/
    -> trains/{name}(x)/generated_images/..
    -> trains/{name}(x)/logs/..
    """
    # create trains/
    traindir = "trains"
    if not os.path.isdir(traindir):
        os.makedirs(traindir)

    # manage trains/train/
    folders_alike = []
    for folder in os.listdir(f'trains/'):
        if name in folder:
            folders_alike.append(folder)

    # folders = [folder if name in folder else None for folder in os.listdir(f'trains/')]
    i = [0 if len(folders_alike) == 0 else int(folder[folder.find(f'{name}') + len(f'{name}'):]) for folder in folders_alike]
    i = 0 if len(i) == 0 else max(i) + 1
    train_dir = f'trains/{name}{str(i)}'

    # manage trains/train/generated_images
    gen_img_dirs = [
        # 'generated_images',
        'generated_images_fixed',
        'generated_grid_images',
        'generated_grid_images_fixed'
    ]

    # the layout
    layout_dir = os.path.join(train_dir, 'layout')

    # model savedir
    modelsave_dir = os.path.join(train_dir, 'model')

    # create folders
    for gen_dir in gen_img_dirs:
        os.makedirs(os.path.join(train_dir, gen_dir))
    os.makedirs(layout_dir)
    os.makedirs(modelsave_dir)

    # save used configs
    with open(os.path.join(layout_dir, 'train.config.backup.yaml'), 'w') as cfg_write:
        with open(config, 'r') as cfg_read:
            cfg_data = yaml.safe_load(cfg_read)
        yaml.dump(cfg_data, cfg_write)

    return train_dir


def gradient_penalty(critic, real, fake, device="cpu"):
    """
    :param critic: the discriminator
    :param real: real image
    :param fake: fake image
    :param device: cpu or cuda
    :return:
    """
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # print(f'GRADIENTPENALTY fake: {fake.shape} real: {real.shape} alpha: {alpha.shape}')

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty



















