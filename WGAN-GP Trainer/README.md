# WGAN-GP Trainer
Wasserstein GAN (WGAN), is a type of generative adversarial network 
that minimizes an approximation of the Earth-Mover's distance (EM) rather 
than the Jensen-Shannon divergence as in the original GAN formulation.

 WGAN-GP is a generative adversarial network that uses the Wasserstein loss 
 formulation **_plus_** a gradient norm penalty to achieve Lipschitz continuity.

### Cat faces
![created](Examples%20generated/cats1.png)


### Human faces
![created](Examples%20generated/celebs1.png)


## File Structure:
### Creating your data folder
The root folder is the hierarchically first one: **CatFaces**, it contains 
the class folders such as **cats**. Same logic applies for celeba dataset.

    .
    ├── data
    │   ├── CatFaces
    │   │   └── cats
    │   └── celeba_gan
    │       └── celeb

### Your models
**architectures** folder contains the LSUN architectures for images which are
64x64 and 128x128.

    ├── models
    │   └── architectures
    │       └── LSUN_WGAN_64.py

### Auto-generated training folders
Each time a new train is started, a new file structure as shown below is created 
to save created images for once in each 100th batch. These created images can be
tracked in Tensorboard, which also includes the losses of generator and critic 
(discriminator). Tensorboard logs are stored in **logs** folder, and at the end of the 
training, the model is saved in **model**. And **layout** folder contains the hyperparameters
that are used to train the desired model.
<br><br>
Each time a new training is started, a folder named **train{ID}** will be created. First training
always called **training0**, if there is a folder already named **training0**, then the ID will be 
incremented and **training1** will be created, and it will keep incrementing as the trainings go.
Note that if there is no **trains** folder, it will create it automatically.

<br><br>
Name of the training file can be given as a parameter with `--name`. As an example: 
`python train.py --name test_catfaces_64`.

    └── trains
        ├── test_catfaces_64
        │   ├── generated_grid_images
        │   ├── generated_grid_images_fixed
        │   ├── generated_images_fixed
        │   ├── layout
        │   ├── logs
        │   │   ├── criticLoss
        │   │   ├── fake_different
        │   │   ├── fake_fixed
        │   │   ├── fake_singular
        │   │   ├── genLoss
        │   │   └── real
        │   └── model
        └── test_celeba_128
            ├── generated_grid_images
            ├── generated_grid_images_fixed
            .
            .

To keep track of the losses and images while training, you need to specify the path where **logs** is kept,
and then go to the terminal and type: `tensorboard --logdir logs`.
<br>
<br>
### **_Example_**:
Create your data folder -> add your data -> tune hyperparameters -> start the training by:

    $ python train.py --name test_catfaces_64

Keep track of the logs:

    $ cd trains/test_catfaces_64
    $ tensorboard --logdir logs




