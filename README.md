# fontina
`fontina` is a PyTorch library that helps with training models for the task of Visual Font
Recognition and doing inference with them.

## Feature highlights

* [DeepFont-like network architecture](https://arxiv.org/pdf/1507.03196v1.pdf)
* Configuration-based synthetic dataset generation
* Configuration-based model training via [PyTorch Lightning](https://lightning.ai/pytorch-lightning)
* Supports training and inference on Linux, MacOS and Windows.

## Using `fontina`

### Installing the dependencies
Starting from a cloned repository directory:

```bash
# Create a virtual environment: this uses venv but any system
# would work!
python -m venv .venv

# Activate the virtual environment: this depends on the OS. See
# the two options below.
# .venv/Scripts/activate # Windows
source .venv/bin/activate # Linux

# Install the dependencies needed to use .
pip install .
```

> **Note**
Windows users **must** manually install the CUDA-based version of PyTorch, as pip will
only install the CPU version on this platform. See [PyTorch Get Started](https://pytorch.org/get-started/locally/)
for the specific command to ru, which should be something along the lines of `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117`.

### **(Optional)** - Installing development dependencies
The following dependencies are only needed to develop `fontina`.

```bash
# Install the developer dependencies.
pip install .[linting]

# Run linting and tests!
make lint
make test
```

### Generating a synthetic dataset
If needed, the model can be trained on synthetic data. `fontina` provides a synthetic
dataset generator that follows part of the recommendations from the [DeepFont paper](https://arxiv.org/pdf/1507.03196v1.pdf)
to make the synthetic data look closer to the real data. To use the generator:

1. Make a copy of `configs/sample.yaml`, e.g. `configs/mymodel.yaml`
2. Open `configs/mymodel.yaml` and tweak the `fonts` section:

```yaml
fonts:

  # ...

  # Force an uniform white background for the generated images.
  # backgrounds_path: "assets/backgrounds"

  samples_per_font: 1000

  # Fill in the paths of the fonts that need to be used to generate
  # the data.
  classes:
    - name: Test Font
      path: "assets/fonts/test/Test.ttf"
    - name: Other Test Font
      path: "assets/fonts/test2/Test2.ttf"
```

3. Run the generation:

```bash
python src/fontina/generate.py -c configs/mymodel.yaml -o outputs/font-images/mymodel
```

After this completes, there should be one directory per configured font in `outputs/font-images/mymodel`.

### Training
`fontina` currently only supports training a DeepFont-like architecture. The training process
has two major steps: unsupervised training of the stacked autoencoders and supervised training
of the full network.

Before starting, make a copy of `configs/sample.yaml`, e.g. `configs/mymodel.yaml` (or use the
existing one that was created for the dataset generation step).

#### Part 1 - Unsupervised training
1. Open `configs/mymodel.yaml` and tweak the `training` section:

```yaml
training:
  # Set this to True to train the stacked autoencoders.
  only_autoencoder: True

  # Don't use an existing checkpoint for the unsupervised training.
  # scae_checkpoint_file: "outputs/models/autoenc/good-checkpoint.ckpt"

  data_root: "outputs/font-images/mymodel"

  # The directory that will contain the model checkpoints.
  output_dir: "outputs/models/mymodel-scae"

  # The size of the batch to use for training.
  batch_size: 128

  # The initial learning rate to use for training.
  learning_rate: 0.01

  epochs: 20

  # Whether or not to use a fraction of the data to run a
  # test cycle on the trained model.
  run_test_cycle: True
```

2. Then run the training with:

```bash
python src/fontina/train.py -c configs/mymodel.yaml
```

#### Part 2 - Supervised training
1. Open `configs/mymodel.yaml` and tweak the `training` section:

```yaml
training:
  # This will freeze the stacked autoencoders.
  only_autoencoder: False

  # Pick the best checkpoint from the unsupervised training.
  scae_checkpoint_file: "outputs/models/mymodel-scae/good-checkpoint.ckpt"

  data_root: "outputs/font-images/mymodel"

  # The directory that will contain the model checkpoints.
  output_dir: "outputs/models/mymodel-full"

  # The size of the batch to use for training.
  batch_size: 128

  # The initial learning rate to use for training.
  learning_rate: 0.01

  epochs: 20

  # Whether or not to use a fraction of the data to run a
  # test cycle on the trained model.
  run_test_cycle: True
```

2. Then run the training with:

```bash
python src/fontina/train.py -c configs/mymodel.yaml
```

### **(Optional)** - Monitor performance using TensorBoard
`fontina` automatically captures the performances of the training runs in a [TensorBoard](https://www.tensorflow.org/tensorboard)-compatible
way. It should be possible to visualize the recorded data by pointing TensorBoard to the logs
directory as follows:

```bash
tensorboard --logdir=lightning_logs
```

### Inference
Once training is complete, the resulting model can be used to run inference.

```bash
python src/fontina/predict.py -w "outputs/models/mymodel-full/best_checkpoint.ckpt" -i "assets/images/test.png"
```
