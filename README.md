# Locality and Weight Sharing Shape Hebbian Principal-Component Learning in Biologically Constrained Visual Models

This repository contains the code used to evaluate Hebbian principal-component analysis (HPCA) across three visual-learning regimes:

1. fully connected HPCA;
2. locally connected, non-weight-shared HPCA; and
3. shared-kernel convolutional HPCA implemented in two established Hebbian-CNN benchmark codebases.

The experiments use MNIST, CIFAR-10, and STL-10. CIFAR-10 is the primary benchmark, while MNIST and STL-10 provide lower- and higher-complexity controls.

This repository is organized for anonymous peer review. Replace the archive and repository placeholders only after the review process permits de-anonymization.

## Repository structure

The repository contains three top-level experiment directories. The commands below assume the following directory names:

```text
.
├── README.md
├── requirements.txt
├── local/
│   ├── local_connected_hebbian_experiment.py
│   ├── hpca_loader.py
│   ├── configs/
│   ├── scripts/
│   └── results/
├── softhebb/
│   ├── README.md
│   ├── configs/
│   ├── scripts/
│   ├── modified_files/
│   └── results/
└── modular-hebbian-cnn/
    ├── README.md
    ├── configs/
    ├── scripts/
    ├── modified_files/
    └── results/
```

If the checked-out repository uses different folder names, only the paths in the command examples must be adjusted.

### `local/`: fully connected and locally connected HPCA

This directory contains the TensorFlow implementation developed for the present study. The same HPCA update is used for both architectural conditions:

- `use_mask=False`: fully connected HPCA baseline;
- `use_mask=True`: locally connected HPCA with fixed receptive-field support and position-specific weights.

The directory also contains the preprocessing pipeline for MNIST, CIFAR-10, and STL-10, together with the energy-pooling and supervised-readout ablations.

### `softhebb/`: original SoftHebb replication and HPCA substitution

This directory extends the public SoftHebb implementation:

- upstream repository: <https://github.com/NeuromorphicComputing/SoftHebb>
- reference condition: original SoftHebb learning rule and protocol;
- matched intervention: HPCA replaces the layer-local synaptic update while the surrounding architecture, competition, preprocessing, schedule, and supervised readout remain tied to the reference implementation.

The directory must retain the upstream license and attribution notices. It should also record the upstream commit, modified files, added HPCA mode, exact configuration files, random seeds, and executable commands used for the paper.

### `modular-hebbian-cnn/`: modular benchmark replication and HPCA substitution

This directory extends the modular Hebbian-CNN benchmark:

- upstream repository: <https://github.com/Julian-JN/Advancing-the-Biological-Plausibility-and-Efficacy-of-Hebbian-Convolutional-Neural-Networks>
- reference condition: the reproduced Hard-WTA BCM benchmark configuration;
- matched intervention: the same benchmark configuration with the local update mode replaced by HPCA.

As in the SoftHebb replication, architecture, competition, preprocessing, normalization, schedule, and readout settings should remain fixed within each matched comparison. The directory must document the upstream commit and all local modifications.

## Code availability statement

The source code and executable configuration files used in this study are available in this anonymized repository for peer review at **PLACEHOLDER URL**. The archived version corresponding to the reported experiments is identified by **PLACEHOLDER ARCHIVE DOI OR RELEASE TAG**.

The repository contains separate implementations and executable configurations for:

- fully connected HPCA;
- locally connected HPCA;
- original SoftHebb reference;
- original SoftHebb architecture with HPCA substitution;
- modular Hebbian-CNN Hard-WTA BCM reference; and
- modular Hebbian-CNN with HPCA substitution.

The two convolutional experiment folders are derived from the public upstream repositories listed above. Their local documentation identifies the upstream revision, modified files, added HPCA modes, configuration files, random seeds, and commands required to reproduce the reported conditions.

## Environment

The local fully connected and locally connected experiments use Python 3.10 and TensorFlow. Install the direct dependencies from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The minimal `requirements.txt` for the TensorFlow implementation is:

```text
numpy==1.23.5
scipy==1.12.0
tensorflow==2.12.0
tensorflow-datasets==4.9.7
```

`tensorflow-datasets` is required for STL-10. MNIST and CIFAR-10 are downloaded through the Keras dataset interface.

The two convolutional replication folders may require separate PyTorch environments matching their respective upstream implementations. Use the environment specification included in each replication folder rather than installing both benchmark stacks into the TensorFlow environment.

## Datasets

The experiments use the official training and test splits of:

| Dataset | Original image shape | Classes | Loader |
|---|---:|---:|---|
| MNIST | 28 x 28 x 1 | 10 | `tf.keras.datasets.mnist` |
| CIFAR-10 | 32 x 32 x 3 | 10 | `tf.keras.datasets.cifar10` |
| STL-10 | 96 x 96 x 3 | 10 | `tensorflow_datasets` |

For the local TensorFlow implementation, STL-10 is area-downsampled from 96 x 96 to 32 x 32 when `stl_downsample_to_32=True`.

No test labels are used for feature learning, preprocessing selection, hyperparameter selection, or early stopping. Test accuracy is evaluated only after the unsupervised feature-learning stage and supervised readout training.

## Local preprocessing pipeline

The preprocessing implemented in `local/hpca_loader.py` is fitted on the training data and then applied to the test data:

1. convert images to `float32`, flatten them, and scale pixel values by `1/255`;
2. subtract the mean of each sample;
3. divide each sample by its root-mean-square magnitude;
4. standardize each feature using training-set mean and variance;
5. normalize each sample to unit Euclidean norm;
6. optionally generate augmented training blocks; and
7. construct stratified, shuffled, batched `tf.data.Dataset` objects.

The current loader implements the following augmentation blocks:

```text
orig, flip, rot, shift
```

Augmentation is applied to training data only. Test data are never augmented.

## Training protocol

All fully connected and locally connected experiments use a two-stage protocol.

### Stage 1: unsupervised HPCA feature learning

Only the first feature layer, `joint_fc`, is updated. The update is computed manually from batch-averaged presynaptic and postsynaptic responses. For the locally connected condition, the fixed receptive-field mask is reapplied after every update, so all weights outside the assigned support remain zero.

Class labels are not used during this stage.

### Stage 2: supervised evaluation of the frozen representation

After HPCA learning:

- `joint_fc` is frozen;
- the energy-pooling layer is frozen;
- the optional nonlinear readout head is trained only when `use_extended_readout=True`; and
- the final classifier is always trained.

The supervised objective is sparse categorical cross-entropy. In the supplied local script, the readout is optimized with Adam at learning rate `0.003` for 500 epochs. The additional nonlinear head is a downstream decoder and does not update the frozen HPCA feature extractor.

## Local experiment controls

The local entry point uses `key=value` command-line arguments:

```bash
python local_connected_hebbian_experiment.py seed=10 augment=True use_mask=True
```

Boolean values accept `true/false`, `1/0`, `yes/no`, or `y/n`, without case sensitivity.

### Configurable parameters

| Parameter | Default | Role |
|---|---:|---|
| `seed` | `10` in `par` | Dataset shuffling, augmentation RNG, model initialization, and receptive-field assignment. Pass this explicitly for every reported run. |
| `deterministic` | `True` | Requests deterministic TensorFlow operations in addition to setting Python, NumPy, and TensorFlow seeds. |
| `batch_size` | `4000` | Batch size used by the stratified training datasets. |
| `num_classes` | `10` | Number of output classes. All three datasets use ten classes. |
| `n_hidden` | derived from `len(pop_id)` | Number of HPCA units. In the supplied script this is coupled to `N_post=20000`; do not change it independently of the mask construction. |
| `epochs` | `20` | Number of unsupervised HPCA passes over each training or augmentation block. |
| `eps` | `5e-5` | HPCA update step size. This is a learning-rate parameter, not merely a numerical epsilon. |
| `input_dropout` | `0.1` | Dropout applied at the model input when the Keras model is called with `training=True`. The manual HPCA call explicitly uses `dropout_rate=0.0`, so this parameter affects supervised readout training, not the supplied unsupervised update loop. |
| `head_dropout` | `0.1` | Dropout after the optional nonlinear readout head during supervised training. |
| `update_step` | `500` | Progress interval and legacy learning-rate update hook. With `epochs=20`, only the epoch-zero progress message is reached. |
| `augment` | `True` | Enables the `orig`, `flip`, `rot`, and `shift` training blocks. |
| `use_mask` | `True` | `True` gives locally connected HPCA; `False` gives the dense baseline. |
| `use_divisive_norm` | `True` | Enables absolute responses followed by population-wise divisive normalization. |
| `use_energy_pooling` | `True` | Concatenates grouped RMS energy features to the HPCA representation. |
| `use_extended_readout` | `True` | Enables one supervised GELU hidden layer before the classifier. `False` gives a linear classifier on the frozen representation. |
| `head_units` | `2048` | Width of the optional supervised GELU readout layer. |

### Fixed settings in the supplied local script

The following values are defined directly in the source and are not currently exposed as command-line arguments:

| Setting | Value |
|---|---:|
| HPCA units, `N_post` | `20000` |
| Receptive-field width, `p` | `5` |
| Initial requested population count | `1000` |
| Divisive-normalization stabilizer, `beta` | `1e-3` |
| Energy-pooling group size | `16` |
| Readout optimizer | Adam |
| Readout learning rate | `0.003` |
| Readout epochs | `500` |
| Test batch size | `128` |

Changing these values defines a different experimental configuration and should be recorded in a separate config or run script.

## Mapping paper conditions to local options

The following settings reproduce the architectural switches implemented by the local script. Use the exact seed list and any dataset-specific settings committed in `local/configs/` for the final repeated experiments.

| Condition | Core settings |
|---|---|
| Fully connected HPCA baseline | `use_mask=False` |
| Standard locally connected HPCA | `use_mask=True augment=True use_energy_pooling=True use_extended_readout=True` |
| No augmentation ablation | standard local settings plus `augment=False` |
| No energy-pooling ablation | standard local settings plus `use_energy_pooling=False` |
| Linear-readout ablation | standard local settings plus `use_extended_readout=False` |
| No divisive normalization | standard local settings plus `use_divisive_norm=False`; this is an implementation option and should only be reported if it belongs to the registered experiment matrix |
| Optimizer-switch variant | same frozen local HPCA representation and nonlinear readout, with the dedicated plateau-triggered Adam-to-SGD readout configuration supplied separately in `local/configs/` or `local/scripts/` |

The optimizer-switch condition is not implemented by the attached base entry point. It must be run through the dedicated script or configuration archived with the reported experiments.

## Running the local experiments

Run commands from the `local/` directory:

```bash
cd local
```

The examples below assume that the archived local entry point exposes the dataset as `dataset=mnist`, `dataset=cifar10`, or `dataset=stl10`. See **Dataset-selection requirement** below if the checked-in script still hardcodes CIFAR-10.

### Standard locally connected HPCA

```bash
python local_connected_hebbian_experiment.py \
  dataset=cifar10 \
  seed=SEED \
  deterministic=True \
  use_mask=True \
  augment=True \
  use_divisive_norm=True \
  use_energy_pooling=True \
  use_extended_readout=True \
  head_units=2048 \
  batch_size=4000 \
  epochs=20 \
  eps=5e-5
```

Replace `dataset=cifar10` with `dataset=mnist` or `dataset=stl10` for the corresponding dataset-specific run.

### Fully connected baseline

```bash
python local_connected_hebbian_experiment.py \
  dataset=cifar10 \
  seed=SEED \
  deterministic=True \
  use_mask=False \
  augment=True \
  use_divisive_norm=True \
  use_energy_pooling=True \
  use_extended_readout=True \
  head_units=2048 \
  batch_size=4000 \
  epochs=20 \
  eps=5e-5
```

### No augmentation

```bash
python local_connected_hebbian_experiment.py \
  dataset=cifar10 seed=SEED deterministic=True \
  use_mask=True augment=False \
  use_divisive_norm=True use_energy_pooling=True \
  use_extended_readout=True head_units=2048 \
  batch_size=4000 epochs=20 eps=5e-5
```

### No energy pooling

```bash
python local_connected_hebbian_experiment.py \
  dataset=cifar10 seed=SEED deterministic=True \
  use_mask=True augment=True \
  use_divisive_norm=True use_energy_pooling=False \
  use_extended_readout=True head_units=2048 \
  batch_size=4000 epochs=20 eps=5e-5
```

### Linear readout

```bash
python local_connected_hebbian_experiment.py \
  dataset=cifar10 seed=SEED deterministic=True \
  use_mask=True augment=True \
  use_divisive_norm=True use_energy_pooling=True \
  use_extended_readout=False \
  batch_size=4000 epochs=20 eps=5e-5
```

### Repeated runs

Do not invent or substitute seed values. Use the exact seed list archived with the experiment configuration. A generic shell loop is:

```bash
mkdir -p results/logs

while read -r seed; do
  python local_connected_hebbian_experiment.py \
    dataset=cifar10 \
    seed="${seed}" \
    deterministic=True \
    use_mask=True \
    augment=True \
    use_divisive_norm=True \
    use_energy_pooling=True \
    use_extended_readout=True \
    head_units=2048 \
    batch_size=4000 \
    epochs=20 \
    eps=5e-5 \
    2>&1 | tee "results/logs/cifar10_local_seed_${seed}.log"
done < configs/seeds.txt
```

Apply the same seed list to matched conditions whenever the experimental design requires paired initialization and data-order controls.

## Dataset-selection requirement

The supplied code snapshot contains a CIFAR-10-specific main block:

```python
H, W, C = 32, 32, 3
# ...
dataset="cifar10"
```

It also builds the model using an input dimensionality of 3072. Therefore, a command containing `dataset=mnist` does not work unless the archived release includes a dataset-aware entry point or a separate MNIST script.

Before journal archival, the `local/` folder must provide one of the following reproducible interfaces:

1. a single entry point that parses `dataset=...` and derives `H`, `W`, `C`, and `input_dim`; or
2. three explicitly named dataset-specific entry points or configs.

For a unified entry point, the essential dataset specification is:

```python
DATASET_SPECS = {
    "mnist":  {"H": 28, "W": 28, "C": 1, "downsample_stl": False},
    "cifar10": {"H": 32, "W": 32, "C": 3, "downsample_stl": True},
    "stl10":  {"H": 32, "W": 32, "C": 3, "downsample_stl": True},
}

dataset = get_cli_value(sys.argv[1:], "dataset", "cifar10", str).lower()
spec = DATASET_SPECS[dataset]
H, W, C = spec["H"], spec["W"], spec["C"]
input_dim = H * W * C
```

`input_dim` must then replace every hardcoded model-build dimension of `3072`. The `dataset` key should also be accepted by the main parameter parser so that it is not reported as unknown.

STL-10 can share the 3072-dimensional model when it is downsampled to 32 x 32. MNIST requires a 784-dimensional input layer.

## Running the SoftHebb replication folder

The `softhebb/` folder should provide two executable run scripts with all benchmark-specific arguments fixed in version-controlled configuration files:

```bash
cd softhebb

# Reproduced original SoftHebb reference
bash scripts/run_reference.sh

# Matched HPCA substitution
bash scripts/run_hpca.sh
```

The HPCA run must change only the layer-local learning-rule mode relative to the matched reference unless a separately identified preprocessing or schedule sensitivity condition is being reproduced.

For every run, the folder-level documentation should state:

- upstream repository URL and commit hash;
- local commit or archive identifier;
- modified source files;
- exact learning-rule mode;
- dataset and preprocessing protocol;
- augmentation protocol;
- layer-specific learning rates and schedule;
- readout configuration;
- seed list; and
- output path.

## Running the modular Hebbian-CNN replication folder

The `modular-hebbian-cnn/` folder should likewise expose the reference and matched HPCA runs:

```bash
cd modular-hebbian-cnn

# Reproduced Hard-WTA BCM reference
bash scripts/run_hard_wta_bcm.sh

# Matched HPCA substitution
bash scripts/run_hpca.sh
```

If the common HPCA preprocessing sensitivity condition is included, keep it in a separately named script or config, for example:

```bash
bash scripts/run_hpca_common_preprocessing.sh
```

Do not overwrite the original benchmark configuration. Reference, HPCA-substitution, and preprocessing-sensitivity runs should remain distinct and independently executable.

## Expected result organization

A recommended results layout is:

```text
results/
├── logs/
├── metrics/
├── checkpoints/
└── figures/
```

Each run should record at least:

```text
dataset
condition
seed
software environment
source commit
configuration file
final test accuracy
```

The supplied local script prints training loss, training accuracy, test loss, and test accuracy to standard output after every supervised epoch. It does not automatically save checkpoints or structured metrics. Redirect standard output with `tee`, or use the archived wrapper scripts to write machine-readable summaries.

Report repeated-run performance as mean and standard deviation over the exact archived seed list. Do not select epochs or configurations using the test set.

## Reference values reported in the manuscript

These values are included as end-to-end checks, not as tolerances for every hardware or software stack.

### Fully connected and locally connected HPCA

| Dataset or condition | Accuracy [%] |
|---|---:|
| CIFAR-10, fully connected HPCA | 52.72 +/- 0.64 |
| CIFAR-10, standard locally connected HPCA | 60.78 +/- 0.51 |
| CIFAR-10, optimizer-switch local variant | 61.54 +/- 0.26 |
| CIFAR-10, no augmentation | 56.87 +/- 0.49 |
| CIFAR-10, no energy pooling | 60.47 +/- 0.71 |
| CIFAR-10, linear readout | 56.65 +/- 0.49 |
| MNIST, fully connected HPCA | 96.96 +/- 0.21 |
| MNIST, locally connected HPCA | 98.42 +/- 0.11 |
| STL-10, fully connected HPCA | 41.67 +/- 0.57 |
| STL-10, locally connected HPCA | 49.70 +/- 0.23 |
| STL-10, optimizer-switch local variant | 50.51 +/- 0.40 |

### Shared-kernel HPCA substitutions

| Dataset or benchmark condition | Accuracy [%] |
|---|---:|
| CIFAR-10, modular benchmark HPCA, original preprocessing | 70.70 +/- 0.51 |
| CIFAR-10, modular benchmark HPCA, common HPCA preprocessing | 74.47 +/- 0.43 |
| CIFAR-10, original SoftHebb architecture with HPCA | 76.89 +/- 0.52 |
| MNIST, modular benchmark HPCA | 98.00 +/- 0.15 |
| MNIST, original SoftHebb architecture with HPCA | 98.82 +/- 0.19 |
| STL-10, modular benchmark HPCA | 66.94 +/- 0.57 |
| STL-10, original SoftHebb architecture with HPCA | 71.32 +/- 0.10 |

Small numerical deviations can arise from hardware kernels and framework behavior. Material discrepancies should first be investigated against the archived commit, environment, preprocessing path, seed list, and exact run configuration.

## Important reproducibility notes

Resolve or explicitly document the following points before creating the final archival release.

### Seed defaults

The supplied script currently uses two fallback values: the initial global-seed parser defaults to `2`, while the parameter dictionary defaults to `10`. Always pass `seed=...` explicitly. A final release should use a single source of truth.

### Hidden-width coupling

Although `n_hidden` is parsed from the command line, the receptive-field mask is constructed using the fixed `N_post=20000`. Changing `n_hidden` alone causes a mask/weight shape mismatch. Expose `N_post` as the controlling option or keep `n_hidden` fixed in the archived configs.

### Augmentation description

The attached loader implements `orig`, horizontal flip, rotation, and spatial shift. Ensure that the manuscript, configuration files, and archived code describe the same augmentation set. Any Gaussian-noise or input-masking augmentation condition must have a corresponding implementation and executable config if it is claimed as part of the reported local experiments.

### Optimizer-switch condition

The attached base script uses Adam throughout the supervised readout stage and does not implement plateau detection or an Adam-to-SGD switch. The optimizer-switch result therefore requires a separate archived script or configuration containing the exact plateau criterion, SGD learning rate, momentum, and switch behavior.

### Output persistence

The attached local entry point does not save a trained model, checkpoint, or structured metrics file. The archived run wrappers should preserve logs and final metrics, and should save checkpoints if checkpoint-based verification is intended.

### Computational resources

The default local model uses 20,000 hidden units and a large dense weight/mask representation. Memory use is substantial, particularly with a batch size of 4,000 and multiple intermediate tensors. Reducing the batch size can be useful for a smoke test, but it changes the minibatch statistics used by the HPCA update and is not a reproduction of the reported configuration unless explicitly validated.

## Reproduction checklist

Before running a reported condition, verify:

- the correct subfolder and upstream revision;
- the archived environment;
- the dataset and preprocessing path;
- the exact seed list;
- the connectivity regime;
- the learning-rule mode;
- augmentation settings;
- divisive normalization and energy pooling;
- readout architecture and optimizer schedule;
- output directory; and
- the source commit recorded with the result.

## License and upstream attribution

The local implementation should be distributed under the license stated in this repository. The `softhebb/` and `modular-hebbian-cnn/` directories remain subject to their respective upstream licenses and attribution requirements. Do not remove upstream copyright, license, or citation files from the replication folders.
