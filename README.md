# Locality and Weight Sharing Shape Hebbian Principal-Component Learning in Biologically Constrained Visual Models

**Authors:** Patrick Inoue, Florian Rohrbein, and Andreas Knoblauch

This repository accompanies the manuscript *Locality and Weight Sharing Shape Hebbian Principal-Component Learning in Biologically Constrained Visual Models* and contains the code used to evaluate Hebbian principal-component analysis (HPCA) in three visual-learning regimes:

1. fully connected HPCA;
2. locally connected, non-weight-shared HPCA; and
3. shared-kernel convolutional HPCA in two established Hebbian-CNN benchmark implementations.

Experiments were conducted on MNIST, CIFAR-10, and STL-10. CIFAR-10 is the primary benchmark; MNIST and STL-10 provide lower- and higher-complexity controls.

## Repository structure

```text
.
├── README.md
├── THIRD_PARTY_NOTICES.md
├── requirements.txt
├── local/
│   ├── LICENSE
│   ├── local_connected_hebbian_experiment.py
│   └── hpca_loader.py
├── softhebb/
└── modular-hebbian-cnn/
```

### `local/`

TensorFlow implementation of the fully connected and locally connected HPCA experiments developed for this study. The same HPCA update is used in both conditions. The `use_mask` switch selects the connectivity regime:

- `use_mask=False`: fully connected HPCA;
- `use_mask=True`: locally connected HPCA with fixed receptive-field support and position-specific weights.

This directory also contains the preprocessing loader, data augmentation, population-wise divisive normalization, energy pooling, and supervised-readout variants.

### `softhebb/`

Replication of the original SoftHebb benchmark and the matched HPCA substitution. This code is based on the upstream repository:

<https://github.com/NeuromorphicComputing/SoftHebb>

The reference condition retains the original SoftHebb learning rule and protocol. In the matched HPCA condition, the layer-local update is replaced by HPCA while the surrounding architecture and benchmark protocol are retained.

### `modular-hebbian-cnn/`

Replication of the modular Hebbian-CNN Hard-WTA BCM benchmark and the matched HPCA substitution. This code is based on the upstream repository:

<https://github.com/Julian-JN/Advancing-the-Biological-Plausibility-and-Efficacy-of-Hebbian-Convolutional-Neural-Networks>

The reference condition uses the reproduced Hard-WTA BCM configuration. The matched condition replaces the local update mode with HPCA while retaining the remaining benchmark configuration.

## Environment

The fully connected and locally connected experiments use Python 3.10 and TensorFlow 2.12. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The root `requirements.txt` contains:

```text
numpy==1.23.5
scipy==1.12.0
tensorflow==2.12.0
tensorflow-datasets==4.9.7
```

`tensorflow-datasets` is used for STL-10. MNIST and CIFAR-10 are loaded through the TensorFlow/Keras dataset interfaces.

The two convolutional replication directories use their own PyTorch environments and benchmark-specific dependency files.

## Datasets and preprocessing

| Dataset | Input images | Classes | Loader |
|---|---:|---:|---|
| MNIST | 28 x 28 x 1 | 10 | `tf.keras.datasets.mnist` |
| CIFAR-10 | 32 x 32 x 3 | 10 | `tf.keras.datasets.cifar10` |
| STL-10 | 96 x 96 x 3 | 10 | `tensorflow_datasets` |

For the local TensorFlow experiments, STL-10 can be area-downsampled from 96 x 96 to 32 x 32 using `stl_downsample_to_32=True`.

The preprocessing pipeline in `local/hpca_loader.py` performs:

1. conversion to `float32`, flattening, and scaling by `1/255`;
2. per-sample mean subtraction;
3. per-sample RMS normalization;
4. feature-wise standardization using training-set statistics;
5. per-sample L2 normalization;
6. optional training augmentation; and
7. construction of stratified, shuffled, batched `tf.data.Dataset` objects.

The implemented augmentation blocks are:

```text
orig, flip, rot, shift
```

Augmentation is applied only to training data. Test data are not augmented.

The dataset name is selected in the call to `load_hpca_preprocessed(dataset=...)` in the experiment entry point. It is not a command-line parameter. The image geometry and model input dimensionality in the selected experiment script must correspond to the selected dataset.

## Training protocol

The fully connected and locally connected experiments use two stages.

### Stage 1: unsupervised HPCA feature learning

Only the first feature layer, `joint_fc`, is updated. The update is computed from batch-averaged presynaptic and postsynaptic responses. Class labels are not used. In the locally connected condition, the fixed connectivity mask is reapplied after every update so that weights outside the assigned receptive fields remain zero.

### Stage 2: supervised evaluation

After unsupervised feature learning, `joint_fc` and the optional energy-pooling layer are frozen. The final classifier is trained with sparse categorical cross-entropy. When `use_extended_readout=True`, an additional supervised GELU hidden layer is trained before the classifier. This downstream readout does not update the frozen HPCA feature extractor.

The local experiment script uses Adam with learning rate `0.003` for 500 supervised readout epochs.

## Experiment controls

The local entry point accepts `key=value` command-line arguments:

```bash
cd local
python local_connected_hebbian_experiment.py seed=10 augment=True use_mask=True
```

Boolean values accept `true/false`, `1/0`, `yes/no`, or `y/n`, without case sensitivity.

| Parameter | Default | Function |
|---|---:|---|
| `seed` | `10` | Controls data shuffling, augmentation, initialization, and receptive-field assignment. Pass it explicitly for every run. |
| `deterministic` | `True` | Requests deterministic TensorFlow operations and sets Python, NumPy, and TensorFlow seeds. |
| `batch_size` | `4000` | Training batch size. |
| `num_classes` | `10` | Number of output classes. |
| `n_hidden` | `len(pop_id)` | Number of HPCA units; it must remain consistent with the receptive-field mask. |
| `epochs` | `20` | Number of unsupervised HPCA training epochs. |
| `eps` | `5e-5` | HPCA update step size. |
| `input_dropout` | `0.1` | Input dropout used during supervised model calls. The unsupervised update loop passes `dropout_rate=0.0`. |
| `head_dropout` | `0.1` | Dropout after the optional nonlinear readout layer. |
| `update_step` | `500` | Interval used by the progress and learning-rate update hook. |
| `augment` | `True` | Enables the `orig`, `flip`, `rot`, and `shift` training blocks. |
| `use_mask` | `True` | Selects locally connected (`True`) or fully connected (`False`) HPCA. |
| `use_divisive_norm` | `True` | Enables absolute responses followed by divisive normalization. |
| `use_energy_pooling` | `True` | Concatenates grouped RMS energy features to the normalized HPCA representation. |
| `use_extended_readout` | `True` | Enables the supervised GELU readout layer before the classifier. |
| `head_units` | `2048` | Width of the optional supervised readout layer. |

With `use_mask=False`, the connection mask is dense. When divisive normalization remains enabled, the implementation assigns all hidden units to one response population.

## Running the local experiments

The commands below control the architectural conditions and ablations. Dataset selection remains source-level as described above.

### Locally connected HPCA

```bash
cd local
python local_connected_hebbian_experiment.py \
  seed=10 deterministic=True \
  batch_size=4000 epochs=20 eps=5e-5 \
  augment=True use_mask=True \
  use_divisive_norm=True \
  use_energy_pooling=True \
  use_extended_readout=True head_units=2048
```

### Fully connected HPCA

```bash
cd local
python local_connected_hebbian_experiment.py \
  seed=10 deterministic=True \
  batch_size=4000 epochs=20 eps=5e-5 \
  augment=True use_mask=False \
  use_divisive_norm=True \
  use_energy_pooling=True \
  use_extended_readout=True head_units=2048
```

### No data augmentation

```bash
cd local
python local_connected_hebbian_experiment.py \
  seed=10 deterministic=True \
  batch_size=4000 epochs=20 eps=5e-5 \
  augment=False use_mask=True \
  use_divisive_norm=True \
  use_energy_pooling=True \
  use_extended_readout=True head_units=2048
```

### No energy pooling

```bash
cd local
python local_connected_hebbian_experiment.py \
  seed=10 deterministic=True \
  batch_size=4000 epochs=20 eps=5e-5 \
  augment=True use_mask=True \
  use_divisive_norm=True \
  use_energy_pooling=False \
  use_extended_readout=True head_units=2048
```

### Linear readout

```bash
cd local
python local_connected_hebbian_experiment.py \
  seed=10 deterministic=True \
  batch_size=4000 epochs=20 eps=5e-5 \
  augment=True use_mask=True \
  use_divisive_norm=True \
  use_energy_pooling=True \
  use_extended_readout=False
```

Use the same command structure with the archived seed values for repeated runs.

## Convolutional benchmark replications

The `softhebb/` and `modular-hebbian-cnn/` directories preserve the respective benchmark implementations, the added HPCA modes, and the exact experiment configurations used for the reported reference and matched HPCA conditions. Each directory contains its own environment and execution documentation. Run convolutional experiments from within the corresponding directory so that its local imports and configuration paths are resolved correctly.

The matched comparisons change the local learning rule while retaining the corresponding benchmark architecture, competition mechanism, preprocessing, normalization, training schedule, and supervised-readout protocol. Separately identified preprocessing or schedule sensitivity experiments use distinct configurations.

## Output

The local script prints training loss, training accuracy, test loss, and test accuracy after each supervised epoch. Logs can be retained with:

```bash
python local_connected_hebbian_experiment.py seed=10 use_mask=True 2>&1 | tee local_seed10.log
```

Repeated-run results are reported as mean and standard deviation over the archived seed set. Test labels are not used during HPCA feature learning or preprocessing estimation.

## Citation

The convolutional replication folders build on the following publications and codebases. Cite the corresponding work when using either replication component.

### SoftHebb

Upstream repository: <https://github.com/NeuromorphicComputing/SoftHebb>

```bibtex
@inproceedings{journe2023hebbian,
  title     = {Hebbian Deep Learning Without Feedback},
  author    = {Journ{\'e}, Adrien and Garcia Rodriguez, Hector and Guo, Qinghai and Moraitis, Timoleon},
  booktitle = {International Conference on Learning Representations},
  year      = {2023},
  url       = {https://openreview.net/forum?id=8gd4M-_Rj1}
}
```

### Modular Hebbian-CNN benchmark

Upstream repository: <https://github.com/Julian-JN/Advancing-the-Biological-Plausibility-and-Efficacy-of-Hebbian-Convolutional-Neural-Networks>

```bibtex
@article{nimmo2025advancing,
  title   = {Advancing the Biological Plausibility and Efficacy of Hebbian Convolutional Neural Networks},
  author  = {Jim{\'e}nez Nimmo, Julian and Mondrag{\'o}n, Esther},
  journal = {Neural Networks},
  volume  = {190},
  pages   = {107628},
  year    = {2025},
  doi     = {10.1016/j.neunet.2025.107628}
}
```

## License and third-party code

The original TensorFlow implementation in `local/` is licensed under the Apache License 2.0; see `LICENSE`.

The licensing scope does not override third-party rights:

- `modular-hebbian-cnn/` is derived from an Apache-2.0-licensed upstream repository and retains its upstream license, copyright notices, and attribution requirements.
- The upstream SoftHebb repository does not provide an explicit open-source license file. The Apache-2.0 license in `local/` therefore does not apply to upstream SoftHebb code in `softhebb/`, and this repository grants no additional rights to that third-party code.

See `THIRD_PARTY_NOTICES.md` for provenance, modifications, and licensing details.
