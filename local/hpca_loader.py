# Copyright 2026 Patrick Inoue
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from scipy import ndimage


def subtract_mean_and_normalize_h_numpy(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    x = x - x.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(np.square(x), axis=1, keepdims=True) + eps)
    return x / rms


def make_stratified_dataset_from_int_labels(X, T, batch_size, seed=0):
    num_classes = tf.reduce_max(T) + 1

    indices_per_class = [tf.where(T == c)[:, 0] for c in range(num_classes)]
    max_count = tf.reduce_max([tf.shape(idx)[0] for idx in indices_per_class])

    upsampled_indices = []
    for idx in indices_per_class:
        reps = tf.cast(tf.math.ceil(max_count / tf.shape(idx)[0]), tf.int32)
        idx_repeated = tf.tile(idx, [reps])
        idx_upsampled = idx_repeated[:max_count]
        upsampled_indices.append(idx_upsampled)

    all_indices = tf.concat(upsampled_indices, axis=0)
    all_indices = tf.random.shuffle(all_indices, seed=seed)

    X_balanced = tf.gather(X, all_indices)
    T_balanced = tf.gather(T, all_indices)

    train_ds = tf.data.Dataset.from_tensor_slices((X_balanced, T_balanced))
    train_ds = train_ds.shuffle(buffer_size=int(tf.shape(X_balanced)[0]), reshuffle_each_iteration=True)
    train_ds = train_ds.batch(batch_size, drop_remainder=True)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    num_batches = int(tf.shape(X_balanced)[0] // batch_size)

    return train_ds, num_batches    

class FeatureWhiten:
    def __init__(self) -> None:
        self.mean_ = None
        self.scale_ = None

    def fit(self, x: np.ndarray, eps: float = 1e-5) -> "FeatureWhiten":
        x = x.astype(np.float32, copy=False)
        self.mean_ = x.mean(axis=0, keepdims=True)
        var = x.var(axis=0, keepdims=True)
        self.scale_ = 1.0 / np.sqrt(var + eps)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("FeatureWhiten.transform called before fit().")
        x = x.astype(np.float32, copy=False)
        return ((x - self.mean_) * self.scale_).astype(np.float32, copy=False)


def unit_scale(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    l2_norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (l2_norms + eps)


def downsample_area_96_to_32_rgb(x: np.ndarray) -> np.ndarray:
    if x.ndim != 4 or tuple(x.shape[1:]) != (96, 96, 3):
        raise ValueError(f"Expected input shape (N, 96, 96, 3), got {x.shape}.")

    x = x.astype(np.float32, copy=False)
    n, h, w, c = x.shape

    return (
        x.reshape(n, 32, 3, 32, 3, c)
         .mean(axis=(2, 4))
         .astype(np.float32, copy=False)
    )


def load_image_dataset(
    dataset: str,
    *,
    stl_downsample_to_32: bool = True,
):
    dataset = dataset.lower()

    if dataset == "cifar10":
        (x_train_img, y_train), (x_test_img, y_test) = tf.keras.datasets.cifar10.load_data()

        y_train = y_train.squeeze().astype(np.int64)
        y_test = y_test.squeeze().astype(np.int64)

        image_shape = (32, 32, 3)
        return x_train_img, y_train, x_test_img, y_test, image_shape

    if dataset == "stl10":
        try:
            import tensorflow_datasets as tfds
        except ImportError as exc:
            raise ImportError(
                "STL-10 requires tensorflow-datasets. Install it with: "
                "pip install tensorflow-datasets"
            ) from exc

        train_ds = tfds.as_numpy(tfds.load("stl10", split="train", batch_size=-1))
        test_ds = tfds.as_numpy(tfds.load("stl10", split="test", batch_size=-1))

        x_train_img = train_ds["image"]
        x_test_img = test_ds["image"]

        y_train = train_ds["label"].astype(np.int64)
        y_test = test_ds["label"].astype(np.int64)

        if stl_downsample_to_32:
            x_train_img = downsample_area_96_to_32_rgb(x_train_img)
            x_test_img = downsample_area_96_to_32_rgb(x_test_img)
            image_shape = (32, 32, 3)
        else:
            image_shape = (96, 96, 3)

        return x_train_img, y_train, x_test_img, y_test, image_shape
    
    if dataset == "mnist":
        (x_train_img, y_train), (x_test_img, y_test) = tf.keras.datasets.mnist.load_data()
    
        # Keras MNIST gives (N, 28, 28); convert to (N, 28, 28, 1)
        x_train_img = x_train_img[..., None]
        x_test_img = x_test_img[..., None]
    
        y_train = y_train.astype(np.int64)
        y_test = y_test.astype(np.int64)
    
        image_shape = (28, 28, 1)
        return x_train_img, y_train, x_test_img, y_test, image_shape

    raise ValueError("dataset must be one of: 'mnist', 'cifar10', 'stl10'")


def spatial_augment_x(
    x: np.ndarray,
    *,
    image_shape: tuple[int, int, int],
    rng: np.random.RandomState,
    mode: str = "orig",
    rotate_kappa: float = 1.0,
    max_shift: int | None = None,
    order: int = 1,
) -> np.ndarray:
    h, w, c = image_shape
    d_expected = h * w * c

    if max_shift is None:
        max_shift = max(1, int(round(0.125 * min(h, w))))  # 4 for 32x32, 12 for 96x96

    if x.ndim == 2:
        n, d = x.shape
        if d != d_expected:
            raise ValueError(f"Expected flattened images with {d_expected} features, got {d}.")
        x_img = x.reshape(n, h, w, c)
        return_flat = True
    elif x.ndim == 4:
        n = x.shape[0]
        if tuple(x.shape[1:]) != image_shape:
            raise ValueError(f"Expected images with shape (N,{h},{w},{c}), got {x.shape}.")
        x_img = x
        return_flat = False
    else:
        raise ValueError(f"Expected x.ndim in {{2, 4}}, got {x.ndim}.")

    x_img = x_img.astype(np.float32, copy=False)

    if mode == "orig":
        out = x_img.copy()

    elif mode == "flip":
        out = x_img[:, :, ::-1, :].copy()

    elif mode == "rot":
        out = np.empty_like(x_img, dtype=np.float32)
        for i in range(n):
            angle = (rng.vonmises(0.0, rotate_kappa) / (4.0 * np.pi)) * 180.0
            out[i] = ndimage.rotate(
                x_img[i],
                angle,
                axes=(0, 1),
                reshape=False,
                mode="wrap",
                order=order,
            ).astype(np.float32, copy=False)

    elif mode == "shift":
        out = np.empty_like(x_img, dtype=np.float32)
        for i in range(n):
            dy = int(rng.randint(-max_shift, max_shift))
            dx = int(rng.randint(-max_shift, max_shift))
            out[i] = ndimage.shift(
                x_img[i],
                shift=(dy, dx, 0),
                mode="reflect",
                order=order,
            ).astype(np.float32, copy=False)

    else:
        raise ValueError("mode must be one of: orig, flip, rot, shift")

    if return_flat:
        return out.reshape(out.shape[0], -1).astype(np.float32, copy=False)

    return out.astype(np.float32, copy=False)


def load_hpca_preprocessed(
    *,
    dataset: str = "cifar10",
    seed: int = 123,
    augment: bool = True,
    augment_modes: tuple[str, ...] = ("orig", "flip", "rot", "shift"),
    model_batch_size: int = 4000,
    as_tensor: bool = True,
    make_stratified: bool = True,
    stl_downsample_to_32: bool = True,
):
    rng = np.random.RandomState(seed)

    x_train_img, y_train, x_test_img, y_test, image_shape = load_image_dataset(
        dataset,
        stl_downsample_to_32=stl_downsample_to_32,
    )

    x_train = x_train_img.astype(np.float32).reshape(len(x_train_img), -1) / 255.0
    x_test = x_test_img.astype(np.float32).reshape(len(x_test_img), -1) / 255.0

    print(f"\n... pre-processing {dataset.upper()}")
    print("Image shape:", image_shape)

    x_train = subtract_mean_and_normalize_h_numpy(x_train)
    x_test = subtract_mean_and_normalize_h_numpy(x_test)

    whitener = FeatureWhiten().fit(x_train)
    x_train = whitener.transform(x_train)
    x_test = whitener.transform(x_test)

    x_train = unit_scale(x_train)
    x_test = unit_scale(x_test)

    if augment:
        x_train_list = [
            unit_scale(
                spatial_augment_x(
                    x_train,
                    image_shape=image_shape,
                    rng=rng,
                    mode=mode,
                )
            )
            for mode in augment_modes
        ]
    else:
        x_train_list = [x_train]
        augment_modes = ("orig",)

    x_test = unit_scale(x_test)

    print("... done")
    print("Number of train augmentation blocks:", len(x_train_list))
    print("Each train block shape:", x_train_list[0].shape)
    print("y_train:", y_train.shape)
    print("X_test:", x_test.shape)
    print("y_test:", y_test.shape)

    if make_stratified:
        train_ds_list = []
        num_batches_list = []

        for i, x_aug in enumerate(x_train_list):
            print(f"Creating stratified dataset for augmentation mode {i}: {augment_modes[i]}")

            ds, n_batches = make_stratified_dataset_from_int_labels(
                X=tf.convert_to_tensor(x_aug, dtype=tf.float32),
                T=tf.convert_to_tensor(y_train, dtype=tf.int64),
                batch_size=model_batch_size,
                seed=seed + i,
            )

            train_ds_list.append(ds)
            num_batches_list.append(n_batches)

        X_test2 = tf.convert_to_tensor(x_test, dtype=tf.float32)
        T_test2 = tf.convert_to_tensor(y_test, dtype=tf.int64)

        return train_ds_list, num_batches_list, X_test2, T_test2

    if as_tensor:
        x_train_list = [
            tf.convert_to_tensor(x_aug, dtype=tf.float32)
            for x_aug in x_train_list
        ]
        x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
        y_train = tf.convert_to_tensor(y_train, dtype=tf.int64)
        y_test = tf.convert_to_tensor(y_test, dtype=tf.int64)

    return x_train_list, y_train, x_test, y_test
