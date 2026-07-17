from __future__ import print_function
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from tensorflow.keras import layers
from tensorflow.keras.layers import Dense

import tensorflow as tf
import random
import numpy as np
import sys
from hpca_loader import load_hpca_preprocessed

def get_cli_value(argv, key, default, cast):
    """
    Minimal parser for key=value command-line arguments.
    Example:
        python script.py seed=123 augment=False
    """
    if argv is None:
        argv = sys.argv[1:]

    prefix = key + "="

    for arg in argv:
        if arg.startswith(prefix):
            value = arg.split("=", 1)[1].strip()
            return cast(value)

    return default


def set_global_seed(seed: int, deterministic: bool = True):
    """
    Sets Python, NumPy, and TensorFlow seeds.
    """
    seed = int(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    if deterministic:
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception as exc:
            print(f"Could not enable TensorFlow op determinism: {exc}")

def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    if v in {"true", "1", "yes", "y"}:
        return True
    if v in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {v}")

def make_local_rf_mask_cifar(H, W, C, N_post, p, seed=0, num_pops=500, dtype=tf.float32):
    """
    Returns:
        M: (N_pre=H*W*C, N_post) connectivity mask
        pop_id: (N_post,) population ID per neuron

    Populations are assigned based on spatial + channel locality, not sequentially.
    """
    N_pre = H * W * C
    g = tf.random.Generator.from_seed(seed)
    
    # -----------------------------
    # Compute grid for population assignment
    # -----------------------------
    pops_per_channel = num_pops // C
    grid_size = int(tf.math.sqrt(tf.cast(pops_per_channel, tf.float32)))  # <-- cast to float
    grid_size = max(grid_size, 1)
    
    # 2D grid of top-left patch coordinates
    x_positions = tf.linspace(0., float(W - p), grid_size)
    y_positions = tf.linspace(0., float(H - p), grid_size)
    x_grid, y_grid = tf.meshgrid(x_positions, y_positions, indexing='ij')
    x_grid = tf.cast(tf.reshape(x_grid, [-1]), tf.int32)
    y_grid = tf.cast(tf.reshape(y_grid, [-1]), tf.int32)
    
    # -----------------------------
    # Assign neurons randomly to grid + channel
    # -----------------------------
    pop_id = tf.zeros([N_post], dtype=tf.int32)
    x0 = tf.zeros([N_post], dtype=tf.int32)
    y0 = tf.zeros([N_post], dtype=tf.int32)
    
    neurons_per_channel = N_post // C
    for c in range(C):
        start = c * neurons_per_channel
        end = (c+1) * neurons_per_channel if c < C-1 else N_post
        n_neurons = end - start
        
        pop_idx = g.uniform([n_neurons], minval=0, maxval=len(x_grid), dtype=tf.int32)
        
        x0 = tf.tensor_scatter_nd_update(x0, tf.range(start, end)[:, None], tf.gather(x_grid, pop_idx))
        y0 = tf.tensor_scatter_nd_update(y0, tf.range(start, end)[:, None], tf.gather(y_grid, pop_idx))
        pop_id = tf.tensor_scatter_nd_update(pop_id, tf.range(start, end)[:, None],
                                             pop_idx + c * len(x_grid))
    
    # -----------------------------
    # Build connectivity mask
    # -----------------------------
    dy, dx = tf.meshgrid(tf.range(p, dtype=tf.int32), tf.range(p, dtype=tf.int32), indexing='ij')
    xs = x0[:, None, None] + dx[None, :, :]
    ys = y0[:, None, None] + dy[None, :, :]
    
    cs = tf.range(C, dtype=tf.int32)[None, None, None, :]
    xs = xs[:, :, :, None]
    ys = ys[:, :, :, None]
    
    pre_idx = ((ys * W + xs) * C + cs)
    pre_idx = tf.reshape(pre_idx, [N_post, p*p*C])
    
    post_idx = tf.repeat(tf.range(N_post, dtype=tf.int32)[:, None], p*p*C, axis=1)
    ij = tf.stack([tf.reshape(pre_idx, [-1]), tf.reshape(post_idx, [-1])], axis=1)
    
    M = tf.scatter_nd(indices=ij, updates=tf.ones([N_post * p*p*C], dtype=dtype),
                      shape=[N_pre, N_post])
    
    return M, pop_id



class EnergyPooling(tf.keras.layers.Layer):
    def __init__(
        self,
        mode: str = "group_l2_concat",  # "square_concat" | "group_l2_concat" | "group_l2_only" | "square_only"
        group_size: int = 16,
        eps: float = 1e-8,
        use_layernorm: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mode = str(mode)
        self.group_size = int(group_size)
        self.eps = float(eps)
        self.use_layernorm = bool(use_layernorm)

        if self.mode not in {"square_concat", "group_l2_concat", "group_l2_only", "square_only"}:
            raise ValueError("mode must be one of: square_concat, group_l2_concat, group_l2_only, square_only")

        self._ln_sq = tf.keras.layers.LayerNormalization(name="ln_sq") if self.use_layernorm else None
        self._ln_g = tf.keras.layers.LayerNormalization(name="ln_g") if self.use_layernorm else None

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "mode": self.mode,
                "group_size": self.group_size,
                "eps": self.eps,
                "use_layernorm": self.use_layernorm,
            }
        )
        return cfg

    def call(self, h: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        h = tf.convert_to_tensor(h)
        if h.shape.rank != 2:
            raise ValueError("EnergyPooling expects a rank-2 tensor (batch, features).")

        if self.mode in {"square_concat", "square_only"}:
            h2 = tf.square(h)
            if self._ln_sq is not None:
                h2 = self._ln_sq(h2)
            return tf.concat([h, h2], axis=-1) if self.mode == "square_concat" else h2

        d = tf.shape(h)[-1]
        g = tf.constant(self.group_size, dtype=d.dtype)
        rem = tf.math.floormod(d, g)
        d_eff = d - rem
        h_eff = h[:, :d_eff]

        G = d_eff // g
        H = tf.reshape(h_eff, (-1, G, self.group_size))  # (B, G, g)

        e = tf.sqrt(tf.reduce_mean(tf.square(H), axis=-1) + self.eps)  # (B, G)
        if self._ln_g is not None:
            e = self._ln_g(e)

        return tf.concat([h, e], axis=-1) if self.mode == "group_l2_concat" else e

class IdentityLayer(tf.keras.layers.Layer):
    def call(self, x, training=False):
        return x
    
class DivisiveNormLayer(Dense):
    def __init__(self, units, pop_id, beta=1e-3, **kwargs):
        super(DivisiveNormLayer, self).__init__(units, **kwargs)
        self.pop_id = tf.constant(pop_id, dtype=tf.int32)  # (N_post,)
        self.beta = beta

    def call(self, inputs):
        # -------------------------------
        # Linear response
        # -------------------------------
        z = super(DivisiveNormLayer, self).call(inputs)   # (B, N_post)

        # Optional nonnegativity (firing rates ≥ 0)
        z = tf.abs(z)

        # -------------------------------
        # Population-based divisive normalization
        # -------------------------------
        z2 = tf.square(z)                                  # (B, N_post)
        z2_T = tf.transpose(z2)                            # (N_post, B)

        num_pops = tf.reduce_max(self.pop_id) + 1

        # SUM of squared activity per population (competition within RF)
        pop_energy = tf.math.unsorted_segment_sum(
            data=z2_T,
            segment_ids=self.pop_id,
            num_segments=num_pops
        )                                                   # (num_pops, B)

        # Map back to neurons
        denom = tf.gather(pop_energy, self.pop_id)          # (N_post, B)
        denom = tf.transpose(denom)                        # (B, N_post)

        # RMS-style normalization per population
        y = z / tf.sqrt(self.beta + denom)

        return y
    
    
class Net(tf.keras.Model):
    def __init__(
        self,
        pop_id,
        num_classes: int = 10,
        n_hidden: int = 20000,
        input_dropout: float = 0.1,
        head_dropout: float = 0.1,
        use_divisive_norm: bool = True,
        use_energy_pooling: bool = True,
        use_extended_readout: bool = True,
        head_units: int = 1024,
    ):
        super().__init__()

        self.use_divisive_norm = bool(use_divisive_norm)
        self.use_energy_pooling = bool(use_energy_pooling)
        self.use_extended_readout = bool(use_extended_readout)

        self.input_dropout = tf.keras.layers.Dropout(input_dropout)
        self.head_dropout = tf.keras.layers.Dropout(head_dropout)

        if self.use_divisive_norm:
            self.joint_fc = DivisiveNormLayer(
                units=n_hidden,
                use_bias=False,
                input_shape=(3072,),
                pop_id=pop_id,
                name="joint_fc",
            )
        else:
            self.joint_fc = layers.Dense(
                units=n_hidden,
                use_bias=False,
                input_shape=(3072,),
                name="joint_fc",
            )

        if self.use_energy_pooling:
            self.bottleneck = EnergyPooling(group_size=16, name="energy_pooling")
        else:
            self.bottleneck = IdentityLayer(name="no_energy_pooling")

        if self.use_extended_readout:
            self.head = layers.Dense(
                units=head_units,
                activation=tf.nn.gelu,
                use_bias=True,
                name="readout_head",
            )
        else:
            self.head = IdentityLayer(name="no_readout_head")

        self.classifier = layers.Dense(
            units=num_classes,
            activation=None,
            use_bias=True,
            name="classifier",
        )

    def call(self, inputs, training=False):
        if training:
            inputs = self.input_dropout(inputs, training=training)

        x = self.joint_fc(inputs)
        x = self.bottleneck(x, training=training)
        x = self.head(x, training=training)

        if training and self.use_extended_readout:
            x = self.head_dropout(x, training=training)

        return self.classifier(x)
        
class Local_Hebbian_Model:
    def __init__(
        self,
        *,
        pop_id,
        batch_size: int = 4000,
        eps: float = 5e-5,
        n_epochs: int = 1000000,
        n_hidden: int = 2000,
        num_classes: int = 10,
        input_dropout: float = 0.1,
        head_dropout: float = 0.1,
        update_step: int = 500,
        use_divisive_norm: bool = True,
        use_energy_pooling: bool = True,
        use_extended_readout: bool = True,
        head_units: int = 1024,
    ):
        self.pop_id = pop_id
        self.batch_size = int(batch_size)
        self.eps = float(eps)
        self.n_epochs = int(n_epochs)
        self.n_hidden = int(n_hidden)
        self.num_classes = int(num_classes)
        self.input_dropout = float(input_dropout)
        self.head_dropout = float(head_dropout)
        self.update_step = int(update_step)

        self.use_divisive_norm = bool(use_divisive_norm)
        self.use_energy_pooling = bool(use_energy_pooling)
        self.use_extended_readout = bool(use_extended_readout)
        self.head_units = int(head_units)

        self.model = None

    def initialize_model(self):
        self.model = Net(
            pop_id=self.pop_id,
            num_classes=self.num_classes,
            n_hidden=self.n_hidden,
            input_dropout=self.input_dropout,
            head_dropout=self.head_dropout,
            use_divisive_norm=self.use_divisive_norm,
            use_energy_pooling=self.use_energy_pooling,
            use_extended_readout=self.use_extended_readout,
            head_units=self.head_units,
        )

        _ = self.model(tf.zeros((1, 3072), dtype=tf.float32), training=False)

        return self.model
    
    
    def predict(self, x, model, hebbian=True, target=None, training=True, dropout_rate=0.1):
        if training and dropout_rate > 0.0:
            x = model.input_dropout(x, training=training)
    
        z = model.joint_fc(x)
        model.joint_fc.z = z
    
        if hebbian:
            self.hebbian_first_layer(model.joint_fc, inputs=x)
    
        z = model.bottleneck(z, training=training)
        z = model.head(z, training=training)
    
        if training and model.use_extended_readout:
            z = model.head_dropout(z, training=training)
    
        z = model.classifier(z)
        model.classifier.z = z
    
        return z
   
            
    def hebbian_first_layer(self, lay, inputs):
        if hasattr(lay, "s1"):
            del lay.s1
        if hasattr(lay, "s2"):
            del lay.s2
    
        W = lay.weights[0]
        D = W.shape[0]
        O = W.shape[1]
    
        s1 = tf.expand_dims(inputs, 2)  # (B, D, 1)
        s2 = tf.expand_dims(lay.z, 1)   # (B, 1, O)
    
        s1_mean = tf.reduce_mean(s1, axis=0)  # (D, 1)
        s2_mean = tf.reduce_mean(s2, axis=0)  # (1, O)
    
        lay.s1 = tf.broadcast_to(s1_mean, [D, O])
        lay.s2 = tf.broadcast_to(s2_mean, [D, O])
                  
        
def setup_model(pop_id, argv=None):
    par = {
    "seed": 10,
    "deterministic": True,

    "batch_size": 4000,
    "num_classes": 10,
    "n_hidden": int(len(pop_id)),
    "epochs": 20,
    "eps": 5e-5,
    "input_dropout": 0.1,
    "head_dropout": 0.1,
    "update_step": 500,

    "augment": True,
    "use_mask": True,
    "use_divisive_norm": True,
    "use_energy_pooling": True,
    "use_extended_readout": True,
    "head_units": 2048,
    }

    bool_keys = {
    "deterministic",
    "augment",
    "use_mask",
    "use_divisive_norm",
    "use_energy_pooling",
    "use_extended_readout",
    }

    if argv is None:
        argv = sys.argv[1:]

    for p in argv:
        parval = p.split("=")
        assert len(parval) == 2, "invalid parameter p=" + str(p)

        par_str = parval[0].strip()
        val_str = parval[1].strip()

        if par_str in par:
            if par_str in bool_keys:
                par[par_str] = str2bool(val_str)
            else:
                par[par_str] = type(par[par_str])(val_str)
        else:
            print("WARNING: Ignoring unknown parameter key " + par_str + "!!!")

    print("par =", par)

    model = Local_Hebbian_Model(
        pop_id=pop_id,
        batch_size=par["batch_size"],
        eps=par["eps"],
        n_epochs=par["epochs"],
        n_hidden=par["n_hidden"],
        num_classes=par["num_classes"],
        input_dropout=par["input_dropout"],
        head_dropout=par["head_dropout"],
        update_step=par["update_step"],
        use_divisive_norm=par["use_divisive_norm"],
        use_energy_pooling=par["use_energy_pooling"],
        use_extended_readout=par["use_extended_readout"],
        head_units=par["head_units"],
    )

    model.initialize_model()

    return model, par

# *****************************************************************
# *****************************************************************
if __name__ == '__main__':
# *****************************************************************
# *****************************************************************
    seed = get_cli_value(sys.argv[1:], "seed", 2, int)
    deterministic = get_cli_value(sys.argv[1:], "deterministic", True, str2bool)

    set_global_seed(seed, deterministic=deterministic)
    H, W, C = 32, 32, 3
    N_post = 20000
    p = 5
    num_pops = 1000

    
    # Temporarily create local mask first.
    M_local, pop_id_local = make_local_rf_mask_cifar(
        H, W, C, N_post, p, seed=seed, num_pops=num_pops
    )
    
    M_T = tf.transpose(M_local)
    unique_cols, pop_id_local = tf.raw_ops.UniqueV2(
        x=M_T,
        axis=[0],
    )
    
    # Parse parameters after pop_id exists.
    model, par = setup_model(pop_id=pop_id_local, argv=sys.argv[1:])
    
    if par["use_mask"]:
        M = M_local
        pop_id = pop_id_local
    else:
        M = tf.ones((H * W * C, N_post), dtype=tf.float32)
    
        # Dense baseline with divisive normalization:
        # all units belong to the same population.
        pop_id = tf.zeros((N_post,), dtype=tf.int32)

    # Recreate model because pop_id changed.
    model, par = setup_model(pop_id=pop_id, argv=sys.argv[1:])
    
    _ = model.model(tf.zeros((1, 3072), dtype=tf.float32), training=False)
    
    W0 = model.model.joint_fc.get_weights()[0]
    M_np = M.numpy().astype(np.float32)
    
    if W0.shape != M_np.shape:
        raise ValueError(f"Mask shape {M_np.shape} does not match W shape {W0.shape}")
    
    model.model.joint_fc.set_weights([(W0 * M_np).astype(np.float32)])
               
        
    augment_modes = ("orig", "flip", "rot", "shift") if par["augment"] else ("orig",)
    
    train_ds_list, num_batches_list, X_test2, T_test2 = load_hpca_preprocessed(
        dataset="cifar10",
        seed=par["seed"],
        augment=par["augment"],
        augment_modes=augment_modes,
        model_batch_size=par["batch_size"],
        make_stratified=True,
        stl_downsample_to_32=True,
    )
    tiny = 1e-9
    layers_list = [model.model.joint_fc]
    
    for epoch in range(model.n_epochs):
        print(epoch)
    
        if epoch % model.update_step == 0:
            if epoch == 40000:
                model.eps /= 2
            print("Epoch:", epoch)
    
        eps = model.eps
    
        for train_ds_mode, num_batches in zip(train_ds_list, num_batches_list):
            for step, (xb, tb) in enumerate(train_ds_mode.take(num_batches)):
    
                model.predict(
                    xb,
                    model.model,
                    hebbian=True,
                    training=True,
                    dropout_rate=0.0,
                )
    
                for lay in layers_list:
                    W = tf.identity(lay.weights[0])
                    s1 = tf.identity(lay.s1)
                    s2 = tf.identity(lay.s2)
    
                    M_cast = tf.cast(M, W.dtype)
    
                    proj = tf.reduce_sum(s1 * (W * M_cast), axis=0, keepdims=True)
                    diff = s2 - proj
                    deltaW = (s1 * diff) * M_cast
    
                    W_new = (W + eps * deltaW) * M_cast
    
                    lay.set_weights([W_new.numpy()])
    
 
# ---------------------------------------------------------------------
# Supervised fine-tuning after Hebbian feature learning
# ---------------------------------------------------------------------

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.003)
    
    # Freeze everything first
    for layer in model.model.layers:
        layer.trainable = False
    
    # Always keep the Hebbian feature layer frozen during supervised readout training
    model.model.joint_fc.trainable = False
    
    # EnergyPooling can contain LayerNorm parameters; freeze it for a pure frozen-feature readout
    model.model.bottleneck.trainable = False
    
    # Train supervised readout
    if par["use_extended_readout"]:
        model.model.head.trainable = True
    else:
        # head is IdentityLayer in this setting; no trainable variables expected
        model.model.head.trainable = False
    
    # Classifier is always trained
    model.model.classifier.trainable = True
    
    # Rebuild trainable variable list after setting layer.trainable
    train_vars = model.model.trainable_variables
    
    print("\nTrainable layers for supervised fine-tuning:")
    for layer in model.model.layers:
        print(
            layer.name,
            "trainable =", layer.trainable,
            "n_vars =", len(layer.trainable_variables),
        )
    
    print("Number of trainable variables:", len(train_vars))
    
    if len(train_vars) == 0:
        raise RuntimeError(
            "No trainable variables found. Check readout/head/classifier trainability."
        )
    
    model.model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["accuracy"],
    )
    
    n_finetune_epochs = 500
    
    for epoch in range(n_finetune_epochs):
        print(f"\nSupervised fine-tuning epoch {epoch + 1}/{n_finetune_epochs}")
    
        epoch_loss = tf.keras.metrics.Mean()
        epoch_acc = tf.keras.metrics.SparseCategoricalAccuracy()
    
        for ds_mode, num_batches in zip(train_ds_list, num_batches_list):
            for step, (xb, tb) in enumerate(ds_mode.take(num_batches)):
    
                # Labels may already be integer labels, but keep the one-hot fallback.
                if tb.shape.rank == 2:
                    tb_ids = tf.argmax(tb, axis=1, output_type=tf.int32)
                else:
                    tb_ids = tf.cast(tb, tf.int32)
    
                with tf.GradientTape() as tape:
                    logits = model.model(xb, training=True)
                    loss = loss_fn(tb_ids, logits)
    
                grads = tape.gradient(loss, train_vars)
    
                # Avoid crashes if any variable has no gradient.
                grads_and_vars = [
                    (g, v) for g, v in zip(grads, train_vars) if g is not None
                ]
    
                optimizer.apply_gradients(grads_and_vars)
    
                epoch_loss.update_state(loss)
                epoch_acc.update_state(tb_ids, logits)
    
        print(
            f"Train loss: {epoch_loss.result().numpy():.4f} | "
            f"Train acc: {epoch_acc.result().numpy():.4f}"
        )
    
        test_loss, test_acc = model.model.evaluate(
            X_test2,
            T_test2,
            batch_size=128,
            verbose=1,
        )
    
        print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}")
