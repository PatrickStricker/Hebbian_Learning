import argparse
import warnings

from utils import (
    load_presets,
    get_device,
    load_config_dataset,
    seed_init_fn,
    str2bool,
)
from model import load_layers
from train import run_sup, run_unsup, check_dimension, training_config, run_hybrid
from log import Log, save_logs

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description="Ray-free CIFAR10 SoftHebb4 run")

parser.add_argument("--preset", default="4SoftHebbCnnCIFAR", type=str)
parser.add_argument("--dataset-unsup", default="CIFAR10_1", type=str)
parser.add_argument("--dataset-sup", default="CIFAR10_50", type=str)
parser.add_argument(
    "--training-mode",
    choices=["successive", "consecutive", "simultaneous"],
    default="successive",
    type=str,
)
parser.add_argument("--resume", choices=[None, "all", "without_classifier"], default=None)
parser.add_argument("--model-name", default="CIFAR10_SoftHebb4_seed0", type=str)
parser.add_argument("--training-blocks", default=None, nargs="+", type=int)
parser.add_argument("--seed-unsup", default=0, type=int)
parser.add_argument("--seed-sup", default=None, type=int)
parser.add_argument("--gpu-id", default=0, type=int)
parser.add_argument("--save", default=True, type=str2bool)
parser.add_argument("--validation-sup", default=False, type=str2bool)
parser.add_argument("--validation-unsup", default=False, type=str2bool)


def main(args):
    name_model = args.preset if args.model_name is None else args.model_name

    blocks = load_presets(args.preset)

    dataset_sup_config = load_config_dataset(args.dataset_sup, args.validation_sup)
    dataset_unsup_config = load_config_dataset(args.dataset_unsup, args.validation_unsup)

    dataset_unsup_config["seed"] = args.seed_unsup

    if args.seed_sup is not None:
        dataset_sup_config["seed"] = args.seed_sup

    if dataset_unsup_config["seed"] is not None:
        seed_init_fn(dataset_unsup_config["seed"])

    device = get_device(args.gpu_id)

    blocks = check_dimension(blocks, dataset_sup_config)

    train_config = training_config(
        blocks,
        dataset_sup_config,
        dataset_unsup_config,
        args.training_mode,
        args.training_blocks,
    )

    model = load_layers(blocks, name_model, args.resume)
    model.reset()
    model = model.to(device)

    log = Log(train_config)

    for train_id, config in train_config.items():
        if config["mode"] == "unsupervised":
            run_unsup(
                config["nb_epoch"],
                config["print_freq"],
                config["batch_size"],
                name_model,
                dataset_unsup_config,
                model,
                device,
                log.unsup[train_id],
                blocks=config["blocks"],
                save=args.save,
                reset=False,
            )

        elif config["mode"] == "supervised":
            run_sup(
                config["nb_epoch"],
                config["print_freq"],
                config["batch_size"],
                config["lr"],
                name_model,
                dataset_sup_config,
                model,
                device,
                log.sup[train_id],
                blocks=config["blocks"],
                save=args.save,
            )

        else:
            run_hybrid(
                config["nb_epoch"],
                config["print_freq"],
                config["batch_size"],
                config["lr"],
                name_model,
                dataset_sup_config,
                model,
                device,
                log.sup[train_id],
                blocks=config["blocks"],
                save=args.save,
            )

    save_logs(log, name_model)


if __name__ == "__main__":
    main(parser.parse_args())