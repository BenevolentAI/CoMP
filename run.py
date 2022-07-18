import argparse
import logging
import os
from pathlib import Path

import pytorch_lightning as pl
import s3fs
import torch
import yaml

from comp.data.loaders import prepare_training_data
from comp import metric_handlers
from comp.nn.utils import Encoder, GaussianDecoder, calc_input_dims
from comp.nn.config import ModelConfig, TrainConfig
from comp.pl.vae import VAE
from comp.pl.cvae import CVAE
from comp.pl.comp import COMP
from comp.pl.trvae import TrVAE
from comp.pl.trainer import create_trainer

logging.basicConfig(
    format="%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

s3_fs = s3fs.S3FileSystem()


def main(
    data_dir: str,
    dataset: str,
    model_config: ModelConfig,
    train_config: TrainConfig,
    use_cuda: bool,
    seed: float,
    output_dir: str,
    enable_profiler: bool = False,
):
    pl.seed_everything(seed)
    tensor_dataset, train_loader, val_loader, metadata_df = prepare_training_data(
        data_dir=data_dir,
        batch_size=train_config.batch_size,
        include_labels=True,
        use_cuda=use_cuda,
    )
    # Instantiate model
    encoder_input_dim, decoder_input_dim = calc_input_dims(tensor_dataset, model_config)
    gene_expression_dim = tensor_dataset.tensors[0].shape[1]

    if model_config.model == "trvae":
        return_hidden = True
    else:
        return_hidden = False

    LOGGER.info(
        f'Input tensor shapes {tensor_dataset.tensors[0].shape[0]} x ({", ".join(str(t.shape[1]) for t in tensor_dataset.tensors)})'
    )
    LOGGER.info(
        f"Encoder input dim {encoder_input_dim}; decoder input dim {decoder_input_dim}; decoder output dim {gene_expression_dim}"
    )

    encoder = Encoder(
        encoder_input_dim,
        model_config.latent_dim,
        model_config.hidden_dim,
        n_layers=model_config.num_layers,
        use_batchnorm=model_config.use_batchnorm,
        bandwidth=model_config.bandwidth,
    )

    decoder = GaussianDecoder(
        gene_expression_dim,
        decoder_input_dim,
        model_config.hidden_dim,
        model_config.num_layers,
        return_hidden=return_hidden,
        use_batchnorm=model_config.use_batchnorm,
    )
    baseline_dist = torch.distributions.Normal(
        loc=tensor_dataset.tensors[0].mean(), scale=1
    )

    with torch.no_grad():
        baseline_logprob = (
            baseline_dist.log_prob(torch.as_tensor(tensor_dataset.tensors[0]))
            .mean()
            .item()
        )
        LOGGER.info(
            "Baseline log prob (using independent marginals): %f", baseline_logprob
        )

    if model_config.model == "vae":
        model = VAE(
            encoder,
            decoder,
            model_config.latent_dim,
            learning_rate=train_config.learning_rate,
            gamma=train_config.gamma,
            beta=model_config.kl_beta,
        )
    elif model_config.model == "cvae":
        model = CVAE(
            encoder,
            decoder,
            model_config.latent_dim,
            penalty=model_config.cvae_penalty,
            penalty_scale=model_config.penalty_scale,
            learning_rate=train_config.learning_rate,
            gamma=train_config.gamma,
            beta=model_config.kl_beta,
        )
    elif model_config.model == "comp":
        model = COMP(
            encoder,
            decoder,
            model_config.latent_dim,
            penalty_scale=model_config.penalty_scale,
            learning_rate=train_config.learning_rate,
            gamma=train_config.gamma,
            beta=model_config.kl_beta,
        )
    elif model_config.model == "trvae":
        model = TrVAE(
            encoder,
            decoder,
            model_config.latent_dim,
            penalty_scale=model_config.penalty_scale,
            learning_rate=train_config.learning_rate,
            gamma=train_config.gamma,
            beta=model_config.kl_beta,
            penalise_z=model_config.penalise_z,
            rbf_version=model_config.rbf_version,
        )
    else:
        assert False, f"{model_config.model} is not handled when creating model"

    checkpoint_callback = None
    tb_dir = os.path.join(output_dir, "logs")
    gpu_arg = 1 if use_cuda else None
    trainer_args = dict(
        output_dir=tb_dir,
        num_epochs=train_config.num_epochs,
        gpus=gpu_arg,
        checkpoint_metric_name="valid_loss",
        checkpoint_monitor_mode="min",
        early_stopping=False,
        early_stopping_delta=1e-6,
        early_stopping_patience=50,
        weights_summary="full",
        check_val_every_n_epoch=train_config.check_val_every_n_epoch,
    )
    if enable_profiler:
        LOGGER.warning(
            f"Pytorch profiler enabled; writing TensorBoard logs to {str(tb_dir)}"
        )
        with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(tb_dir),
        ) as profiler:
            trainer, checkpoint_callback = create_trainer(
                **trainer_args, profiler=profiler
            )
            trainer.fit(model, train_loader, val_dataloaders=val_loader)
    else:
        trainer, checkpoint_callback = create_trainer(**trainer_args)
        trainer.fit(model, train_loader, val_dataloaders=val_loader)

    if checkpoint_callback is not None:
        model_list = [k for k in checkpoint_callback.best_k_models.keys()]
    else:
        LOGGER.info(
            "No checkpoint callback found, calculating metrics and results for current model instance instead."
        )
        model_list = [model]

    metric_handlers.calc_metrics(
        output_dir=output_dir,
        dataset=tensor_dataset,
        sample_metadata_df=metadata_df,
        models=model_list,
        dataset_name=dataset,
        model_type=model_config.model,
        load_model_fn=model.load_from_checkpoint,
        use_cuda=use_cuda,
    )


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--dataset", choices=["tumour_cl", "kang", "uci-income", "tech-batch"])

    # Model config
    parser.add_argument("--model", choices=["vae", "cvae", "comp", "trvae"])
    parser.add_argument("--hidden-dim", type=int, default=10)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--penalty-scale", type=float, default=1.0)
    parser.add_argument("--cvae-penalty", default=None)
    parser.add_argument(
        "--kl-beta",
        type=float,
        default=1.0,
        help="Beta-VAE scale factor for the KL term in the VAE ELBO",
    )
    parser.add_argument(
        "--use-batchnorm",
        type=int,
        default=0,
        help="Whether to use batchnorm in the decoder.",
    )
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=0.1,
        help="The constant value of the posterior Gaussian scale.",
    )
    parser.add_argument(
        "--penalise-z",
        type=int,
        default=0,
        help="Whether to penalise z. If False, penalise first hidden layer. Applicable to TrVAE",
    )
    parser.add_argument(
        "--rbf-version",
        type=int,
        default=0,  # This is multiscale version from TrVAE
        help="RBF kernel version. Applicable to TrVAE only. For versions, see the global variables in the modules.",
    )

    # Training config
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--check-val-every-n-epoch", type=int, default=1)

    parser.add_argument("--use-cuda", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="/tmp/comp")
    parser.add_argument(
        "--profiler",
        action="store_true",
        default=False,
        help="Enable Pytorch profiler, logging to TensorBoard (Lightning only).",
    )
    return parser


if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()

    output_dir = args.output_dir
    if args.output_dir[0:2] != "s3":
        Path(output_dir).mkdir(
            parents=True, exist_ok=True
        )  # if s3, assume folder already exists
        (Path(output_dir) / "latents").mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "umaps").mkdir(parents=True, exist_ok=True)

    config_path = os.path.join(output_dir, "config.yaml")
    if output_dir[0:2] == "s3":
        with s3_fs.open(config_path, "w") as fp:
            yaml.dump(vars(args), fp)
    else:
        with open(config_path, "w") as fp:
            yaml.dump(vars(args), fp)

    main(
        data_dir=args.data_dir,
        dataset=args.dataset,
        model_config=ModelConfig(
            model=args.model,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            penalty_scale=args.penalty_scale,
            cvae_penalty=args.cvae_penalty,
            kl_beta=args.kl_beta,
            use_batchnorm=bool(args.use_batchnorm),
            bandwidth=args.bandwidth,
            penalise_z=bool(args.penalise_z),
            rbf_version=args.rbf_version,
        ),
        train_config=TrainConfig(
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            gamma=1.0,
            check_val_every_n_epoch=args.check_val_every_n_epoch,
        ),
        use_cuda=args.use_cuda,
        seed=args.seed,
        output_dir=output_dir,
        enable_profiler=args.profiler,
    )
