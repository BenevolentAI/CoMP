from pytorch_lightning.callbacks import EarlyStopping, LambdaCallback, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.trainer import Trainer


def create_trainer(
    output_dir,
    num_epochs,
    gpus,
    checkpoint_metric_name,
    checkpoint_save_k=1,
    checkpoint_monitor_mode="min",
    early_stopping=False,
    early_stopping_delta=1e-4,
    early_stopping_patience=3,
    profiler=None,
    check_val_every_n_epoch=10,
    weights_summary="top",
):
    # Also return the checkpoint callback to allow retreival of best k models for calculating metrics
    checkpoint_callback = ModelCheckpoint(
        monitor=checkpoint_metric_name,
        dirpath=output_dir,
        # Interpolate the metric name into the str, e.g. '...{my_metric:.4e}'
        filename=f"{{epoch:04d}}-{{{checkpoint_metric_name:s}:.4e}}",
        save_top_k=checkpoint_save_k,
        mode=checkpoint_monitor_mode,
        period=check_val_every_n_epoch,
        save_last=False,
    )
    callbacks = [checkpoint_callback]
    if early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor=checkpoint_metric_name,
                min_delta=early_stopping_delta,
                patience=early_stopping_patience,
                verbose=True,
                mode=checkpoint_monitor_mode,
            )
        )

    if profiler is not None:
        callbacks.append(LambdaCallback(on_train_batch_end=lambda *_: profiler.step()))
    trainer = Trainer(
        max_epochs=num_epochs,
        logger=TensorBoardLogger(save_dir=output_dir),
        callbacks=callbacks,
        gpus=gpus,
        weights_summary=weights_summary,
        check_val_every_n_epoch=check_val_every_n_epoch,
    )
    return trainer, checkpoint_callback
