from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

datamodule_params = {
    "system": "Lorenz",
    "n_samples": 250, #500,
    "n_timesteps": 50, #100,
    "pts_per_period": 25, #50,
    "seed": 0,
    "batch_size": 8, #32,
    "noise": 0.1,
    "make_data": False, #True,
    "save_data": False #True,
}

model_params = {
    "input_dim": 3,
    "hidden_size": 10,
    "output_dim": 3
}

model_ckpt_params = {
    "monitor": "val_loss",
    "mode": "min",
    "save_top_k": 1,
    "save_last": True,
    "verbose": False,
    "dirpath": "./chkpts",
    "auto_insert_metric_name": True,
}

callbacks = [] #[ModelCheckpoint(**model_ckpt_params)]

csv_logger_params = {"save_dir": ".", "version": "", "name": ""}
tensorboard_logger_params = {"save_dir": ".", "version": "", "name": ""}

loggers = [
    #CSVLogger(**csv_logger_params),
    #TensorBoardLogger(**tensorboard_logger_params), #turn on when ready to get final results
]

trainer_params = {"gradient_clip_val": 200, "max_epochs": 100, "log_every_n_steps": 5}