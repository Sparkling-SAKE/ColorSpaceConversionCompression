# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
# from dataset import OpenImageLightning
# Change Vimeo dataset
from dataset import Vimeo90kSeptupletLightning
#from lightning_model import LightningModel
from lightning_model_DLEC import LightningModel
#from gained_mean_scale_hyperprior import GainedMeanScaleHyperprior
from gained_mean_scale_hyperprior_DLEC import GainedMeanScaleHyperprior
from collections import OrderedDict
import hydra
@hydra.main(config_path="config", config_name="base_v2")
def main(cfg: DictConfig):

    save_dir: Path = Path(hydra.utils.get_original_cwd()) / cfg.save_dir
        
    if (
        not cfg.overwrite
        and not cfg.resume_training
        and len(list(save_dir.glob("*.ckpt"))) > 0
    ):
        raise RuntimeError(
            "Checkpoints detected in save directory: set resume_training=True"
            " to restore trainer state from these checkpoints, or set overwrite=True"
            " to ignore them."
        )

    save_dir.mkdir(exist_ok=True, parents=True)
    # last_checkpoint = save_dir / "last.ckpt"
    last_checkpoint = save_dir / "last.ckpt"
        
    model = GainedMeanScaleHyperprior(**cfg.model)
    
    ##########################################################
    ##########################################################
    ######## Load Weights Partially ################################
    # last_checkpoint = "/home/oem/HD/GMSH/save/last_epoch411.ckpt"
    # pretrained_dict = torch.load(last_checkpoint)
    # model_dict = model.state_dict()
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)
    ##########################################################
    # last_checkpoint_ = '/home/oem/HD/MSH/save_q8/epoch=210-step=4615836.ckpt'
    # state_dict_ = torch.load(last_checkpoint_)
    # state_dict = state_dict_["state_dict"]
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k[6:]
    #     new_state_dict[name] = v
        
    # model_dict = model.state_dict()
    # model_dict.update(new_state_dict)
    # model.load_state_dict(model_dict)
    ##########################################################
    ##########################################################
    ##########################################################
    
    print("!@#$!@#$!@#$!@#$!@#$!@#$!@$!@$!@$@!$!@#$!@$!@#$!@$!@$@!$!@$!@$")
    print(cfg)
    print("!@#$!@#$!@#$!@#$!@#$!@#$!@$!@$!@$@!$!@#$!@$!@#$!@$!@$@!$!@$!@$")

    lightning_model = LightningModel(model, **cfg.training_loop)
    
    data = Vimeo90kSeptupletLightning(**cfg.data, pin_memory=cfg.ngpu != 0)

    loggers = [hydra.utils.instantiate(logger_cfg) for logger_cfg in cfg.loggers]
    trainer = Trainer(
        **cfg.trainer,
        logger=loggers,
        callbacks=[
            LearningRateMonitor(),
            ModelCheckpoint(**cfg.save_model),
        ],
        resume_from_checkpoint=last_checkpoint
        if last_checkpoint.exists() and cfg.resume_training
        else None,
    )

    trainer.fit(lightning_model, datamodule=data)


if __name__ == "__main__":
    main()
