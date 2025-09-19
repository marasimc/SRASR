r""" 
This file is copied from symbolicregression/train.py.
"""

import argparse
import logging
import math
import os
import random
import sys

import sys
sys.path.append('../../.')
# print(sys.modules)

import numpy as np
import torch
from fairseq import (
    checkpoint_utils,
    distributed_utils,
    options,
    quantization_utils,
    tasks,
    utils,
)
from fairseq.logging import meters, metrics, progress_bar
import time

## symbolic regression related code start >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
import json
from pathlib import Path
import symbolicregression
from symbolicregression.slurm import init_signal_handler, init_distributed_mode
from symbolicregression.envs import build_env
from symbolicregression.model import build_modules
from symbolicregression.model import check_model_params, build_modules
from symbolicregression.parsers import get_parser
from symbolicregression.trainer import Trainer
from symbolicregression.evaluate import evaluate_pmlb, evaluate_in_domain

from knnbox.sr_models.enhance_knn_sr import SR_knnModel
## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< end

knn_model_type = 'enhance'
BATCH_SIZE = 64  #8 #16   # NOTE: change batch size here
MAX_EPOCH = 50          
STEPS = 500               # NOTE: change steps per epoch here, default = 3000
SAVE_PERIODIC = 1         # NOTE: change save period here, default = 25

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.train")


## symbolic regression related code start >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def get_params():
    parser = get_parser()
    params = parser.parse_args([])
    
    # check parameters
    check_model_params(params)
    
    return params
## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< end

def main(args, model_path='../../symbolicregression_utils/weights/model1.pt'):
    utils.import_user_module(args)

    assert (
        args.max_tokens is not None or args.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"

    metrics.reset()

    np.random.seed(args.seed)
    utils.set_torch_seed(args.seed)

    if distributed_utils.is_master(args):
        checkpoint_utils.verify_checkpoint_directory(args.save_dir)

    # Print args
    logger.info(args)

    sr_params = get_params()
    init_distributed_mode(sr_params)
    if sr_params.is_slurm_job:
        init_signal_handler()
    
    # CPU / CUDA
    if not sr_params.cpu:
        assert torch.cuda.is_available()
    symbolicregression.utils.CUDA = not sr_params.cpu

    # init environment
    ## parameter config start >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    file_postfix = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    checkpoint_path = 'checkpoints_'+file_postfix
    sr_params.dump_path = checkpoint_path
    # sr_params.export_data = True
    sr_params.batch_size = BATCH_SIZE               
    sr_params.n_steps_per_epoch = STEPS         
    sr_params.max_epoch = MAX_EPOCH        
    sr_params.eval_size = 150        
    sr_params.label_smoothing = args.label_smoothing
    sr_params.eval_in_domain = True
    sr_params.save_periodic = SAVE_PERIODIC      
    ## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    
    # build environment / modules / trainer / evaluator
    if sr_params.batch_size_eval is None:
        sr_params.batch_size_eval = int(1.5 * sr_params.batch_size)
    # if sr_params.eval_dump_path is None:
    #     sr_params.eval_dump_path = Path(sr_params.dump_path) / "evals_all"
    #     if not os.path.isdir(sr_params.eval_dump_path):
    #         os.makedirs(sr_params.eval_dump_path)
    
    env = build_env(sr_params)
    # print('equation_word2id["<PAD>"]: ', env.equation_word2id["<PAD>"])     # 82
    # print('float_word2id["<PAD>"]: ', env.float_word2id["<PAD>"])           # 8
    modules = build_modules(env, sr_params)     # can not be deleted, because it will define get_length_after_batching parameter in env
    
    if knn_model_type == 'enhance':
        print('################# enhanced adaptive knn sr model ##########')
        knn_model = SR_knnModel(args, env.equation_id2word).cuda()
    else:
        raise NotImplementedError
    trainer = Trainer(modules, env, sr_params, path=model_path, root='', knn_model=knn_model)     # define trainer and reload model from model_path
    logger.info("Trainer initialized with model path: {}".format(model_path))
    
    # disable grad of origin model (including embedder, encoder, decoder)
    for model in trainer.modules.values():
        for name, param in model.named_parameters():
            param.requires_grad = False 
    
    # start training
    logger.info(
        "training on {} devices (GPUs/TPUs)".format(args.distributed_world_size)
    )
    trainer.n_equations = 0
    for _ in range(sr_params.max_epoch):
    
        logger.info("============ Starting epoch %i ... ============" % trainer.epoch)

        knn_model.train()
        trainer.inner_epoch = 0
        while trainer.inner_epoch < trainer.n_steps_per_epoch:
            # training steps
            for task_id in np.random.permutation(len(sr_params.tasks)):
                task = sr_params.tasks[task_id]
                if sr_params.export_data:
                    trainer.export_data(task)
                else:
                    trainer.enc_dec_step(task)
                trainer.iter()

        logger.info("============ End of epoch %i ============" % trainer.epoch)
        if sr_params.debug_train_statistics:
            for task in sr_params.tasks:
                trainer.get_generation_statistics(task)
        
        trainer.save_periodic()
        
        # validation 
        try:
            if sr_params.eval_in_domain:
                knn_model.eval()
                scores = evaluate_in_domain(
                    trainer,
                    sr_params,
                    trainer.modules,
                    "valid1",
                    "functions",
                    verbose=True,
                    ablation_to_keep=None,
                    save=False,
                    logger=None,
                    save_file=sr_params.save_eval_dic,
                    knn_model=knn_model,
                )
                logger.info("__log__:%s" % json.dumps(scores))
            
            # TODO: test eval_in_pmlb
            
            trainer.save_best_model(scores, prefix="BFGS", suffix="fit")

        except Exception as e:
            logger.info("Exception during validation: %s" % str(e))
            scores = None
            
        # end of epoch
        trainer.end_epoch(scores)
                
                


def should_stop_early(args, valid_loss):
    # skip check if no validation was done in the current epoch
    if valid_loss is None:
        return False
    if args.patience <= 0:
        return False

    def is_better(a, b):
        return a > b if args.maximize_best_checkpoint_metric else a < b

    prev_best = getattr(should_stop_early, "best", None)
    if prev_best is None or is_better(valid_loss, prev_best):
        should_stop_early.best = valid_loss
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        if should_stop_early.num_runs >= args.patience:
            logger.info(
                "early stop since valid performance hasn't improved for last {} runs".format(
                    args.patience
                )
            )
            return True
        else:
            return False


def cli_main(modify_parser=None):
    parser = options.get_training_parser()
    parser.add_argument("--optor_max_k", type=int, default=4)
    parser.add_argument("--const_max_k", type=int, default=8)
    parser.add_argument("--checkpoint_prefix", type=str, default="checkpoint_last-")
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)
    if args.profile:
        with torch.cuda.profiler.profile():
            with torch.autograd.profiler.emit_nvtx():
                distributed_utils.call_main(args, main)
    else:
        distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()

    

