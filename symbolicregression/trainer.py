# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


import json
import os
import io
import sys
import time
from logging import getLogger
from collections import OrderedDict
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from .optim import get_optimizer
from .utils import to_cuda
from collections import defaultdict
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import copy

# if torch.cuda.is_available():
has_apex = True
try:
    import apex
except:
    has_apex = False

logger = getLogger()

class Trainer(object):
    def __init__(self, modules, env, params, path=None, root=None, knn_model=None):
        """
        Initialize trainer.
        """

        # modules / params
        self.modules = modules
        self.knn_model = knn_model
        self.params = params
        self.env = env

        # epoch / iteration size
        self.n_steps_per_epoch = params.n_steps_per_epoch
        self.inner_epoch = self.total_samples = self.n_equations = 0
        self.infos_statistics = defaultdict(list)
        self.errors_statistics = defaultdict(int)

        # data iterators
        self.iterators = {}

        # set parameters
        self.set_parameters()

        # set optimizer
        self.set_optimizer()

        # float16 / distributed (AMP)
        self.scaler = None

        # validation metrics
        self.metrics = []
        metrics = [m for m in params.validation_metrics.split(",") if m != ""]
        for m in metrics:
            m = (m, False) if m[0] == "_" else (m, True)
            self.metrics.append(m)
        self.best_metrics = {
            metric: (-np.infty if biggest else np.infty)
            for (metric, biggest) in self.metrics
        }

        # training statistics
        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0
        self.stats = OrderedDict(
            [("processed_e", 0)]
            + [("processed_w", 0)]
            + sum(
                [[(x, []), (f"{x}-AVG-STOP-PROBS", [])] for x in env.TRAINING_TASKS], []
            )
        )
        self.last_time = time.time()

        # reload potential checkpoints
        self.reload_checkpoint(path=path, root=root)

        if params.export_data:
            assert params.reload_data == ""
            params.export_path_prefix = os.path.join(params.dump_path, "data.prefix")
            self.file_handler_prefix = io.open(
                params.export_path_prefix, mode="a", encoding="utf-8"
            )
            logger.info(
                f"Data will be stored in prefix in: {params.export_path_prefix} ..."
            )
            
        # reload exported data
        if params.reload_data != "":
            logger.info(params.reload_data)
            # assert params.num_workers in [0, 1] ##TODO: why have that?
            assert params.export_data is False
            s = [x.split(",") for x in params.reload_data.split(";") if len(x) > 0]
            assert (
                len(s)
                >= 1
            )
            self.data_path = {
                task: (
                    train_path if train_path != "" else None,
                    valid_path if valid_path != "" else None,
                    test_path if test_path != "" else None,
                )
                for task, train_path, valid_path, test_path in s
            }

            logger.info(self.data_path)

            for task in self.env.TRAINING_TASKS:
                assert (task in self.data_path) == (task in params.tasks)
        else:
            self.data_path = None

        # create data loaders
        if not params.eval_only:
            if params.env_base_seed < 0:
                params.env_base_seed = np.random.randint(1_000_000_000)
            self.dataloader = {
                task: iter(self.env.create_train_iterator(task, self.data_path, params))
                for task in params.tasks
            }
        
        self.stopping_criterion = None

    def set_new_train_iterator_params(self, args={}):
        params = self.params
        if params.env_base_seed < 0:
            params.env_base_seed = np.random.randint(1_000_000_000)
        self.dataloader = {
            task: iter(
                self.env.create_train_iterator(task, self.data_path, params, args)
            )
            for task in params.tasks
        }
        logger.info(
            "Succesfully replaced training iterator with following args:{}".format(args)
        )
        return

    def set_parameters(self):
        """
        Set parameters.
        """
        self.parameters = {}
        named_params = []
        for v in self.modules.values():
            named_params.extend(
                [(k, p) for k, p in v.named_parameters() if p.requires_grad]
            )
        if self.knn_model:
            for k,p in self.knn_model.named_parameters():
                named_params.append((k, p))
                
        self.parameters["model"] = [p for k, p in named_params]
        for k, v in self.parameters.items():
            logger.info("Found %i parameters in %s." % (len(v), k))
            assert len(v) >= 1

    def set_optimizer(self):
        """
        Set optimizer.
        """
        params = self.params
        self.optimizer = get_optimizer(
            self.parameters["model"], params.lr, params.optimizer
        )
        logger.info("Optimizer: %s" % type(self.optimizer))


    def optimize(self, loss):
        """
        Optimize.
        """
        # check NaN
        if (loss != loss).data.any():
            logger.warning("NaN detected")

        params = self.params

        # optimizer
        optimizer = self.optimizer

        # regular optimization
        # if params.amp == -1:
        optimizer.zero_grad()
        loss.backward()
        if params.clip_grad_norm > 0:
            clip_grad_norm_(self.parameters["model"], params.clip_grad_norm)
        
        # anneal lr, TODO: if useful or not
        # TODO: add relevant code
        
        optimizer.step()


    def iter(self):
        """
        End of iteration.
        """
        self.n_iter += 1
        self.n_total_iter += 1
        self.print_stats()

    def print_stats(self):
        """
        Print statistics about the training.
        """
        if self.n_total_iter % self.params.print_freq != 0:
            return

        s_total_eq = "- Total Eq: " + "{:.2e}".format(self.n_equations)
        s_iter = "%7i - " % self.n_total_iter
        s_stat = " || ".join(
            [
                "{}: {:7.4f}".format(k.upper().replace("_", "-"), np.mean(v))
                for k, v in self.stats.items()
                if type(v) is list and len(v) > 0
            ]
        )
        for k in self.stats.keys():
            if type(self.stats[k]) is list:
                del self.stats[k][:]

        # learning rates
        s_lr = (" - LR: ") + " / ".join(
            "{:.4e}".format(group["lr"]) for group in self.optimizer.param_groups
        )

        # processing speed
        new_time = time.time()
        diff = new_time - self.last_time
        s_speed = "{:7.2f} equations/s - {:8.2f} words/s - ".format(
            self.stats["processed_e"] * 1.0 / diff,
            self.stats["processed_w"] * 1.0 / diff,
        )
        max_mem = torch.cuda.max_memory_allocated() / 1024 ** 2
        s_mem = " MEM: {:.2f} MB - ".format(max_mem)
        self.stats["processed_e"] = 0
        self.stats["processed_w"] = 0
        self.last_time = new_time
        # log speed + stats + learning rate
        logger.info(s_iter + s_speed + s_mem + s_stat + s_lr + s_total_eq)

    def get_generation_statistics(self, task):

        total_eqs = sum(
            x.shape[0]
            for x in self.infos_statistics[list(self.infos_statistics.keys())[0]]
        )
        logger.info("Generation statistics (to generate {} eqs):".format(total_eqs))

        all_infos = defaultdict(list)
        for info_type, infos in self.infos_statistics.items():
            all_infos[info_type] = torch.cat(infos).tolist()
            infos = [torch.bincount(info) for info in infos]
            max_val = max([info.shape[0] for info in infos])
            aggregated_infos = torch.cat(
                [
                    F.pad(info, (0, max_val - info.shape[0])).unsqueeze(-1)
                    for info in infos
                ],
                -1,
            ).sum(-1)
            non_zeros = aggregated_infos.nonzero(as_tuple=True)[0]
            vals = [
                (
                    non_zero.item(),
                    "{:.2e}".format(
                        (aggregated_infos[non_zero] / aggregated_infos.sum()).item()
                    ),
                )
                for non_zero in non_zeros
            ]
            logger.info("{}: {}".format(info_type, vals))
        all_infos = pd.DataFrame(all_infos)
        g = sns.PairGrid(all_infos)
        g.map_upper(sns.scatterplot)
        g.map_lower(sns.kdeplot, fill=True)
        g.map_diag(sns.histplot, kde=True)
        plt.savefig(
            os.path.join(self.params.dump_path, "statistics_{}.png".format(self.epoch))
        )

        str_errors = "Errors ({} eqs)\n ".format(total_eqs)
        for error_type, count in self.errors_statistics.items():
            str_errors += "{}: {}, ".format(error_type, count)
        logger.info(str_errors[:-2])
        self.errors_statistics = defaultdict(int)
        self.infos_statistics = defaultdict(list)

    def save_checkpoint(self, name, include_optimizer=True):
        """
        Save the model / checkpoints.
        """
        if not self.params.is_master:
            return

        # path = os.path.join(self.params.dump_path, "%s.pth" % name)
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
        path = os.path.join('checkpoints', "%s.pth" % name)
        logger.info("Saving %s to %s ..." % (name, path))

        data = {
            "epoch": self.epoch,
            "n_total_iter": self.n_total_iter,
            "best_metrics": self.best_metrics,
            # "best_stopping_criterion": self.best_stopping_criterion,
            "params": {k: v for k, v in self.params.__dict__.items()},
        }

        for k, v in self.modules.items():
            logger.warning(f"Saving {k} parameters ...")
            data[k] = v.state_dict()

        if include_optimizer:
            logger.warning("Saving optimizer ...")
            data["optimizer"] = self.optimizer.state_dict()
            if self.scaler is not None:
                data["scaler"] = self.scaler.state_dict()

        # torch.save(data, path)
        
        ## knn-sr related code start >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        ### save the combiner model
        assert hasattr(self.knn_model.args, "knn_combiner_path"), "you must provide knn_combiner_path"
        self.knn_model.optor_combiner.dump(self.knn_model.args.knn_combiner_path+'/optor', name_prefix=name)
        self.knn_model.const_combiner.dump(self.knn_model.args.knn_combiner_path+'/const', name_prefix=name)
        logger.info("dumped combiner to {}".format(self.knn_model.args.knn_combiner_path))
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< end

    def reload_checkpoint(self, path=None, root=None, requires_grad=True):
        """
        Reload a checkpoint if we find one.
        """
        if path is None:
            path = "checkpoint.pth"

        if self.params.reload_checkpoint != "":
            checkpoint_path = os.path.join(self.params.reload_checkpoint, path)
            assert os.path.isfile(checkpoint_path)
        else:
            if root is not None:
                checkpoint_path = os.path.join(root, path)
            else:
                checkpoint_path = os.path.join(self.params.dump_path, path)
            if not os.path.isfile(checkpoint_path):
                logger.warning(
                    "Checkpoint path does not exist, {}".format(checkpoint_path)
                )
                return

        logger.warning(f"Reloading checkpoint from {checkpoint_path} ...")
        # data = torch.load(checkpoint_path, map_location="cpu")      # type(data) = ModelWrapper
        data = torch.load(checkpoint_path)      # type(data) = ModelWrapper
        
        # reload model parameters
        self.modules["embedder"] = data.embedder
        self.modules["encoder"] = data.encoder
        self.modules["decoder"] = data.decoder

        # # reload model parameters
        # for k, v in self.modules.items():
        #     weights = data[k]
        #     try:
        #         weights = data[k]
        #         v.load_state_dict(weights)
        #     except RuntimeError:  # remove the 'module.'
        #         weights = {name.partition(".")[2]: v for name, v in data[k].items()}
        #         v.load_state_dict(weights)
        #     v.requires_grad = requires_grad

        # # reload optimizer
        # if self.params.amp == -1 or not self.params.nvidia_apex:
        #     logger.warning("Reloading checkpoint optimizer ...")
        #     self.optimizer.load_state_dict(data["optimizer"])
        # else:
        #     logger.warning("Not reloading checkpoint optimizer.")
        #     for group_id, param_group in enumerate(self.optimizer.param_groups):
        #         if "num_updates" not in param_group:
        #             logger.warning("No 'num_updates' for optimizer.")
        #             continue
        #         logger.warning("Reloading 'num_updates' and 'lr' for optimizer.")
        #         param_group["num_updates"] = data["optimizer"]["param_groups"][
        #             group_id
        #         ]["num_updates"]
        #         param_group["lr"] = self.optimizer.get_lr_for_step(
        #             param_group["num_updates"]
        #         )

        # if self.params.fp16 and not self.params.nvidia_apex:
        #     logger.warning("Reloading gradient scaler ...")
        #     self.scaler.load_state_dict(data["scaler"])
        # else:
        #     assert self.scaler is None and "scaler" not in data

        # # reload main metrics
        # self.epoch = data["epoch"] + 1
        # self.n_total_iter = data["n_total_iter"]
        # self.best_metrics = data["best_metrics"]
        # self.best_stopping_criterion = data["best_stopping_criterion"]
        # logger.warning(
        #     f"Checkpoint reloaded. Resuming at epoch {self.epoch} / iteration {self.n_total_iter} ..."
        # )

    def save_periodic(self):
        """
        Save the models periodically.
        """
        if not self.params.is_master:
            return
        if (
            self.params.save_periodic > 0
            and self.epoch % self.params.save_periodic == 0
        ):
            self.save_checkpoint("periodic-%i" % self.epoch)

    def save_best_model(self, scores, prefix=None, suffix=None):
        """
        Save best models according to given validation metrics.
        """
        if not self.params.is_master:
            return
        for metric, biggest in self.metrics:
            _metric = metric
            if prefix is not None:
                _metric = prefix + "|" + _metric
            if suffix is not None:
                _metric = _metric + "_" + suffix
            if _metric not in scores:
                logger.warning('Metric "%s" not found in scores!' % _metric)
                continue
            factor = 1 if biggest else -1

            if metric in self.best_metrics:
                best_so_far = factor * self.best_metrics[metric]
            else:
                best_so_far = -np.inf
            if factor * scores[_metric] > best_so_far:
                self.best_metrics[metric] = scores[_metric]
                logger.info("New best score for %s: %.6f" % (metric, scores[_metric]))
                self.save_checkpoint("best-%s" % metric)

    def end_epoch(self, scores):
        """
        End the epoch.
        """
        # stop if the stopping criterion has not improved after a certain number of epochs
        if self.stopping_criterion is not None and (
            self.params.is_master or not self.stopping_criterion[0].endswith("_mt_bleu")
        ):
            metric, biggest = self.stopping_criterion
            assert metric in scores, metric
            factor = 1 if biggest else -1
            if factor * scores[metric] > factor * self.best_stopping_criterion:
                self.best_stopping_criterion = scores[metric]
                logger.info(
                    "New best validation score: %f" % self.best_stopping_criterion
                )
                self.decrease_counts = 0
            else:
                logger.info(
                    "Not a better validation score (%i / %i)."
                    % (self.decrease_counts, self.decrease_counts_max)
                )
                self.decrease_counts += 1
            if self.decrease_counts > self.decrease_counts_max:
                logger.info(
                    "Stopping criterion has been below its best value for more "
                    "than %i epochs. Ending the experiment..."
                    % self.decrease_counts_max
                )
                exit()
        self.save_checkpoint("checkpoint_last-")
        self.epoch += 1

    def get_batch(self, task):
        """
        Return a training batch for a specific task.
        """
        try:
            batch, errors = next(self.dataloader[task])
        except Exception as e:
            print(e)
            logger.error(
                "An unknown exception of type {0} occurred in line {1} when fetching batch. "
                "Arguments:{2!r}. Restarting ...".format(
                    type(e).__name__, sys.exc_info()[-1].tb_lineno, e.args
                )
            )
            raise
        return batch, errors

    def export_data(self, task):
        """
        Export data to the disk.
        """
        samples, _ = self.get_batch(task)
        for info in samples["infos"]:
            samples["infos"][info] = list(map(str, samples["infos"][info].tolist()))

        def get_dictionary_slice(idx, dico):
            x = {}
            for d in dico:
                x[d] = dico[d][idx]
            return x

        def float_list_to_str_lst(lst, float_precision):
            for i in range(len(lst)):
                for j in range(len(lst[i])):
                    str_float = f"%.{float_precision}e" % lst[i][j]
                    lst[i][j] = str_float
            return lst

        processed_e = len(samples)
        for i in range(processed_e):
            # prefix
            outputs = {**get_dictionary_slice(i, samples["infos"])}
            x_to_fit = samples["x_to_fit"][i].tolist()
            y_to_fit = samples["y_to_fit"][i].tolist()
            outputs["x_to_fit"] = float_list_to_str_lst(
                x_to_fit, self.params.float_precision
            )
            outputs["y_to_fit"] = float_list_to_str_lst(
                y_to_fit, self.params.float_precision
            )
            outputs["tree"] = samples["tree"][i].prefix()

            self.file_handler_prefix.write(json.dumps(outputs) + "\n")
            self.file_handler_prefix.flush()

        # number of processed sequences / words
        self.n_equations += self.params.batch_size
        self.total_samples += self.params.batch_size
        self.stats["processed_e"] += len(samples)
        
        self.inner_epoch += 1

    def enc_dec_step(self, task):
        """
        Encoding / decoding step.
        """
        params = self.params
        embedder, encoder, decoder = (
            self.modules["embedder"],
            self.modules["encoder"],
            self.modules["decoder"],
        )
        embedder.train()
        encoder.train()
        decoder.train()
        env = self.env

        samples, errors = self.get_batch(task)

        if self.params.debug_train_statistics:
            for info_type, info in samples["infos"].items():
                self.infos_statistics[info_type].append(info)
            for error_type, count in errors.items():
                self.errors_statistics[error_type] += count

        x_to_fit = samples["x_to_fit"]      # list type-> (bs, number_of_points, num_X)
        y_to_fit = samples["y_to_fit"]      # list type-> (bs, number_of_points, num_Y)

        x1 = []
        for seq_id in range(len(x_to_fit)):
            x1.append([])
            for seq_l in range(len(x_to_fit[seq_id])):
                x1[seq_id].append([x_to_fit[seq_id][seq_l], y_to_fit[seq_id][seq_l]])

        x1, len1 = embedder(x1)             

        if self.params.use_skeleton:
            x2, len2 = self.env.batch_equations(
                self.env.word_to_idx(
                    samples["skeleton_tree_encoded"], float_input=False
                )
            )
        else:
            x2, len2 = self.env.batch_equations(    # x2->(max_src_len, bs); len2=(bs)
                self.env.word_to_idx(samples["tree_encoded"], float_input=False)
            )

        alen = torch.arange(params.max_src_len, dtype=torch.long, device=len2.device)   # (max_src_len, )
        pred_mask = (alen[:, None] < len2[None] - 1)        # (max_src_len, bs)
        
        y = x2[1:].masked_select(pred_mask[:-1])        # (all_token_nums, )
        assert len(y) == (len2 - 1).sum().item()        
        x2, len2, y = to_cuda(x2, len2, y)

        if params.amp == -1 or params.nvidia_apex:
            encoded = encoder("fwd", x=x1, lengths=len1, causal=False)
            decoded = decoder(
                "fwd",
                x=x2,
                lengths=len2,
                causal=True,
                src_enc=encoded.transpose(0, 1),
                src_len=len1,
            )   # decoded -> (max_src_len, bs, dim)
            
            decoded_probs, net_loss = decoder(
                "predict", tensor=decoded, pred_mask=pred_mask, y=y, get_scores=False
            )       # for debugging
            ## knn-sr related code start >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            decoded = decoded.transpose(0, 1)  # [bs, max_src_len, dim]
            decoded_probs = decoder.output_layer(decoded)  # [bs, max_src_len, vocab_size]
            ### perform knn-sr
            prev_words = torch.cat([x2.transpose(0, 1)[:, 0:1], x2.transpose(0, 1)[:, :-1]], dim=-1)  # [bs, max_src_len]
            net_output = self.knn_model(decoded, decoded_probs, x2.transpose(0, 1))
            lprobs = self.knn_model.get_normalized_probs(net_output, log_probs=True, output_layer=decoder.output_layer, prev_words=prev_words)     
            
            padding_id = self.env.equation_word2id["<PAD>"]
            to_padding = torch.full((1, x2.size(1)), padding_id).to(x2.device)  # [1, bs]
            target = torch.cat((x2[1:], to_padding), dim=0)  # [max_src_len, bs]

            loss, nll_loss = label_smoothed_nll_loss(
                lprobs.view(-1, lprobs.size(-1)),
                target.view(-1),
                self.params.label_smoothing,
                ignore_index=padding_id,
                reduce=True,
            )
            if self.n_total_iter % self.params.print_freq == 0:
                logger.info('loss: {} | nll_loss: {} | net_loss: {}'.format(loss, nll_loss, net_loss))
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< end
            
            ## debug >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # loss_1, nll_loss_1 = label_smoothed_nll_loss(
            #     decoded_probs.view(-1, decoded_probs.size(-1)),
            #     target.view(-1),
            #     self.params.label_smoothing,
            #     ignore_index=padding_id,
            #     reduce=True,
            # )
            ## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            
        else:
            raise NotImplementedError("This part has not been implemented yet.")
            with torch.cuda.amp.autocast():
                encoded = encoder("fwd", x=x1, lengths=len1, causal=False)
                decoded = decoder(
                    "fwd",
                    x=x2,
                    lengths=len2,
                    causal=True,
                    src_enc=encoded.transpose(0, 1),
                    src_len=len1,
                )
                _, loss = decoder(
                    "predict",
                    tensor=decoded,
                    pred_mask=pred_mask,
                    y=y,
                    get_scores=False,
                )
        self.stats[task].append(loss.item())

        # optimize
        self.optimize(loss)

        # number of processed sequences / words
        self.inner_epoch += 1
        self.n_equations += len1.size(0)
        self.stats["processed_e"] += len1.size(0)
        self.stats["processed_w"] += (len1 + len2 - 2).sum().item()



def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
        target = target.masked_select(~pad_mask)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    # if reduce:
    #     nll_loss = nll_loss.sum()
    #     smooth_loss = smooth_loss.sum()
    if reduce:
        nll_loss = nll_loss.mean()
        smooth_loss = smooth_loss.mean()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss
