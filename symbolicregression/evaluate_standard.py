import numpy as np
import pandas as pd
import os
from collections import OrderedDict, defaultdict
from symbolicregression.model.sklearn_wrapper import SymbolicTransformerRegressor , get_top_k_features
from symbolicregression.model.model_wrapper import ModelWrapper
import symbolicregression.model.utils_wrapper as utils_wrapper

from symbolicregression.metrics import compute_metrics
from sklearn.model_selection import train_test_split
import time
from tqdm import tqdm
import copy
import json


POINT_DIR = 'symbolicregression_utils/datasets_standard/datasets'
all_benchmark = [
    'Nguyen',
    'Jin',
    'Neat',
    'Keijzer',
    'Livemore',
    'R_rational',
]

def get_len(benchmark):
    return len(json.load(open(f'{POINT_DIR}/{benchmark}.json', 'r')))

def read_file(benchmark):
    benchmark_data = json.load(open(f'{POINT_DIR}/{benchmark}.json', 'r'))
    for eq_name in benchmark_data:
        yield (benchmark_data[eq_name]['X'], benchmark_data[eq_name]['y'], eq_name)


def evaluate_standard(
        trainer,
        params,
        model,
        target_noise=0.0,
        random_state=29910,
        verbose=False,
        save=True,
        filter_fn=None,
        logger=None,
        save_file=None,
        save_suffix="./eval_result_standard/eval_standard.csv",
        knn_model=None,
        benchmark = 'Nguyen'
    ):
        scores = defaultdict(list)
        env = trainer.env
        params = params
        embedder = model.embedder
        encoder = model.encoder
        decoder = model.decoder
        embedder.eval()
        encoder.eval()
        decoder.eval()

        mw = ModelWrapper(
            env=env,
            embedder=embedder,
            encoder=encoder,
            decoder=decoder,
            knn_model=knn_model,
            beam_length_penalty=params.beam_length_penalty,
            beam_size=params.beam_size,
            max_generated_output_len=params.max_generated_output_len,
            beam_early_stopping=params.beam_early_stopping,
            beam_temperature=params.beam_temperature,
            beam_type=params.beam_type,
        )

        dstr = SymbolicTransformerRegressor(
            model=mw,
            max_input_points=params.max_input_points,
            n_trees_to_refine=params.n_trees_to_refine,
            max_number_bags=params.max_number_bags,
            rescale=params.rescale,
        )
        
        first_write = True
        if save:
            save_file = save_suffix

        run_start = time.time()
        rng = np.random.RandomState(random_state)
        pbar = tqdm(total=get_len(benchmark))
        for X, y, problem_name in read_file(benchmark):
            print("problem_name : ", problem_name)

            X_origin = np.array(X)
            if np.all(X_origin[:, 1] == 0):
                X = X_origin[:, 0]
                X = np.expand_dims(X, axis=-1)
            else:
                X = X_origin.copy()
            y = np.array(y)
            y = np.expand_dims(y, axis=-1)

            # TODO: testing
            # x_to_fit, x_to_predict, y_to_fit, y_to_predict = train_test_split(
            #     X, y, test_size=0.25, shuffle=True, random_state=random_state
            # )
            x_to_fit, x_to_predict, y_to_fit, y_to_predict = X, X, y, y

            scale = target_noise * np.sqrt(np.mean(np.square(y_to_fit)))
            noise = rng.normal(loc=0.0, scale=scale, size=y_to_fit.shape)
            y_to_fit += noise

            dstr.fit(x_to_fit, y_to_fit, verbose=verbose)
            problem_results = defaultdict(list)
           
            for refinement_type in dstr.retrieve_refinements_types():
                best_gen = copy.deepcopy(
                    dstr.retrieve_tree(refinement_type=refinement_type, with_infos=True)
                )
                predicted_tree = best_gen["predicted_tree"]
                if predicted_tree is None:
                    continue
                del best_gen["predicted_tree"]
                if "metrics" in best_gen:
                    del best_gen["metrics"]

                problem_results["predicted_tree"].append(predicted_tree)
                problem_results["predicted_tree_prefix"].append(
                    predicted_tree.prefix() if predicted_tree is not None else None
                )
                for info, val in best_gen.items():
                    problem_results[info].append(val)

                y_tilde_to_fit = dstr.predict(x_to_fit, refinement_type=refinement_type)
                results_fit = compute_metrics(
                    {
                        "true": [y_to_fit],
                        "predicted": [y_tilde_to_fit],
                        "predicted_tree": [predicted_tree],
                    },
                    metrics=params.validation_metrics,
                )
                for k, v in results_fit.items():
                    problem_results[k + "_fit"].extend(v)
                    scores[refinement_type + "|" + k + "_fit"].extend(v)

                y_tilde_to_predict = dstr.predict(
                    x_to_predict, refinement_type=refinement_type
                )
                results_predict = compute_metrics(
                    {
                        "true": [y_to_predict],
                        "predicted": [y_tilde_to_predict],
                        "predicted_tree": [predicted_tree],
                    },
                    metrics=params.validation_metrics,
                )
                for k, v in results_predict.items():
                    problem_results[k + "_predict"].extend(v)
                    scores[refinement_type + "|" + k + "_predict"].extend(v)

            problem_results = pd.DataFrame.from_dict(problem_results)
            problem_results.insert(0, "problem", problem_name)
            # problem_results.insert(0, "formula", formula)
            problem_results["input_dimension"] = x_to_fit.shape[1]

            if save:
                if first_write:
                    problem_results.to_csv(save_file, index=False)
                    first_write = False
                else:
                    problem_results.to_csv(
                        save_file, mode="a", header=False, index=False
                    )
            pbar.update(1)
        for k, v in scores.items():
            scores[k] = np.nanmean(v)
        
        run_end = time.time()
        scores["runtime"] = run_end - run_start
        scores["avg_runtime"] = scores["runtime"] / len(benchmark)
        
        with open(save_file[:-4] + "_scores.json", "w") as f:
            json.dump(scores, f, indent=4)
        
        return scores