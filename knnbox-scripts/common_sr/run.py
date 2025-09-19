import os
import torch
import numpy as np
import requests

import sys
sys.path.append('../../.')
import symbolicregression
from symbolicregression.envs import build_env
from symbolicregression.model import build_modules
from symbolicregression.parsers import get_parser
from symbolicregression.trainer import Trainer
from symbolicregression.slurm import init_signal_handler, init_distributed_mode
from symbolicregression.evaluate import evaluate_pmlb, evaluate_in_domain
from symbolicregression.evaluate_standard import evaluate_standard, all_benchmark


def load_model_from_pt():
    #load E2E model
    model_path = "../../symbolicregression_utils/weights/model1.pt" 
    try:
        if not os.path.isfile(model_path): 
            print("Model not found, downloading from the internet...")
            url = "https://dl.fbaipublicfiles.com/symbolicregression/model1.pt"
            r = requests.get(url, allow_redirects=True)
            open(model_path, 'wb').write(r.content)
        if not torch.cuda.is_available():
            model = torch.load(model_path, map_location=torch.device('cpu'))
        else:
            model = torch.load(model_path)
        print("Model successfully loaded!") 

    except Exception as e:
        print("ERROR: model not loaded! path was: {}".format(model_path))
        print(e)
    return model

def load_model_from_stateDict(modules):
    model_path = "./checkpoints/checkpoint.pth"
    
    data = torch.load(model_path)
    # reload model parameters
    for k, v in modules.items():
        weights = data[k]
        try:
            weights = data[k]
            v.load_state_dict(weights)
        except RuntimeError:  # remove the 'module.'
            weights = {name.partition(".")[2]: v for name, v in data[k].items()}
            v.load_state_dict(weights)
    return modules

if __name__ == '__main__':
    #load data:
    parser = get_parser()
    parser.add_argument('--knn_type', type=str, default=None, help='knn type')
    parser.add_argument('--knn_mode', type=str, default='inference', help='knn mode')
    parser.add_argument('--knn_datastore_path', type=str, default=None, help='knn data store path')
    parser.add_argument('--knn_k', type=int, default=8, help='knn k')
    parser.add_argument('--knn_lambda', type=float, default=0.7, help='knn lambda')
    parser.add_argument('--knn_temperature', type=float, default=10.0, help='knn temperature')
    parser.add_argument('--knn_max_k', type=int, default=8, help='knn max k')
    parser.add_argument('--knn_combiner_path', type=str, default=None, help='knn combiner path')
    
    parser.add_argument("--eval_on_standard", type=bool, default=False)
    parser.add_argument("--optor_max_k", type=int, default=4)
    parser.add_argument("--const_max_k", type=int, default=8)
    parser.add_argument("--checkpoint_prefix", type=str, default="checkpoint_last-")
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--steps", type=str, default="")
    params = parser.parse_args()
    init_distributed_mode(params)
    if params.is_slurm_job:
        init_signal_handler()

    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed(params.seed)
    
    # CPU / CUDA
    if not params.cpu:
        assert torch.cuda.is_available()
    params.eval_only = True
    symbolicregression.utils.CUDA = not params.cpu
     
    env = build_env(params)
    env.rng = np.random.RandomState(0)
    modules = build_modules(env, params)
    trainer = Trainer(modules, env, params)
    
    ## load E2E model
    # type 1:
    model = load_model_from_pt()
    state_dict_encoder = model.encoder.state_dict()
    state_dict_decoder = model.decoder.state_dict()
    # type 2:
    # model = load_model_from_stateDict(modules)
    # state_dict_encoder = model['encoder'].state_dict()
    # state_dict_decoder = model['decoder'].state_dict()
    
    params.batch_size_eval = 1
    params.multi_gpu = False
    params.is_slurm_job = False
    params.random_state = 14423
    params.eval_verbose_print = True
    params.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(params.no_prefix_cache, params.no_seq_cache)
    if (params.no_prefix_cache==False) and (params.no_seq_cache == False):
        params.cache = 'b' #both cachings are arctivated
    elif (params.no_prefix_cache==True) and (params.no_seq_cache ==False):
        params.cache = 's' #only sequence caching
    elif (params.no_prefix_cache==False) and (params.no_seq_cache==True):
        params.cache = 'k' #only top-k caching
    else:
        params.cache = 'n' # no caching


    # load knn-model if corresponding parameter is set
    if params.knn_type:
        if params.knn_type == 'enhance':
            print("##################### knn-mode: enhance...")
            from knnbox.sr_models.enhance_knn_sr import SR_knnModel
            knn_model = SR_knnModel(params, env.equation_id2word).cuda()
        else:
            raise NotImplementedError
        knn_model.eval()
    else:
        print("###################### knn-mode: None...")
        knn_model = None
    
    
    # evaluate functions 
    if params.eval_in_domain:
        scores = evaluate_in_domain(
                trainer,
                params,
                model,
                "valid1",
                "functions",
                verbose=True,
                ablation_to_keep=None,
                save=True,
                logger=None,
                save_file=params.save_eval_dic,
                knn_model=knn_model,
            )
        print("Pare-trained E2E scores: ", scores)
    
    if params.eval_on_pmlb:
        target_noise = params.target_noise
        random_state = params.random_state
        data_type = params.pmlb_data_type
        save = params.save_results

        if data_type == "feynman":
            filter_fn = lambda x: x["dataset"].str.contains("feynman")
        elif data_type == "strogatz":
            print("Strogatz data")
            filter_fn = lambda x: x["dataset"].str.contains("strogatz")
        elif data_type == "603_fri_c0_250_50":
            filter_fn = lambda x: x["dataset"].str.contains("603_fri_c0_250_50")
        else:
            filter_fn = lambda x: ~(
                x["dataset"].str.contains("strogatz")
                | x["dataset"].str.contains("feynman")
            )

        save_file = "eval_pmlb"
        os.makedirs(save_file, exist_ok=True)
        pmlb_scores = evaluate_pmlb(
            trainer,
            params,
            model,
            target_noise=target_noise,
            verbose=params.eval_verbose_print,
            random_state=random_state,
            save=save,
            filter_fn=filter_fn,
            save_file=None,
            save_suffix="./{}/eval_pmlb_{}_pretrained_{}.csv".format(save_file, data_type, params.beam_type),
            knn_model=knn_model,
        )
        print("scores: ", pmlb_scores)
    
    if params.eval_on_standard:
        target_noise = params.target_noise
        random_state = params.random_state
        save = True
        save_file = "eval_standard"
        os.makedirs(save_file, exist_ok=True)

        import json
        all_result = {}
        for benchmark in all_benchmark:
            standard_scores = evaluate_standard(
                trainer,
                params,
                model,
                target_noise=target_noise,
                verbose=params.eval_verbose_print,
                random_state=random_state,
                save=save,
                filter_fn=None,
                save_file=None,
                save_suffix="./{}/eval_standard_{}.csv".format(save_file, benchmark),
                knn_model=knn_model,
                benchmark=benchmark
            )
            print("/n ----------- {} --------------/n".format(benchmark))
            print("scores: ", standard_scores)
            
            all_result[benchmark] = standard_scores

        with open(save_file+"/eval_standard_all.json", "w") as f:
            json.dump(all_result, f, indent=4)
        