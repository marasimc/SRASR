import sys
sys.path.append('..')
from symbolicregression.envs import build_env
from symbolicregression.parsers import get_parser
from symbolicregression.slurm import init_signal_handler, init_distributed_mode

parser = get_parser()
params = parser.parse_args()
init_distributed_mode(params)
if params.is_slurm_job:
    init_signal_handler()
env = build_env(params)

import json
data = {
    'equation_id2word': env.equation_id2word,
    'equation_word2id': env.equation_word2id,
    'equation_words': env.equation_words,
    'float_id2word': env.float_id2word,
    'float_word2id': env.float_word2id,
    'float_words': env.float_words
}
with open('env_data.json', 'w') as f:
    json.dump(data, f, indent=4)

# env.equation_id2word
# env.equation_word2id
# env.equation_words
# env.float_id2word
# env.float_word2id
# env.float_words