from typing import Any, Dict, List, Optional, Tuple
import math
from torch import Tensor
import torch
from torch import nn
from knnbox.common_utils import (
    global_vars,
    select_keys_with_pad_mask,
    disable_model_grad,
    enable_module_grad,
    archs,
)
from knnbox.datastore import Datastore
from knnbox.retriever import Retriever
from knnbox.sr_models.enhance_combiner import EnhancedAdaptiveCombiner


class SR_knnModel(nn.Module):

    def __init__(self, args, dictionary):
        super().__init__()
        
        optor_max_k = args.optor_max_k 
        const_max_k = args.const_max_k 
        checkpoint_prefix = args.checkpoint_prefix 
        
        self.update_num = 0
        ## load optor datastore
        self.optor_datastore = Datastore.load(args.knn_datastore_path+'/optor', load_list=["vals"])
        self.optor_datastore.load_faiss_index("keys")
        self.optor_retriever = Retriever(datastore=self.optor_datastore, k=optor_max_k)          
        ## load const datastore
        self.const_datastore = Datastore.load(args.knn_datastore_path+'/const', load_list=["vals"])
        self.const_datastore.load_faiss_index("keys")
        self.const_retriever = Retriever(datastore=self.const_datastore, k=const_max_k)
        
        if args.knn_mode == "train_metak":
            self.optor_combiner = EnhancedAdaptiveCombiner(
                max_k=optor_max_k, 
                probability_dim=len(dictionary),
                k_trainable=(args.knn_k_type=="trainable"),
                lambda_trainable=(args.knn_lambda_type=="trainable"), 
                lamda_=args.knn_lambda,
                temperature_trainable=(args.knn_temperature_type=="trainable"), 
                temperature=args.knn_temperature
            )
            self.const_combiner = EnhancedAdaptiveCombiner(
                max_k=const_max_k, 
                probability_dim=len(dictionary),
                k_trainable=(args.knn_k_type=="trainable"),
                lambda_trainable=(args.knn_lambda_type=="trainable"), 
                lamda_=args.knn_lambda,
                temperature_trainable=(args.knn_temperature_type=="trainable"), 
                temperature=args.knn_temperature
            )
        elif args.knn_mode == "inference":
            self.optor_combiner = EnhancedAdaptiveCombiner.load(args.knn_combiner_path+'/optor', checkpoint_prefix=checkpoint_prefix)
            self.const_combiner = EnhancedAdaptiveCombiner.load(args.knn_combiner_path+'/const', checkpoint_prefix=checkpoint_prefix)

        if args.knn_mode == "train_metak":
            disable_model_grad(self)
            enable_module_grad(self, "combiner")

        self.args = args
        self.add_args()
        self.onnx_trace = True
    
    def add_args(self):
        # self.args.knn_mode = "train_metak"
        # self.args.knn_datastore_path = "datastore"
        # self.args.knn_max_k = 8
        # self.args.knn_combiner_path = "combiner"
        pass
        
    
    def forward(
        self,
        decoded_last_hidden: Tensor,        # [bs, seq_len, dim]
        decoded_probs: Tensor,              # [bs, seq_len, V]
        target: Optional[Tensor] = None,    # [bs, seq_len]
    ):
        if self.args.knn_mode == "inference" or self.args.knn_mode == "train_metak":
            decoded_last_hidden = decoded_last_hidden + torch.randn_like(decoded_last_hidden)*5.0
            
            self.optor_retriever.retrieve(decoded_last_hidden, return_list=["vals", "distances"]) 
            self.const_retriever.retrieve(decoded_last_hidden, return_list=["vals", "distances"])
        else:
            raise NotImplementedError("Only inference mode is supported in this model.")
        
        extra = {'last_hidden': decoded_last_hidden, 'target': target}
        return decoded_probs, extra
    
    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        prev_words: Tensor,                # [bs, seq_len]
        output_layer
    ):
        r"""
        we overwrite this function to change the probability calculation process.
        step 1. 
            calculate the knn probability based on retrieved resultes
        step 2.
            combine the knn probability with NMT's probability 
        """
        if self.args.knn_mode == "inference" or self.args.knn_mode == "train_metak":
            ### A. caculate optor knn probability
            knn_prob = self.optor_combiner.get_knn_prob(**self.optor_retriever.results, neural_model_logit=net_output[0], device=net_output[0].device)  # [bs, 1, V]
            optor_combined_prob, _ = self.optor_combiner.get_combined_prob(knn_prob, net_output[0], log_probs=log_probs)
            
            ### B. caculate const knn probability
            knn_prob = self.const_combiner.get_knn_prob(**self.const_retriever.results, neural_model_logit=net_output[0], device=net_output[0].device)  # [bs, 1, V]
            const_combined_prob, _ = self.const_combiner.get_combined_prob(knn_prob, net_output[0], log_probs=log_probs)
            
            ### C. combine
            if len(prev_words.shape) == 1:
                prev_words = prev_words.unsqueeze(-1)
            optor_mask = (prev_words<=88) | ((prev_words>=91) & (prev_words<=291))
            const_mask = ((prev_words==89) | (prev_words==90)) | (prev_words>=292)
            combined_prob = optor_combined_prob * optor_mask[:, :, None] + const_combined_prob * const_mask[:, :, None]
            
            return combined_prob
        else:
            raise NotImplementedError("Only inference mode is supported in this model.")