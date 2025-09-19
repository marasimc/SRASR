import torch
import torch.nn.functional as F

def js_divergence_subset(p_lm: torch.Tensor,
                         p_knn: torch.Tensor,
                         eps: float = 1e-12):

    orig_shape = p_lm.shape[:-1]
    V = p_lm.size(-1)
    p_lm_flat = p_lm.reshape(-1, V)
    p_knn_flat = p_knn.reshape(-1, V)

    js_list, ent_list = [], []
    for lm_vec, knn_vec in zip(p_lm_flat, p_knn_flat):
        mask = knn_vec > 0
        if not mask.any():
            js_list.append(torch.tensor(0., device=lm_vec.device))
            ent_list.append(torch.tensor(0., device=lm_vec.device))
            continue

        p_lm_sub = lm_vec[mask]
        p_lm_sub = p_lm_sub / (p_lm_sub.sum() + eps)

        p_knn_sub = knn_vec[mask]
        p_knn_sub = p_knn_sub / (p_knn_sub.sum() + eps)

        m = 0.5 * (p_lm_sub + p_knn_sub)
        kl_lm  = (p_lm_sub * torch.log((p_lm_sub + eps) / (m + eps))).sum()
        kl_knn = (p_knn_sub * torch.log((p_knn_sub + eps) / (m + eps))).sum()
        js_list.append(0.5 * (kl_lm + kl_knn))

        ent_list.append(-torch.sum(p_knn_sub * torch.log(p_knn_sub + eps)))

    js_tensor  = torch.stack(js_list).reshape(orig_shape)       
    ent_tensor = torch.stack(ent_list).reshape(orig_shape)     
    
    # return js_tensor, ent_tensor
    return torch.cat([js_tensor.unsqueeze(-1), ent_tensor.unsqueeze(-1)], dim=-1)  
