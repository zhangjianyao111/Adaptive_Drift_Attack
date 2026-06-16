import copy
import logging
import warnings
from typing import Any, Dict, List, Tuple

import torch
import torch.utils.data as data
import tenseal as ts
import numpy as np
import random

from utils.util_crypto import context_ckks
from utils.util_model import (
    extract_parameters,
    flatten_model_parameters,
    load_model_from_parameters,
    get_gaussian_noise,
    get_laplace_noise,
)
from utils.util_sys import wrap_torch_median
from utils.util_sys import wrap_torch_sort

from utils.util_logger import logger

warnings.filterwarnings("ignore", category=UserWarning, module="tenseal")


def fusion_avg(model_updates: Dict[int, torch.nn.Module]) -> Dict[str, torch.Tensor]:
    avgerage_params = {}
    with torch.no_grad():
        for key in next(iter(model_updates.values())).state_dict():
            weighted_params = torch.zeros_like(
                next(iter(model_updates.values())).state_dict()[key].float()
            )
            for _, model in model_updates.items():
                param = model.state_dict()[key].float()
                weighted_params += param * 1.0 / len(model_updates)
            avgerage_params[key] = weighted_params

    return avgerage_params


def fusion_fedavg(
    model_updates: Dict[int, torch.nn.Module],
    data_size: Dict[int, int],
) -> Dict[str, torch.Tensor]:

    total_data_size = sum(data_size.values())
    weighted_avg_params = {}
    with torch.no_grad():
        for key in next(iter(model_updates.values())).state_dict():
            weighted_params = torch.zeros_like(
                next(iter(model_updates.values())).state_dict()[key].float()
            )
            for client_id, model in model_updates.items():
                weight = data_size[client_id] / total_data_size
                param = model.state_dict()[key].float()
                weighted_params += param * weight
            weighted_avg_params[key] = weighted_params

    return weighted_avg_params


def fusion_krum(
    model_updates: Dict[int, torch.nn.Module],
    max_expected_adversaries=1,
    device=torch.device("cpu"),
) -> Dict[str, torch.Tensor]:

    with torch.no_grad():
        ids = list(model_updates.keys())
        updates = [extract_parameters(model_updates[id]) for id in ids]
        updates = [update.to(device) for update in updates]
        num_updates = len(updates)
        updates_stack = torch.stack(updates)

        dist_matrix = torch.cdist(updates_stack, updates_stack, p=2)
        values, indices = torch.topk(
            dist_matrix,
            k=num_updates - max_expected_adversaries - 1,
            dim=1,
            largest=False,
            sorted=True,
        )
        # logger.debug(f"current krum values: {values}")
        scores = values.sum(dim=1)
        # logger.debug(f"current krum scores: {scores}")
        min_indices = torch.argmin(scores).item()
        logger.debug(f"current krum min index: {min_indices}")
        selected_id = ids[min_indices]
        logger.info(f"selected client id: {selected_id}")

    selected_model = model_updates[selected_id]
    krum_params = selected_model.state_dict()
    return krum_params


def fusion_median(
    model_updates: Dict[int, torch.nn.Module],
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    median_params = {}
    with torch.no_grad():
        for key in next(iter(model_updates.values())).state_dict():
            params = torch.stack(
                [model.state_dict()[key].float() for model in model_updates.values()]
            )
            median_params[key] = wrap_torch_median(params, dim=0, device=device)

    return median_params


def fusion_clipping_median1(
    model_updates: Dict[int, torch.nn.Module],
    clipping_threshold=0.1,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    median_params = {}
    with torch.no_grad():
        for key in next(iter(model_updates.values())).state_dict():
            params = torch.stack(
                [model.state_dict()[key].float() for model in model_updates.values()]
            )
            median_params[key] = wrap_torch_median(params, dim=0, device=device)
            median_params[key] = torch.clamp(
                median_params[key], -clipping_threshold, clipping_threshold
            )
    return median_params

def fusion_clipping_median(
    model_updates: Dict[int, torch.nn.Module],
    clipping_threshold=1.0,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    median_params = {}
    with torch.no_grad():
        for key in next(iter(model_updates.values())).state_dict():
            params = torch.stack(
                [
                    model.state_dict()[key].to(device).float()
                    for model in model_updates.values()
                ]
            )
            params = torch.nan_to_num(params, nan=0.0, posinf=1e6, neginf=-1e6)
            median = wrap_torch_median(params, dim=0, device=device)
            if isinstance(median, tuple):
                median = median[0]
            median_params[key] = torch.clamp(
                median, -clipping_threshold, clipping_threshold
            )
    return median_params


def fusion_trimmed_mean(
    model_updates: Dict[int, torch.nn.Module],
    trimmed_ratio: float = 0.1,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    trimmed_mean_params = {}
    with torch.no_grad():
        for key in next(iter(model_updates.values())).state_dict():
            params = torch.stack(
                [model.state_dict()[key].float() for model in model_updates.values()]
            )
            lower = int(params.size(0) * trimmed_ratio)
            upper = int(params.size(0) * (1 - trimmed_ratio))
            params = wrap_torch_sort(params, dim=0, device=device)[lower:upper]
            trimmed_mean_params[key] = torch.mean(params, dim=0)

    return trimmed_mean_params


def fusion_cos_defense(
    global_model: torch.nn.Module,
    model_updates: Dict[int, torch.nn.Module],
    similarity_threshold: float = None,
) -> Dict[str, Any]:

    global_last_layer = list(global_model.parameters())[-2].view(-1)
    models = list(model_updates.values())
    last_layers = [list(model.parameters())[-2].view(-1) for model in models]

    with torch.no_grad():
        scores = torch.abs(
            torch.nn.functional.cosine_similarity(
                torch.stack(last_layers),
                global_last_layer,
            )
        )
        # print(scores)
        logger.info(f"current fusion scores: {scores}")
        min_score = torch.min(scores)
        scores = (scores - min_score) / (torch.max(scores) - min_score)
        logger.info(f"normalized fusion scores: {scores}")

        if similarity_threshold is None:
            similarity_threshold = torch.mean(scores)
        logger.info(f"similarity threshold: {similarity_threshold}")

        benign_indices = scores >= similarity_threshold
        if torch.sum(benign_indices) == 0:
            logger.warning("No models are considered benign based on the threshold.")
            logger.warning("Return global model of last round.")
            return global_model.state_dict()

        logger.info(f"current round client list: {model_updates.keys()}")
        logger.info(f"potential malicide indices: {benign_indices}")
        logger.info(f"checked benign indices: {benign_indices}")

        weight = 1 / torch.sum(benign_indices).float()
        fractions = benign_indices.float() * weight
        logger.info(f"current fusion fractions: {fractions}")

        weighted_params = copy.deepcopy(global_model.state_dict())
        for param_key in weighted_params.keys():
            temp_param = torch.zeros_like(
                global_model.state_dict()[param_key], dtype=torch.float32
            )
            for model, fraction in zip(models, fractions):
                temp_param += model.state_dict()[param_key] * fraction
            weighted_params[param_key].copy_(temp_param)
            # OUR OPTIMIZATION FOR DEFENSE
            # weighted_params[param_key] = torch.clamp(
            #     weighted_params[param_key], -0.1, 0.1
            # )

    return weighted_params


def fusion_dual_defense1(
    global_model: torch.nn.Module,
    model_updates: Dict[int, torch.nn.Module],
    data_size: Dict[int, int],
    similarity_threshold: float = None,
    epsilon: float = None,
) -> Dict[str, torch.Tensor]:
    # simulate the hyper guard defense (privacy-preserving robust aggregation)
    # 1) each client generates pre-preprocessed update
    global_last_layer = list(global_model.parameters())[-2].view(-1)
    last_layers = {
        client_id: list(model.parameters())[-2].view(-1)
        for client_id, model in model_updates.items()
    }
    mormalized_global = global_last_layer / torch.norm(global_last_layer)
    normalized_locals = {
        client_id: last_layer / torch.norm(last_layer)
        for client_id, last_layer in last_layers.items()
    }
    
    # 2) encrypt and send to the fusion server
    encrypted_global = ts.ckks_vector(
        context_ckks, mormalized_global.flatten().tolist()
    )
    encrypted_locals = {
        client_id: ts.ckks_vector(context_ckks, normalized_local.flatten().tolist())
        for client_id, normalized_local in normalized_locals.items()
    }
    encrypted_updates = {}
    for client_id, model in model_updates.items():
        flattened_parameters = flatten_model_parameters(model)
        encrypted_parameters = [
            ts.ckks_vector(context_ckks, param) for param in flattened_parameters
        ]
        encrypted_updates[client_id] = encrypted_parameters

    if epsilon is not None and isinstance(epsilon, float):
        gaussian_nosie = get_gaussian_noise(
            1, epsilon=epsilon, delta=1.0 / encrypted_global.size(), sensitivity=1
        )
        encrypted_global = (
            encrypted_global + gaussian_nosie.tolist() * encrypted_global.size()
        )

    encrypted_scores = {
        client_id: encrypted_local.dot(encrypted_global)
        for client_id, encrypted_local in encrypted_locals.items()
    }

    # 4) each client decrypt the scores and send back the benigns for validation
    client_selections = {}
    for client_id in model_updates.keys():
        scores = {
            client_id: np.abs(encrypted_score.decrypt())
            for client_id, encrypted_score in encrypted_scores.items()
        }
        logger.debug(f"client {client_id} scores: {scores}")
        min_score = np.min(list(scores.values()))
        max_score = np.max(list(scores.values()))
        diff_score = max_score - min_score
        scores_norm = {
            client_id: (score - min_score) / diff_score
            for client_id, score in scores.items()
        }
        logger.debug(f"client {client_id} norm scores: {scores_norm}")
        if similarity_threshold is None:
            similarity_threshold = np.mean(list(scores_norm.values()))
        logger.debug(f"client {client_id} similarity threshold: {similarity_threshold}")
        selected_benigns = [
            id for id, score in scores_norm.items() if score >= similarity_threshold
        ]
        logger.info(f"client {client_id} selected fusion benigns: {selected_benigns}")
        if len(selected_benigns) == 0:
            raise ValueError("No models are considered benign based on the threshold.")
        client_selections[client_id] = selected_benigns

    # 5) server counts and find the majority beningn selections
    count = {}
    for _, benigns in client_selections.items():
        _tuple = tuple(benigns)
        if _tuple in count:
            count[_tuple] += 1
        else:
            count[_tuple] = 1
    benigns = None
    max_count = 0
    for _benigns, _cnt in count.items():
        if _cnt > max_count:
            max_count = _cnt
            benigns = _benigns

    # 6) final secure aggregation
    logger.debug(f"final fusion benigns: {benigns}")
    total_size = sum(data_size[benign_id] for benign_id in benigns)
    fused_enc_params = [0] * len(encrypted_updates[benigns[0]])
    for benign_id in benigns:
        enc_param = encrypted_updates[benign_id]
        fusion_weight = data_size[benign_id] / total_size


        weighted_enc_param = [_p * fusion_weight for _p in enc_param]
        fused_enc_params = [x + y for x, y in zip(fused_enc_params, weighted_enc_param)]

    # 7) send to client for decryption
    _params = [param.decrypt() for param in fused_enc_params]
    fused_model = load_model_from_parameters(_params, global_model)
    fused_params = fused_model.state_dict()

    return fused_params



def drift_defense_basis(
    global_model: torch.nn.Module,
    model_updates: Dict[int, torch.nn.Module],
    data_size: Dict[int, int],
    similarity_threshold: float = None,
    epsilon: float = None,
) -> Dict[str, torch.Tensor]:
    
    # 1) 提取最后一层参数
    global_last_layer = list(global_model.parameters())[-2].view(-1)
    last_layers = {
        cid: list(model.parameters())[-2].view(-1)
        for cid, model in model_updates.items()
    }
    '''
    # 1) 提取全模型参数
    global_last_layer = torch.cat([p.detach().view(-1) for p in global_model.parameters()])

    last_layers = {
        cid: torch.cat([
            p.detach().view(-1) for p in model.parameters()])
        for cid, model in model_updates.items()
    }
    '''
    # 2) 相似度用归一化向量
    normalized_global = global_last_layer / torch.norm(global_last_layer)
    normalized_locals = {
        cid: last_layer / torch.norm(last_layer)
        for cid, last_layer in last_layers.items()
    }

    encrypted_global = ts.ckks_vector(context_ckks, normalized_global.flatten().tolist())
    encrypted_locals = {
        cid: ts.ckks_vector(context_ckks, normalized_local.flatten().tolist())
        for cid, normalized_local in normalized_locals.items()
    }
    
    
    # 3) 原始向量密文，用于 norm²
    encrypted_updates = {}
    for cid, model in model_updates.items():
        flattened_parameters = flatten_model_parameters(model)
        encrypted_parameters = [
            ts.ckks_vector(context_ckks, param) for param in flattened_parameters
        ]
        encrypted_updates[cid] = encrypted_parameters
    
    # 4) 差分隐私噪声（可选）
    if epsilon is not None and isinstance(epsilon, float):
        gaussian_noise = get_gaussian_noise(
            1, epsilon=epsilon, delta=1.0 / encrypted_global.size(), sensitivity=1
        )
        encrypted_global = (
            encrypted_global + gaussian_noise.tolist() * encrypted_global.size()
        )

    # 5) 服务器计算相似度（归一化向量点积）
    encrypted_scores = {
        cid: encrypted_local.dot(encrypted_global)
        for cid, encrypted_local in encrypted_locals.items()
    }

    # 6) 服务器计算 norm²（原始向量平方和）
    encrypted_norms = {
        cid: sum([param.dot(param) for param in enc_params])
        for cid, enc_params in encrypted_updates.items()
    }


    # 7) 客户端解密
    scores = {cid: float(np.abs(enc_score.decrypt()[0])) for cid, enc_score in encrypted_scores.items()}
    norm_scores = {cid: float(encrypted_norm.decrypt()[0]) for cid, encrypted_norm in encrypted_norms.items()}

    # 相似度归一化
    min_score, max_score = min(scores.values()), max(scores.values())
    diff_score = max_score - min_score
    scores_norm = {cid: (s - min_score) / (diff_score + 1e-8) for cid, s in scores.items()}

    if similarity_threshold is None:
        similarity_threshold = np.mean(list(scores_norm.values()))

    # ===== 每个客户端进行选择 =====
    client_selections = {}
    for cid in model_updates.keys():
        
        # 第一轮：相似度筛选
        mu_sim = np.mean(list(scores_norm.values()))
        sigma_sim = np.std(list(scores_norm.values()))
        first_pass = [
            id for id, s in scores_norm.items()
            if (mu_sim - sigma_sim) <= s <= (mu_sim + sigma_sim)
    ]
        
        # 第二轮：方向贡献筛选
        ratios = {id: scores_norm[id] / (norm_scores[id] + 1e-8) for id in first_pass}
        mu, sigma = np.mean(list(ratios.values())), np.std(list(ratios.values()))
        def energy_ratio_ok(xid):
            return (mu - sigma) <= ratios[xid] <= (mu + sigma)
        selected_benigns = [xid for xid in first_pass if energy_ratio_ok(xid)]
    

        if len(selected_benigns) == 0:
            raise ValueError(f"Client {cid} 没有选出任何 benign — 检查阈值设置")
        client_selections[cid] = selected_benigns
        logger.info(f"[DEBUG] Client {cid} 投票选择: {selected_benigns}")
        

        '''
        # 仅保留相似度筛选
        mu_sim = np.mean(list(scores_norm.values()))
        sigma_sim = np.std(list(scores_norm.values()))
        first_pass = [
            id for id, s in scores_norm.items()
            if (mu_sim - sigma_sim) <= s <= (mu_sim + sigma_sim)
        ]
        selected_benigns = first_pass
        if len(selected_benigns) == 0:
            raise ValueError(f"Client {cid} 没有选出任何 benign — 检查相似度阈值设置")
        client_selections[cid] = selected_benigns
        logger.info(f"[DEBUG] Client {cid} 相似度投票选择: {selected_benigns}")
        '''

        '''
        # 仅保留方向贡献筛选
        first_pass = list(scores_norm.keys())
        ratios = {id: scores_norm[id] / (norm_scores[id] + 1e-8) for id in first_pass}
        mu, sigma = np.mean(list(ratios.values())), np.std(list(ratios.values()))
        def energy_ratio_ok(xid):
            return (mu - sigma) <= ratios[xid] <= (mu + sigma)
        selected_benigns = [xid for xid in first_pass if energy_ratio_ok(xid)]
        if len(selected_benigns) == 0:
            raise ValueError(f"Client {cid} 没有选出任何 benign — 检查方向贡献阈值设置")
        client_selections[cid] = selected_benigns
        logger.info(f"[DEBUG] Client {cid} 方向贡献投票选择: {selected_benigns}")
        '''


    # ===== 多数投票（逐元素统计） =====
    element_count = {}
    for _, benigns in client_selections.items():
        for bid in benigns:
            element_count[bid] = element_count.get(bid, 0) + 1

    # 取超过半数的客户端 ID
    majority_threshold = len(model_updates) // 2
    benigns = tuple([bid for bid, cnt in element_count.items() if cnt >= majority_threshold]) 
    if len(benigns) == 0:
        raise ValueError("最终没有任何 benign 客户端被选中 — 检查投票逻辑")
    logger.info(f"[DEBUG] 最终多数投票结果: {benigns}")

    # ===== 安全聚合 =====
    fused_enc_params = [0] * len(encrypted_updates[benigns[0]])
    total_size = sum(data_size[bid] for bid in benigns)
    for bid in benigns:
        enc_param = encrypted_updates[bid]
        fusion_weight = data_size[bid] / total_size
        weighted_enc_param = [_p * fusion_weight for _p in enc_param]
        fused_enc_params = [x + y for x, y in zip(fused_enc_params, weighted_enc_param)]

    # 解密并返回模型
    _params = [param.decrypt() for param in fused_enc_params]
    fused_model = load_model_from_parameters(_params, global_model)
    fused_params = fused_model.state_dict()

    return fused_params





# ============================================================
# CIFAR-10 compatible CKKS helper functions
# 只用于 CIFAR-10 / ResNet 等大模型参数分块适配
# ============================================================

CKKS_CHUNK_SIZE = 4096

def split_tensor_to_chunks(tensor, chunk_size=CKKS_CHUNK_SIZE):
    flat = tensor.detach().cpu().view(-1).tolist()

    if len(flat) == 0:
        return [[]]

    return [
        flat[i:i + chunk_size]
        for i in range(0, len(flat), chunk_size)
    ]


def encrypt_float_tensor(tensor, chunk_size=CKKS_CHUNK_SIZE):
    chunks = split_tensor_to_chunks(tensor, chunk_size)
    encrypted_chunks = []
    for chunk in chunks:
        if len(chunk) == 0:
            continue
        encrypted_chunks.append(ts.ckks_vector(context_ckks, chunk))
    if len(encrypted_chunks) == 0:
        raise ValueError("Empty float tensor cannot be encrypted.")
    return encrypted_chunks


def encrypted_dot_chunked(enc_a, enc_b):
    if len(enc_a) != len(enc_b):
        raise ValueError(
            f"Encrypted chunk length mismatch: {len(enc_a)} vs {len(enc_b)}"
        )

    result = enc_a[0].dot(enc_b[0])

    for a, b in zip(enc_a[1:], enc_b[1:]):
        result += a.dot(b)

    return result


def encrypted_norm2_chunked(enc_chunks):
    result = enc_chunks[0].dot(enc_chunks[0])

    for x in enc_chunks[1:]:
        result += x.dot(x)

    return result


def encrypted_sampled_norm2(sampled_parts):
    result = None

    for enc_chunks, scale_l in sampled_parts:
        part_norm = encrypted_norm2_chunked(enc_chunks) * scale_l

        if result is None:
            result = part_norm
        else:
            result += part_norm

    if result is None:
        raise ValueError("No sampled parameter parts for norm-square computation.")

    return result


def encrypt_model_state_dict(model, chunk_size=CKKS_CHUNK_SIZE):
    encrypted_state = {}
    meta = {}

    state = model.state_dict()

    for k, v in state.items():
        is_float = torch.is_floating_point(v)

        meta[k] = {
            "shape": tuple(v.shape),
            "dtype": v.dtype,
            "is_float": is_float,
        }
        if is_float:
            encrypted_state[k] = encrypt_float_tensor(v, chunk_size)
        else:
            encrypted_state[k] = v.detach().cpu().clone()
    return encrypted_state, meta


def decrypt_encrypted_state_dict(fused_enc_state, meta, global_model):
    ref_state = global_model.state_dict()
    fused_state = {}

    for k, ref_v in ref_state.items():
        if meta[k]["is_float"]:
            flat_values = []

            for enc_chunk in fused_enc_state[k]:
                flat_values.extend(enc_chunk.decrypt())

            flat_values = flat_values[:ref_v.numel()]

            fused_state[k] = (
                torch.tensor(flat_values, dtype=ref_v.dtype)
                .view_as(ref_v)
                .to(ref_v.device)
            )
        else:
            fused_state[k] = (
                fused_enc_state[k]
                .to(ref_v.device)
                .to(dtype=ref_v.dtype)
            )
    return fused_state



def drift_defense(
    global_model: torch.nn.Module,
    model_updates: Dict[int, torch.nn.Module],
    data_size: Dict[int, int],
    similarity_threshold: float = None,
    epsilon: float = None,
) -> Dict[str, torch.Tensor]:

    # =====================================================
    # 1) 提取最后一层参数：用于 similarity
    # CIFAR-10 / ResNet 一般为 fc.weight，通常对应 list(parameters())[-2]
    # =====================================================
    global_last_layer = list(global_model.parameters())[-2].detach().cpu().view(-1)

    last_layers = {
        cid: list(model.parameters())[-2].detach().cpu().view(-1)
        for cid, model in model_updates.items()
    }

    # =====================================================
    # 2) 相似度：最后一层单位向量归一化后再分块加密
    # =====================================================
    normalized_global = global_last_layer / (torch.norm(global_last_layer) + 1e-8)

    normalized_locals = {
        cid: last_layer / (torch.norm(last_layer) + 1e-8)
        for cid, last_layer in last_layers.items()
    }

    encrypted_global = encrypt_float_tensor(normalized_global)

    encrypted_locals = {
        cid: encrypt_float_tensor(normalized_local)
        for cid, normalized_local in normalized_locals.items()
    }

    # =====================================================
    # 3) 分层抽样更新向量密文：用于近似 full-layer update norm²
    #    每一层固定抽取一部分参数，并用 d_l / m_l 做放大
    #    n_i ≈ Σ_l (d_l / m_l) * Σ_{j∈S_l} (W_i,j^l - W_g,j^l)^2
    # =====================================================
    sample_ratio = 0.1
    encrypted_sampled_raw = {}

    global_params = [
        p.detach().cpu()
        for p in global_model.parameters()
    ]

    for cid, model in model_updates.items():
        sampled_parts = []

        for p_local, p_global in zip(model.parameters(), global_params):
            # 关键：这里计算客户端更新量，而不是完整模型参数
            vec = (p_local.detach().cpu() - p_global).view(-1)

            d_l = vec.numel()
            if d_l == 0:
                continue

            m_l = max(1, int(d_l * sample_ratio))

            # 同一层维度下，所有客户端使用相同抽样位置，保证 sampled norm² 可比
            idx = np.linspace(0, d_l - 1, m_l, dtype=int)

            sampled_vec = vec[idx]
            scale_l = d_l / m_l

            enc_sampled_chunks = encrypt_float_tensor(sampled_vec)

            sampled_parts.append((enc_sampled_chunks, scale_l))

        encrypted_sampled_raw[cid] = sampled_parts

    # =====================================================
    # 4) 全层模型参数密文：只用于最终安全聚合
    # =====================================================
    encrypted_updates = {}
    meta = None

    for cid, model in model_updates.items():
        enc_state, meta_state = encrypt_model_state_dict(model)
        encrypted_updates[cid] = enc_state

        if meta is None:
            meta = meta_state

    # =====================================================
    # 5) 差分隐私噪声，可选
    # =====================================================
    if epsilon is not None and isinstance(epsilon, float):
        noisy_encrypted_global = []

        for enc_chunk in encrypted_global:
            chunk_size = enc_chunk.size()

            gaussian_noise = get_gaussian_noise(
                1,
                epsilon=epsilon,
                delta=1.0 / chunk_size,
                sensitivity=1
            )

            noisy_chunk = enc_chunk + gaussian_noise.tolist() * chunk_size
            noisy_encrypted_global.append(noisy_chunk)

        encrypted_global = noisy_encrypted_global

    # =====================================================
    # 6) 服务器计算相似度：最后一层单位向量密文点积
    # =====================================================
    encrypted_scores = {
        cid: encrypted_dot_chunked(encrypted_local, encrypted_global)
        for cid, encrypted_local in encrypted_locals.items()
    }

    # =====================================================
    # 7) 服务器计算 update norm²：分层抽样近似 full-layer update norm²
    #    n_i ≈ Σ_l (d_l / m_l) * Σ_{j∈S_l} (W_i,j^l - W_g,j^l)^2
    # =====================================================
    encrypted_norms = {
        cid: encrypted_sampled_norm2(sampled_parts)
        for cid, sampled_parts in encrypted_sampled_raw.items()
    }

    # =====================================================
    # 8) 客户端解密辅助筛选统计量
    # =====================================================
    scores = {
        cid: float(np.abs(enc_score.decrypt()[0]))
        for cid, enc_score in encrypted_scores.items()
    }

    norm_scores = {}
    raw_norm_scores = {}
    invalid_norm_clients = set()

    MIN_VALID_NORM = 1e-8

    for cid, enc_norm in encrypted_norms.items():
        raw_norm = float(enc_norm.decrypt()[0])
        raw_norm_scores[cid] = raw_norm

        if (not np.isfinite(raw_norm)) or raw_norm <= MIN_VALID_NORM:
            invalid_norm_clients.add(cid)
            norm_scores[cid] = np.nan

            logger.warning(
                f"[WARNING] invalid update norm score detected: "
                f"client={cid}, raw_norm={raw_norm}. "
                f"This client will be removed before norm screening."
            )
        else:
            norm_scores[cid] = raw_norm

    logger.info(f"[DEBUG] raw similarity scores: {scores}")
    logger.info(f"[DEBUG] raw sampled full-layer update norm scores: {raw_norm_scores}")
    logger.info(f"[DEBUG] valid sampled full-layer update norm scores: {norm_scores}")
    logger.info(f"[DEBUG] invalid norm clients: {invalid_norm_clients}")

    # =====================================================
    # 9) 相似度标量归一化：映射到 [0, 1]
    # =====================================================
    min_score, max_score = min(scores.values()), max(scores.values())
    diff_score = max_score - min_score

    scores_norm = {
        cid: (s - min_score) / (diff_score + 1e-8)
        for cid, s in scores.items()
    }

    logger.info(f"[DEBUG] normalized similarity scores: {scores_norm}")

    if similarity_threshold is None:
        similarity_threshold = np.mean(list(scores_norm.values()))

    # =====================================================
    # 10) 客户端进行两阶段筛选
    # =====================================================
    client_selections = {}

    tau_sim = 1
    tau_norm = 2.0

    # 最大间隔触发阈值：
    # 用于判断 norm² 是否出现明显断层。
    gap_trigger = 10.0
    for cid in model_updates.keys():

        # -------------------------------------------------
        # 第一轮：相似度筛选
        # 使用 median 作为中心，MAD 作为稳健尺度
        # -------------------------------------------------
        sim_values = np.asarray(list(scores_norm.values()), dtype=np.float64)

        mu_sim = np.median(sim_values)
        mad_sim = np.median(np.abs(sim_values - mu_sim))
        sigma_sim = 1.4826 * mad_sim

        if sigma_sim < 1e-8:
            sigma_sim = np.std(sim_values) + 1e-8

        first_pass = [
            xid for xid, s in scores_norm.items()
            if (mu_sim - tau_sim * sigma_sim) <= s <= (mu_sim + tau_sim * sigma_sim)
        ]

        # -------------------------------------------------
        # 剔除 norm² 非法客户端
        # -------------------------------------------------
        first_pass_before_norm_filter = list(first_pass)

        first_pass = [
            xid for xid in first_pass
            if xid not in invalid_norm_clients
        ]

        removed_by_invalid_norm = [
            xid for xid in first_pass_before_norm_filter
            if xid in invalid_norm_clients
        ]

        if len(removed_by_invalid_norm) > 0:
            logger.warning(
                f"[WARNING] Client {cid} removed invalid-norm clients "
                f"before norm screening: {removed_by_invalid_norm}"
            )

        if len(first_pass) == 0:
            raise ValueError(
                f"Client {cid} 第一阶段后没有任何 valid-norm 客户端 — "
                f"检查 norm² 解密结果或攻击强度"
            )

        # -------------------------------------------------
        # 第二轮：full-layer update norm² 上界筛选
        # -------------------------------------------------
        candidate_norms = {
            xid: norm_scores[xid]
            for xid in first_pass
            if xid in norm_scores
            and np.isfinite(norm_scores[xid])
            and norm_scores[xid] > MIN_VALID_NORM
        }

        if len(candidate_norms) == 0:
            raise ValueError(
                f"Client {cid} 第二阶段没有任何 valid update norm² 客户端 — "
                f"检查 norm² 解密结果"
            )

        sorted_norm_items = sorted(candidate_norms.items(), key=lambda x: x[1])
        norm_values = np.asarray([n for _, n in sorted_norm_items], dtype=np.float64)

        median_all = np.median(norm_values)
        max_norm = np.max(norm_values)
        min_norm = np.min(norm_values)

        # 旧指标：只用于日志观察，不作为触发条件
        spread_ratio = max_norm / (median_all + 1e-8)

        # -------------------------------------------------
        # 新触发逻辑：排序后看相邻 norm² 的最大跳变
        # 比 max / median 更适合双峰分布
        # -------------------------------------------------
        if len(norm_values) >= 4:
            gaps = norm_values[1:] / (norm_values[:-1] + 1e-8)
            max_gap_idx = int(np.argmax(gaps))
            max_gap_ratio = float(gaps[max_gap_idx])
        else:
            gaps = np.asarray([], dtype=np.float64)
            max_gap_idx = -1
            max_gap_ratio = 1.0

        if max_gap_ratio < gap_trigger:
            # 没有明显大范数断层：不强筛 norm
            selected_benigns = list(candidate_norms.keys())

            mu_norm = median_all
            sigma_norm = np.std(norm_values) + 1e-8
            upper_norm = float("inf")

        else:
            # 有明显断层：断层前面的低范数组作为正常参考组
            ref_items = sorted_norm_items[:max_gap_idx + 1]

            # 至少保留 2 个参考点，避免尺度估计退化
            if len(ref_items) < 2:
                ref_items = sorted_norm_items[:max(2, len(sorted_norm_items) // 2)]

            ref_norms = np.asarray(
                [n for _, n in ref_items],
                dtype=np.float64
            )

            mu_norm = np.median(ref_norms)
            mad_norm = np.median(np.abs(ref_norms - mu_norm))
            sigma_norm = 1.4826 * mad_norm

            if sigma_norm < 1e-8:
                sigma_norm = np.std(ref_norms) + 1e-8

            upper_norm = mu_norm + tau_norm * sigma_norm

            selected_benigns = [
                xid for xid, n in candidate_norms.items()
                if n <= upper_norm
            ]

        if len(selected_benigns) == 0:
            raise ValueError(
                f"Client {cid} 第二阶段没有选出任何 benign — 检查 norm² 阈值设置"
            )

        client_selections[cid] = selected_benigns

        logger.info(f"[DEBUG] Client {cid} robust sim center: {mu_sim}")
        logger.info(f"[DEBUG] Client {cid} robust sim scale: {sigma_sim}")
        logger.info(f"[DEBUG] Client {cid} first_pass: {first_pass}")
        logger.info(f"[DEBUG] Client {cid} robust update norm center: {mu_norm}")
        logger.info(f"[DEBUG] Client {cid} robust update norm scale: {sigma_norm}")
        logger.info(f"[DEBUG] Client {cid} update norm min: {min_norm}")
        logger.info(f"[DEBUG] Client {cid} update norm max: {max_norm}")
        logger.info(f"[DEBUG] Client {cid} norm spread ratio: {spread_ratio}")
        logger.info(f"[DEBUG] Client {cid} norm max gap ratio: {max_gap_ratio}")
        logger.info(f"[DEBUG] Client {cid} norm upper: {upper_norm}")
        logger.info(f"[DEBUG] Client {cid} candidate norms: {candidate_norms}")
        logger.info(f"[DEBUG] Client {cid} 投票选择: {selected_benigns}")

    # =====================================================
    # 11) 多数投票
    # =====================================================
    element_count = {}

    for _, benigns_list in client_selections.items():
        for bid in benigns_list:
            element_count[bid] = element_count.get(bid, 0) + 1

    majority_threshold = len(model_updates) // 2

    benigns = tuple([
        bid for bid, cnt in element_count.items()
        if cnt >= majority_threshold and bid not in invalid_norm_clients
    ])

    if len(benigns) == 0:
        raise ValueError("最终没有任何 benign 客户端被选中 — 检查投票逻辑")

    logger.info(f"[DEBUG] 投票计数: {element_count}")
    logger.info(f"[DEBUG] 最终多数投票结果: {benigns}")
    logger.info(f"[DEBUG] benigns num: {len(benigns)}")

    # =====================================================
    # 12) 最终安全聚合：仍然使用完整 encrypted_updates
    # =====================================================
    total_size = sum(data_size[bid] for bid in benigns)

    fusion_weights = {
        bid: data_size[bid] / total_size
        for bid in benigns
    }

    logger.info(f"[DEBUG] fusion weights: {fusion_weights}")
    logger.info(f"[DEBUG] fusion weight sum: {sum(fusion_weights.values())}")

    first_bid = benigns[0]
    fused_enc_state = {}

    for k in encrypted_updates[first_bid].keys():

        if meta[k]["is_float"]:
            num_chunks = len(encrypted_updates[first_bid][k])
            fused_chunks = []

            for chunk_idx in range(num_chunks):
                fused_chunk = (
                    encrypted_updates[first_bid][k][chunk_idx]
                    * fusion_weights[first_bid]
                )

                for bid in benigns[1:]:
                    fused_chunk += (
                        encrypted_updates[bid][k][chunk_idx]
                        * fusion_weights[bid]
                    )

                fused_chunks.append(fused_chunk)

            fused_enc_state[k] = fused_chunks

        else:
            # 非浮点 buffer，例如 BatchNorm num_batches_tracked
            # 不能做加权平均，直接复制第一个 benign 的值
            fused_enc_state[k] = encrypted_updates[first_bid][k].clone()

    # =====================================================
    # 13) 解密并返回聚合模型 state_dict
    # =====================================================
    fused_params = decrypt_encrypted_state_dict(
        fused_enc_state=fused_enc_state,
        meta=meta,
        global_model=global_model,
    )

    # =====================================================
    # 14) 检查 NaN / Inf
    # =====================================================
    for k, v in fused_params.items():
        if torch.is_floating_point(v):
            if torch.isnan(v).any() or torch.isinf(v).any():
                raise ValueError(f"NaN/Inf detected in fused parameter: {k}")

    return fused_params






def drift_defense3(
    global_model: torch.nn.Module,
    model_updates: Dict[int, torch.nn.Module],
    data_size: Dict[int, int],
    similarity_threshold: float = None,
    epsilon: float = None,
) -> Dict[str, torch.Tensor]:

    # =====================================================
    # 1) 提取最后一层参数：用于 similarity
    # CIFAR-10 / ResNet 一般为 fc.weight，通常对应 list(parameters())[-2]
    # =====================================================
    global_last_layer = list(global_model.parameters())[-2].detach().cpu().view(-1)

    last_layers = {
        cid: list(model.parameters())[-2].detach().cpu().view(-1)
        for cid, model in model_updates.items()
    }

    # =====================================================
    # 2) 相似度：最后一层单位向量归一化后再分块加密
    # =====================================================
    normalized_global = global_last_layer / (torch.norm(global_last_layer) + 1e-8)

    normalized_locals = {
        cid: last_layer / (torch.norm(last_layer) + 1e-8)
        for cid, last_layer in last_layers.items()
    }

    encrypted_global = encrypt_float_tensor(normalized_global)

    encrypted_locals = {
        cid: encrypt_float_tensor(normalized_local)
        for cid, normalized_local in normalized_locals.items()
    }

    # =====================================================
    # 3) 分层抽样更新向量密文：用于近似 full-layer update norm²
    #    每一层固定抽取一部分参数，并用 d_l / m_l 做放大
    #    n_i ≈ Σ_l (d_l / m_l) * Σ_{j∈S_l} (W_i,j^l - W_g,j^l)^2
    # =====================================================
    sample_ratio = 0.1
    encrypted_sampled_raw = {}

    global_params = [
        p.detach().cpu()
        for p in global_model.parameters()
    ]

    for cid, model in model_updates.items():
        sampled_parts = []

        for p_local, p_global in zip(model.parameters(), global_params):
            # 计算客户端更新量，而不是完整模型参数
            vec = (p_local.detach().cpu() - p_global).view(-1)

            d_l = vec.numel()
            if d_l == 0:
                continue

            m_l = max(1, int(d_l * sample_ratio))

            # 同一层维度下，所有客户端使用相同抽样位置，保证 sampled norm² 可比
            idx = np.linspace(0, d_l - 1, m_l, dtype=int)

            sampled_vec = vec[idx]
            scale_l = d_l / m_l

            enc_sampled_chunks = encrypt_float_tensor(sampled_vec)

            sampled_parts.append((enc_sampled_chunks, scale_l))

        encrypted_sampled_raw[cid] = sampled_parts

    # =====================================================
    # 4) 全层模型参数密文：只用于最终安全聚合
    # =====================================================
    encrypted_updates = {}
    meta = None

    for cid, model in model_updates.items():
        enc_state, meta_state = encrypt_model_state_dict(model)
        encrypted_updates[cid] = enc_state

        if meta is None:
            meta = meta_state

    # =====================================================
    # 5) 差分隐私噪声，可选
    # =====================================================
    if epsilon is not None and isinstance(epsilon, float):
        noisy_encrypted_global = []

        for enc_chunk in encrypted_global:
            chunk_size = enc_chunk.size()

            gaussian_noise = get_gaussian_noise(
                1,
                epsilon=epsilon,
                delta=1.0 / chunk_size,
                sensitivity=1
            )

            noisy_chunk = enc_chunk + gaussian_noise.tolist() * chunk_size
            noisy_encrypted_global.append(noisy_chunk)

        encrypted_global = noisy_encrypted_global

    # =====================================================
    # 6) 服务器计算相似度：最后一层单位向量密文点积
    # =====================================================
    encrypted_scores = {
        cid: encrypted_dot_chunked(encrypted_local, encrypted_global)
        for cid, encrypted_local in encrypted_locals.items()
    }

    # =====================================================
    # 7) 服务器计算 update norm²：分层抽样近似 full-layer update norm²
    #    n_i ≈ Σ_l (d_l / m_l) * Σ_{j∈S_l} (W_i,j^l - W_g,j^l)^2
    # =====================================================
    encrypted_norms = {
        cid: encrypted_sampled_norm2(sampled_parts)
        for cid, sampled_parts in encrypted_sampled_raw.items()
    }

    # =====================================================
    # 8) 客户端解密辅助筛选统计量
    # =====================================================
    scores = {
        cid: float(np.abs(enc_score.decrypt()[0]))
        for cid, enc_score in encrypted_scores.items()
    }

    norm_scores = {}
    raw_norm_scores = {}
    invalid_norm_clients = set()

    MIN_VALID_NORM = 1e-8

    for cid, enc_norm in encrypted_norms.items():
        raw_norm = float(enc_norm.decrypt()[0])
        raw_norm_scores[cid] = raw_norm

        if (not np.isfinite(raw_norm)) or raw_norm <= MIN_VALID_NORM:
            invalid_norm_clients.add(cid)
            norm_scores[cid] = np.nan

            logger.warning(
                f"[WARNING] invalid update norm score detected: "
                f"client={cid}, raw_norm={raw_norm}. "
                f"This client will be removed before norm screening."
            )
        else:
            norm_scores[cid] = raw_norm

    logger.info(f"[DEBUG] raw similarity scores: {scores}")
    logger.info(f"[DEBUG] raw sampled full-layer update norm scores: {raw_norm_scores}")
    logger.info(f"[DEBUG] valid sampled full-layer update norm scores: {norm_scores}")
    logger.info(f"[DEBUG] invalid norm clients: {invalid_norm_clients}")

    # =====================================================
    # 9) 相似度标量归一化：映射到 [0, 1]
    # =====================================================
    min_score, max_score = min(scores.values()), max(scores.values())
    diff_score = max_score - min_score

    scores_norm = {
        cid: (s - min_score) / (diff_score + 1e-8)
        for cid, s in scores.items()
    }

    logger.info(f"[DEBUG] normalized similarity scores: {scores_norm}")

    if similarity_threshold is None:
        similarity_threshold = np.mean(list(scores_norm.values()))

    # =====================================================
    # 10) 客户端进行两阶段筛选
    # =====================================================
    client_selections = {}

    tau_sim = 1
    tau_norm = 2.0

    # Scaling 大范数断层触发阈值
    gap_trigger = 10.0

    # ALIE / 弱异常场景的软 norm 筛选阈值
    tau_norm_soft = 2.5

    # 软筛选至少保留的客户端数，避免无攻击或弱异常时误杀太多导致学习变慢
    min_keep = max(2, len(model_updates) // 2)

    for cid in model_updates.keys():

        # -------------------------------------------------
        # 第一轮：相似度筛选
        # 使用 median 作为中心，MAD 作为稳健尺度
        # -------------------------------------------------
        sim_values = np.asarray(list(scores_norm.values()), dtype=np.float64)

        mu_sim = np.median(sim_values)
        mad_sim = np.median(np.abs(sim_values - mu_sim))
        sigma_sim = 1.4826 * mad_sim

        if sigma_sim < 1e-8:
            sigma_sim = np.std(sim_values) + 1e-8

        first_pass = [
            xid for xid, s in scores_norm.items()
            if (mu_sim - tau_sim * sigma_sim) <= s <= (mu_sim + tau_sim * sigma_sim)
        ]

        # -------------------------------------------------
        # 剔除 norm² 非法客户端
        # -------------------------------------------------
        first_pass_before_norm_filter = list(first_pass)

        first_pass = [
            xid for xid in first_pass
            if xid not in invalid_norm_clients
        ]

        removed_by_invalid_norm = [
            xid for xid in first_pass_before_norm_filter
            if xid in invalid_norm_clients
        ]

        if len(removed_by_invalid_norm) > 0:
            logger.warning(
                f"[WARNING] Client {cid} removed invalid-norm clients "
                f"before norm screening: {removed_by_invalid_norm}"
            )

        if len(first_pass) == 0:
            raise ValueError(
                f"Client {cid} 第一阶段后没有任何 valid-norm 客户端 — "
                f"检查 norm² 解密结果或攻击强度"
            )

        # -------------------------------------------------
        # 第二轮：full-layer update norm² 筛选
        # -------------------------------------------------
        candidate_norms = {
            xid: norm_scores[xid]
            for xid in first_pass
            if xid in norm_scores
            and np.isfinite(norm_scores[xid])
            and norm_scores[xid] > MIN_VALID_NORM
        }

        if len(candidate_norms) == 0:
            raise ValueError(
                f"Client {cid} 第二阶段没有任何 valid update norm² 客户端 — "
                f"检查 norm² 解密结果"
            )

        sorted_norm_items = sorted(candidate_norms.items(), key=lambda x: x[1])
        norm_values = np.asarray([n for _, n in sorted_norm_items], dtype=np.float64)

        median_all = np.median(norm_values)
        max_norm = np.max(norm_values)
        min_norm = np.min(norm_values)

        # 仅用于日志观察
        spread_ratio = max_norm / (median_all + 1e-8)

        # -------------------------------------------------
        # 先检测是否存在大范数断层
        # 用相邻 norm² 的最大跳变识别 Scaling 类攻击
        # -------------------------------------------------
        if len(norm_values) >= 4:
            gaps = norm_values[1:] / (norm_values[:-1] + 1e-8)
            max_gap_idx = int(np.argmax(gaps))
            max_gap_ratio = float(gaps[max_gap_idx])
        else:
            gaps = np.asarray([], dtype=np.float64)
            max_gap_idx = -1
            max_gap_ratio = 1.0

        if max_gap_ratio >= gap_trigger:
            # -------------------------------------------------
            # 情况 1：有明显大范数断层
            # 主要用于防 Scaling 这类大范数攻击
            # -------------------------------------------------
            ref_items = sorted_norm_items[:max_gap_idx + 1]

            if len(ref_items) < 2:
                ref_items = sorted_norm_items[:max(2, len(sorted_norm_items) // 2)]

            ref_norms = np.asarray(
                [n for _, n in ref_items],
                dtype=np.float64
            )

            mu_norm = np.median(ref_norms)
            mad_norm = np.median(np.abs(ref_norms - mu_norm))
            sigma_norm = 1.4826 * mad_norm

            if sigma_norm < 1e-8:
                sigma_norm = np.std(ref_norms) + 1e-8

            upper_norm = mu_norm + tau_norm * sigma_norm

            selected_benigns = [
                xid for xid, n in candidate_norms.items()
                if n <= upper_norm
            ]

            norm_screen_mode = "max_gap"

        else:
            # -------------------------------------------------
            # 情况 2：没有明显大范数断层
            # 回退到普通 robust norm 筛选，保留对 ALIE / 弱异常的敏感性
            # -------------------------------------------------
            mu_norm = np.median(norm_values)
            mad_norm = np.median(np.abs(norm_values - mu_norm))
            sigma_norm = 1.4826 * mad_norm

            if sigma_norm < 1e-8:
                sigma_norm = np.std(norm_values) + 1e-8

            upper_norm = mu_norm + tau_norm_soft * sigma_norm

            selected_benigns = [
                xid for xid, n in candidate_norms.items()
                if n <= upper_norm
            ]

            norm_screen_mode = "soft_mad"

            # 如果软筛选误杀太多，则回退到 first_pass，避免无攻击阶段学习变慢
            if len(selected_benigns) < min_keep:
                selected_benigns = list(candidate_norms.keys())
                upper_norm = float("inf")
                norm_screen_mode = "fallback_keep_all"

        if len(selected_benigns) == 0:
            raise ValueError(
                f"Client {cid} 第二阶段没有选出任何 benign — 检查 norm² 阈值设置"
            )

        client_selections[cid] = selected_benigns

        logger.info(f"[DEBUG] Client {cid} robust sim center: {mu_sim}")
        logger.info(f"[DEBUG] Client {cid} robust sim scale: {sigma_sim}")
        logger.info(f"[DEBUG] Client {cid} first_pass: {first_pass}")
        logger.info(f"[DEBUG] Client {cid} norm screen mode: {norm_screen_mode}")
        logger.info(f"[DEBUG] Client {cid} robust update norm center: {mu_norm}")
        logger.info(f"[DEBUG] Client {cid} robust update norm scale: {sigma_norm}")
        logger.info(f"[DEBUG] Client {cid} update norm min: {min_norm}")
        logger.info(f"[DEBUG] Client {cid} update norm max: {max_norm}")
        logger.info(f"[DEBUG] Client {cid} norm spread ratio: {spread_ratio}")
        logger.info(f"[DEBUG] Client {cid} norm max gap ratio: {max_gap_ratio}")
        logger.info(f"[DEBUG] Client {cid} norm upper: {upper_norm}")
        logger.info(f"[DEBUG] Client {cid} candidate norms: {candidate_norms}")
        logger.info(f"[DEBUG] Client {cid} 投票选择: {selected_benigns}")

    # =====================================================
    # 11) 多数投票
    # =====================================================
    element_count = {}

    for _, benigns_list in client_selections.items():
        for bid in benigns_list:
            element_count[bid] = element_count.get(bid, 0) + 1

    majority_threshold = len(model_updates) // 2

    benigns = tuple([
        bid for bid, cnt in element_count.items()
        if cnt >= majority_threshold and bid not in invalid_norm_clients
    ])

    if len(benigns) == 0:
        raise ValueError("最终没有任何 benign 客户端被选中 — 检查投票逻辑")

    logger.info(f"[DEBUG] 投票计数: {element_count}")
    logger.info(f"[DEBUG] 最终多数投票结果: {benigns}")
    logger.info(f"[DEBUG] benigns num: {len(benigns)}")

    # =====================================================
    # 12) 最终安全聚合：仍然使用完整 encrypted_updates
    # =====================================================
    total_size = sum(data_size[bid] for bid in benigns)

    fusion_weights = {
        bid: data_size[bid] / total_size
        for bid in benigns
    }

    logger.info(f"[DEBUG] fusion weights: {fusion_weights}")
    logger.info(f"[DEBUG] fusion weight sum: {sum(fusion_weights.values())}")

    first_bid = benigns[0]
    fused_enc_state = {}

    for k in encrypted_updates[first_bid].keys():

        if meta[k]["is_float"]:
            num_chunks = len(encrypted_updates[first_bid][k])
            fused_chunks = []

            for chunk_idx in range(num_chunks):
                fused_chunk = (
                    encrypted_updates[first_bid][k][chunk_idx]
                    * fusion_weights[first_bid]
                )

                for bid in benigns[1:]:
                    fused_chunk += (
                        encrypted_updates[bid][k][chunk_idx]
                        * fusion_weights[bid]
                    )

                fused_chunks.append(fused_chunk)

            fused_enc_state[k] = fused_chunks

        else:
            # 非浮点 buffer，例如 BatchNorm num_batches_tracked
            # 不能做加权平均，直接复制第一个 benign 的值
            fused_enc_state[k] = encrypted_updates[first_bid][k].clone()

    # =====================================================
    # 13) 解密并返回聚合模型 state_dict
    # =====================================================
    fused_params = decrypt_encrypted_state_dict(
        fused_enc_state=fused_enc_state,
        meta=meta,
        global_model=global_model,
    )

    # =====================================================
    # 14) 检查 NaN / Inf
    # =====================================================
    for k, v in fused_params.items():
        if torch.is_floating_point(v):
            if torch.isnan(v).any() or torch.isinf(v).any():
                raise ValueError(f"NaN/Inf detected in fused parameter: {k}")

    return fused_params



















def encrypt_full_state_dict(
    model_updates: Dict[int, torch.nn.Module],
) -> Dict[int, Dict[str, ts.CKKSVector]]:
    """
    Encrypt floating-point tensors in state_dict.

    Compared with encrypting only model.parameters(), this includes BatchNorm
    buffers such as running_mean and running_var, which are important for
    CIFAR-10 models.
    """
    encrypted_updates = {}

    for client_id, model in model_updates.items():
        encrypted_state = {}

        for key, value in model.state_dict().items():
            if torch.is_floating_point(value):
                flat_value = value.detach().view(-1).cpu().tolist()
                encrypted_state[key] = ts.ckks_vector(context_ckks, flat_value)

        encrypted_updates[client_id] = encrypted_state

    return encrypted_updates


def decrypt_full_state_dict(
    fused_enc_state: Dict[str, ts.CKKSVector],
    global_model: torch.nn.Module,
    model_updates: Dict[int, torch.nn.Module],
    benigns,
) -> Dict[str, torch.Tensor]:
    """
    Decrypt encrypted floating-point state_dict tensors and reconstruct a full
    state_dict. Non-floating buffers, such as num_batches_tracked, are copied
    from the first selected benign client.
    """
    benigns = list(benigns)
    first_benign_state = model_updates[benigns[0]].state_dict()
    global_state = global_model.state_dict()

    fused_state = {}

    for key, global_value in global_state.items():
        if torch.is_floating_point(global_value):
            if key in fused_enc_state:
                decrypted_value = fused_enc_state[key].decrypt()
                fused_tensor = torch.tensor(
                    decrypted_value,
                    dtype=global_value.dtype,
                    device=global_value.device,
                ).view_as(global_value)

                fused_state[key] = fused_tensor.clone()
            else:
                fused_state[key] = global_value.clone()
        else:
            # Non-floating buffers, e.g., BatchNorm num_batches_tracked.
            # Do not average integer tensors.
            if key in first_benign_state:
                fused_state[key] = first_benign_state[key].clone()
            else:
                fused_state[key] = global_value.clone()

    return fused_state


def fusion_dual_defense(
    global_model: torch.nn.Module,
    model_updates: Dict[int, torch.nn.Module],
    data_size: Dict[int, int],
    similarity_threshold: float = None,
    epsilon: float = None,
) -> Dict[str, torch.Tensor]:
    # simulate the hyper guard defense (privacy-preserving robust aggregation)

    # 1) each client generates pre-preprocessed update
    global_last_layer = list(global_model.parameters())[-2].view(-1)
    last_layers = {
        client_id: list(model.parameters())[-2].view(-1)
        for client_id, model in model_updates.items()
    }

    mormalized_global = global_last_layer / torch.norm(global_last_layer)
    normalized_locals = {
        client_id: last_layer / torch.norm(last_layer)
        for client_id, last_layer in last_layers.items()
    }

    # 2) encrypt and send to the fusion server
    encrypted_global = ts.ckks_vector(
        context_ckks, mormalized_global.flatten().tolist()
    )

    encrypted_locals = {
        client_id: ts.ckks_vector(context_ckks, normalized_local.flatten().tolist())
        for client_id, normalized_local in normalized_locals.items()
    }

    # Important modification:
    # The original code only encrypted model.parameters().
    # Here we encrypt the floating-point tensors in the full state_dict,
    # including BatchNorm running_mean and running_var.
    encrypted_updates = encrypt_full_state_dict(model_updates)

    if epsilon is not None and isinstance(epsilon, float):
        gaussian_nosie = get_gaussian_noise(
            1, epsilon=epsilon, delta=1.0 / encrypted_global.size(), sensitivity=1
        )
        encrypted_global = (
            encrypted_global + gaussian_nosie.tolist() * encrypted_global.size()
        )

    encrypted_scores = {
        client_id: encrypted_local.dot(encrypted_global)
        for client_id, encrypted_local in encrypted_locals.items()
    }

    # 4) each client decrypt the scores and send back the benigns for validation
    client_selections = {}

    for client_id in model_updates.keys():
        scores = {
            client_id: np.abs(encrypted_score.decrypt())
            for client_id, encrypted_score in encrypted_scores.items()
        }

        logger.debug(f"client {client_id} scores: {scores}")

        min_score = np.min(list(scores.values()))
        max_score = np.max(list(scores.values()))
        diff_score = max_score - min_score

        scores_norm = {
            client_id: (score - min_score) / diff_score
            for client_id, score in scores.items()
        }

        logger.debug(f"client {client_id} norm scores: {scores_norm}")

        if similarity_threshold is None:
            similarity_threshold = np.mean(list(scores_norm.values()))

        logger.debug(f"client {client_id} similarity threshold: {similarity_threshold}")

        selected_benigns = [
            id for id, score in scores_norm.items() if score >= similarity_threshold
        ]

        logger.info(f"client {client_id} selected fusion benigns: {selected_benigns}")

        if len(selected_benigns) == 0:
            raise ValueError("No models are considered benign based on the threshold.")

        client_selections[client_id] = selected_benigns

    # 5) server counts and find the majority beningn selections
    count = {}

    for _, benigns in client_selections.items():
        _tuple = tuple(benigns)
        if _tuple in count:
            count[_tuple] += 1
        else:
            count[_tuple] = 1

    benigns = None
    max_count = 0

    for _benigns, _cnt in count.items():
        if _cnt > max_count:
            max_count = _cnt
            benigns = _benigns

    # 6) final secure aggregation
    logger.debug(f"final fusion benigns: {benigns}")

    total_size = sum(data_size[benign_id] for benign_id in benigns)

    # Important modification:
    # Aggregate encrypted full state_dict floating tensors instead of only
    # model.parameters().
    fused_enc_state = {}

    first_benign_state = model_updates[benigns[0]].state_dict()

    for key, value in first_benign_state.items():
        if not torch.is_floating_point(value):
            continue

        fused_enc_param = None

        for benign_id in benigns:
            fusion_weight = data_size[benign_id] / total_size
            weighted_enc_param = encrypted_updates[benign_id][key] * fusion_weight

            if fused_enc_param is None:
                fused_enc_param = weighted_enc_param
            else:
                fused_enc_param = fused_enc_param + weighted_enc_param

        fused_enc_state[key] = fused_enc_param

    # 7) send to client for decryption
    fused_params = decrypt_full_state_dict(
        fused_enc_state=fused_enc_state,
        global_model=global_model,
        model_updates=model_updates,
        benigns=benigns,
    )

    return fused_params