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


def fusion_clipping_median(
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




def fusion_dual_defense2(
    global_model: torch.nn.Module,
    model_updates: Dict[int, torch.nn.Module],
    data_size: Dict[int, int],
    similarity_threshold: float = None,
    epsilon: float = None,
) -> Dict[str, torch.Tensor]:


    # ---------- helper ----------
    def flatten_selected(params, idx_list):
        parts = []
        for i in idx_list:
            t = params[i]
            parts.append(t.view(-1))
        return torch.cat(parts)

    # ---------- 1) 只修改这一段：提取“最后一层 + 随机一层” ----------
    global_params = list(global_model.parameters())
    total_layers = len(global_params)

    # 最后一层（weight）通常是倒数第二层 param[-2]
    last_layer_idx = total_layers - 2

    # 随机选一层（不包括最后一层）
    random_idx_candidates = list(range(0, total_layers - 2))
    random_layer_idx = random.choice(random_idx_candidates)

    selected_indices = [last_layer_idx, random_layer_idx]

    # 全局 embedding
    global_last_layer = flatten_selected(global_params, selected_indices)

    # 客户端 embedding
    last_layers = {
        cid: flatten_selected(list(m.parameters()), selected_indices)
        for cid, m in model_updates.items()
    }

    # ---------- 2) Normalize ----------
    def safe_normalize(t):
        n = torch.norm(t)
        return t / (n + 1e-12)

    global_vec = safe_normalize(global_last_layer).cpu().detach().numpy().tolist()
    local_vecs = {
        cid: safe_normalize(v).cpu().detach().numpy().tolist()
        for cid, v in last_layers.items()
    }

    # ---------- 3) TenSEAL encrypt ----------
    encrypted_global = ts.ckks_vector(context_ckks, global_vec)
    encrypted_locals = {
        cid: ts.ckks_vector(context_ckks, vec) for cid, vec in local_vecs.items()
    }

    # ---------- 4) DP Noise ----------
    if epsilon is not None and isinstance(epsilon, float):
        noise = get_gaussian_noise(
            len(global_vec),
            epsilon=epsilon,
            delta=1.0 / max(1, len(global_vec)),
            sensitivity=1,
        )
        encrypted_noise = ts.ckks_vector(context_ckks, noise.tolist())
        encrypted_global = encrypted_global + encrypted_noise

    # ---------- 5) encrypted dot product ----------
    encrypted_scores = {
        cid: enc.dot(encrypted_global) for cid, enc in encrypted_locals.items()
    }

    # ---------- 6) decrypt scores ----------
    raw_scores = {}
    for cid, enc in encrypted_scores.items():
        val = enc.decrypt()
        if isinstance(val, list):
            val = float(np.array(val).sum())
        raw_scores[cid] = abs(val)

    vals = np.array(list(raw_scores.values()), dtype=float)
    min_s, max_s = vals.min(), vals.max()
    diff = max_s - min_s

    if diff < 1e-12:
        # 当所有分数一致时，归一化为 1.0
        norm_scores = {cid: 1.0 for cid in raw_scores.keys()}
    else:
        norm_scores = {
            cid: (s - min_s) / diff for cid, s in raw_scores.items()
        }

    # 阈值为 mean
    if similarity_threshold is None:
        similarity_threshold = float(np.mean(list(norm_scores.values())))

    # 严格按照阈值选择 benign
    selected_benigns = [
        cid for cid, score in norm_scores.items() if score >= similarity_threshold
    ]

    # 这里不做任何额外兜底或替换逻辑
    logger.info(f"fusion_dual_defense2 selected benigns: {selected_benigns}")

    # ---------- 7) Encrypt all model updates ----------
    encrypted_updates = {}
    for cid, model in model_updates.items():
        flat_params = flatten_model_parameters(model)
        encrypted_updates[cid] = [ts.ckks_vector(context_ckks, p) for p in flat_params]

    client_selections = {}
    for client_id in model_updates.keys():
        scores_for_client = {
            cid: abs(encrypted_scores[cid].decrypt()) if not isinstance(encrypted_scores[cid].decrypt(), (list, tuple, np.ndarray)) else float(np.sum(encrypted_scores[cid].decrypt()))
            for cid in encrypted_scores.keys()
        }
        # normalize per-client
        s_vals = np.array(list(scores_for_client.values()), dtype=float)
        s_min, s_max = float(s_vals.min()), float(s_vals.max())
        s_diff = s_max - s_min
        if s_diff < 1e-12:
            s_norm = {cid: 1.0 for cid in scores_for_client.keys()}
        else:
            s_norm = {cid: (sc - s_min) / s_diff for cid, sc in scores_for_client.items()}

        # threshold (use same similarity_threshold)
        selected_by_client = [cid for cid, sc in s_norm.items() if sc >= similarity_threshold]
        client_selections[client_id] = selected_by_client

    # count frequency of each selected-set and pick majority set
    count = {}
    for _, benigns in client_selections.items():
        tup = tuple(sorted(benigns))
        count[tup] = count.get(tup, 0) + 1

    # pick the most common benign tuple
    if len(count) == 0:
        raise ValueError("No client selections available for majority vote.")
    benigns_tuple = max(count.items(), key=lambda x: x[1])[0]
    benigns = list(benigns_tuple)

    # ---------- 9) weighted encrypted fusion ----------
    total_size = sum(data_size[b] for b in benigns)
    first_enc = encrypted_updates[benigns[0]]

    fused_enc_params = []
    for p in first_enc:
        d = p.decrypt()
        if isinstance(d, list):
            fused_enc_params.append(ts.ckks_vector(context_ckks, [0.0] * len(d)))
        else:
            fused_enc_params.append(ts.ckks_vector(context_ckks, [0.0]))

    for bid in benigns:
        w = data_size[bid] / total_size
        for i, p in enumerate(encrypted_updates[bid]):
            fused_enc_params[i] = fused_enc_params[i] + (p * w)

    # ---------- 10) decrypt fused parameters ----------
    _params = [np.asarray(p.decrypt()).astype(float).tolist() for p in fused_enc_params]
    fused_model = load_model_from_parameters(_params, global_model)
    return fused_model.state_dict()




def compute_val_grad_last_layer_flat(
    global_model: torch.nn.Module,
    val_loader,
    device: torch.device,
    loss_fn: torch.nn.Module,
) -> torch.Tensor:
    """
    计算验证集损失对 global_model 的倒数第二个参数（你代码里用 -2 那层）的梯度，
    返回 flatten 后的一维向量。
    """
    global_model.eval()
    global_model.zero_grad(set_to_none=True)

    last_param = list(global_model.parameters())[-2]

    for x, y in val_loader:
        x = x.to(device)
        y = y.to(device)

        out = global_model(x)
        loss = loss_fn(out, y)
        loss.backward()

    g = last_param.grad
    if g is None:
        raise RuntimeError(
            "Validation gradient is None. Check model forward/loss and that param[-2] participates."
        )

    g_flat = g.detach().view(-1).clone()
    global_model.zero_grad(set_to_none=True)
    return g_flat


def fusion_dual_defense3(
    global_model: torch.nn.Module,
    model_updates: Dict[int, torch.nn.Module],
    data_size: Dict[int, int],
    similarity_threshold: float = None,
    epsilon: float = None,
    # ======== 新增：贡献度二次筛选所需（不传就不会启动二次筛选）========
    val_loader=None,
    device=None,
    loss_fn=None,
    contrib_threshold: float = None,
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
        diff_score = diff_score if diff_score != 0 else 1e-12  # 防止除0
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

    # ============== 第一轮结束：原逻辑得到 benigns ==============
    logger.debug(f"final fusion benigns: {benigns}")

    # ============================================================
    # 5.5) Contribution-based second-pass filtering (新增部分)
    #      ✅ 不删任何原逻辑
    #      ✅ 第二轮仍然全体客户端投票
    #      ✅ 改成逐人计票（per-candidate voting）
    # ============================================================
    if val_loader is not None:
        if device is None or loss_fn is None:
            raise ValueError("val_loader provided but device/loss_fn is None.")

        # (1) server computes plaintext validation gradient (same layer -2)
        g_val_last = compute_val_grad_last_layer_flat(global_model, val_loader, device, loss_fn)
        g_val_last = g_val_last / (torch.norm(g_val_last) + 1e-12)
        g_val_list = g_val_last.detach().cpu().tolist()

        candidates = list(benigns)  # 第一轮投票出来的候选 benign
        if len(candidates) == 0:
            raise ValueError("No candidates from first-round voting. Cannot run contribution gate.")

        # (2) server computes encrypted contribution scores for candidates only
        # score_i = - <g_val, (local - global)>
        # 这里 local/global 用的是你已经构造的 normalized last layer 的 CKKS 密文
        encrypted_contrib_scores = {}
        shared_noise = None
        
        if epsilon is not None and isinstance(epsilon, float):
            shared_noise = float(get_gaussian_noise(
                1, epsilon=epsilon, delta=1.0, sensitivity=1
            )[0])

        for cid in candidates:
            enc_delta = encrypted_locals[cid] - encrypted_global
            enc_score = - enc_delta.dot(g_val_list)
            if shared_noise is not None:
                enc_score = enc_score + shared_noise
            encrypted_contrib_scores[cid] = enc_score

        # (3) ALL clients vote per-candidate
        votes = {cid: 0 for cid in candidates}

        for voter_id in model_updates.keys():
            # voter decrypts all contribution scores (simulation)
            contrib_scores = {
                cid: float(enc_score.decrypt())
                for cid, enc_score in encrypted_contrib_scores.items()
            }
            logger.debug(f"client {voter_id} contrib scores: {contrib_scores}")

            # normalize to [0,1] for stable thresholding
            min_s = float(np.min(list(contrib_scores.values())))
            max_s = float(np.max(list(contrib_scores.values())))
            diff = max(max_s - min_s, 1e-12)

            contrib_scores_norm = {cid: (s - min_s) / diff for cid, s in contrib_scores.items()}

            # threshold rule: same style as your cosine gate
            if contrib_threshold is None:
                thr = float(np.mean(list(contrib_scores_norm.values())))
            else:
                thr = float(contrib_threshold)

            kept = [cid for cid, s in contrib_scores_norm.items() if s >= thr]
            logger.info(f"client {voter_id} kept by contrib gate: {kept}")

            # per-candidate voting
            for cid in kept:
                votes[cid] += 1

        # (4) majority rule on each candidate
        total_voters = len(model_updates)
        majority = total_voters // 2 + 1

        benigns2 = [cid for cid, v in votes.items() if v >= majority]

        logger.info(f"contrib per-candidate votes: {votes}")
        logger.info(f"majority threshold: {majority}/{total_voters}")
        logger.info(f"final benigns after contrib gate: {benigns2}")

        # strict: no safeguard
        if len(benigns2) == 0:
            raise ValueError(
                "Contribution second-pass removed all candidates. No benigns left for aggregation."
            )

        benigns = tuple(benigns2)

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




def drift_defense(
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
        #first_pass = [id for id, s in scores_norm.items() if s >= similarity_threshold]
        # 第一轮：相似度筛选（区间检测）
        mu_sim = np.mean(list(scores_norm.values()))
        sigma_sim = np.std(list(scores_norm.values()))
        first_pass = [
            id for id, s in scores_norm.items()
            if (mu_sim - sigma_sim) <= s <= (mu_sim + sigma_sim)
    ]

        # 第二轮：能量比例筛选
        ratios = {id: scores_norm[id] / (norm_scores[id] + 1e-8) for id in first_pass}
        mu, sigma = np.mean(list(ratios.values())), np.std(list(ratios.values()))

        def energy_ratio_ok(xid):
            return (mu - sigma) <= ratios[xid] <= (mu + sigma)

        selected_benigns = [xid for xid in first_pass if energy_ratio_ok(xid)]
        if len(selected_benigns) == 0:
            raise ValueError(f"Client {cid} 没有选出任何 benign — 检查阈值设置")
        client_selections[cid] = selected_benigns
        logger.info(f"[DEBUG] Client {cid} 投票选择: {selected_benigns}")

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



