from typing import List
import math
import copy

import torch
import numpy as np
import torch.nn.functional as F
from utils.util_logger import logger

from utils.models import ResNet18, MNISTCNN, FashionMNISTCNN


def get_client_model(dataset: str, num_parties: int, device: torch.device) -> dict:
    """
    Returns the client models based on the dataset.

    Args:
        dataset (str): The dataset used for training.
        num_parties (int): The number of parties in the federated learning system.
        device (torch.device): The device to use for the model.
    Returns:
        dict: A dictionary containing the client models.
    """

    client_models = {id: None for id in range(num_parties)}
    for client_id in range(num_parties):
        if dataset == "mnist":
            model = MNISTCNN()
        elif dataset == "fmnist":
            model = FashionMNISTCNN()
        elif dataset == "cifar10" or dataset == "svhn":
            model = ResNet18()
        else:
            raise ValueError("Invalid dataset")

        if model is not None:
            model.to(device)
        client_models[client_id] = model

    return client_models


def get_server_model(dataset: str, device: torch.device) -> torch.nn.Module:
    """
    Returns the server model based on the dataset.

    Args:
        dataset (str): The dataset used for training.
        device (torch.device): The device to use for the model.
    Returns:
        torch.nn.Module: The server model.
    """
    if dataset == "mnist":
        model = MNISTCNN()
    elif dataset == "fmnist":
        model = FashionMNISTCNN()
    elif dataset == "cifar10" or dataset == "svhn":
        model = ResNet18()
    if model is not None:
        model.to(device)

    return model


def extract_parameters(model: torch.nn.Module) -> torch.Tensor:
    params = [p.view(-1) for p in model.parameters()]
    return torch.cat(params)


def flatten_model_parameters(model: torch.nn.Module) -> List[List[float]]:
    """
    Converts each layer's parameters of a PyTorch model into a one-dimensional array format and stores them in a list.

    Args:
        model (torch.nn.Module): The PyTorch model to process.

    Returns:
        List[List[float]]: A list containing one-dimensional arrays of parameters for each layer of the model.
    """
    flattened_parameters = [
        param.data.flatten().tolist() for param in model.parameters()
    ]

    return flattened_parameters


def load_model_from_parameters(
    flatten_parameters: List[List[float]], model: torch.nn.Module
) -> torch.nn.Module:
    """
    Recovers the model from the flattened parameters.

    Args:
        flatten_parameters (List[List[float]]): The flattened parameters of the model.
        model (torch.nn.Module): The model to recover.

    Returns:
        torch.nn.Module: The recovered model.
    """
    for param, flatten_param in zip(model.parameters(), flatten_parameters):
        param.data = torch.tensor(flatten_param).view(param.data.shape)

    return model


def get_gaussian_noise(
    size: int, epsilon: float = 0.5, delta: float = 1e-5, sensitivity: float = 1
) -> np.ndarray:

    sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    noise = np.random.normal(0, sigma, size)

    return noise


def get_laplace_noise(
    size: int, epsilon: float = 0.5, sensitivity: float = 1
) -> np.ndarray:

    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale, size)
    return noise


def ipm_attack_craft_model(
    old_model, new_model, action: int = 5, b: int = -1
) -> torch.nn.Module:
    crafted_model = copy.deepcopy(old_model)

    for old_param, new_param, crafted_param in zip(
        old_model.parameters(), new_model.parameters(), crafted_model.parameters()
    ):
        weight_diff = old_param.data - new_param.data
        crafted_weight_diff = b * weight_diff * action
        crafted_param.data = old_param.data - crafted_weight_diff

    return crafted_model


def _fang_attack_compute_lambda(
    param_updates: torch.Tensor, param_global: torch.Tensor, n_attackers: int
) -> float:

    distances = []
    n_benign, d = param_updates.shape
    for update in param_updates:
        distance = torch.norm((param_updates - update), dim=1)
        distances = (
            distance[None, :]
            if not len(distances)
            else torch.cat((distances, distance[None, :]), 0)
        )

    distances[distances == 0] = 10000
    distances = torch.sort(distances, dim=1)[0]
    scores = torch.sum(distances[:, : n_benign - 2 - n_attackers], dim=1)
    min_score = torch.min(scores)
    term_1 = min_score / (
        (n_benign - n_attackers - 1) * torch.sqrt(torch.Tensor([d]))[0]
    )
    max_wre_dist = torch.max(torch.norm((param_updates - param_global), dim=1)) / (
        torch.sqrt(torch.Tensor([d]))[0]
    )

    return term_1 + max_wre_dist


def _fang_attack_multi_krum(
    param_updates: torch.Tensor, n_attackers: int, multi_k=False
):
    nusers = param_updates.shape[0]
    candidates = []
    candidate_indices = []
    remaining_updates = param_updates
    all_indices = np.arange(len(param_updates))

    while len(remaining_updates) > 2 * n_attackers + 2:
        distances = []
        for update in remaining_updates:
            distance = torch.norm((remaining_updates - update), dim=1) ** 2
            distances = (
                distance[None, :]
                if not len(distances)
                else torch.cat((distances, distance[None, :]), 0)
            )

        distances = torch.sort(distances, dim=1)[0]
        scores = torch.sum(
            distances[:, : len(remaining_updates) - 2 - n_attackers], dim=1
        )
        indices = torch.argsort(scores)[: len(remaining_updates) - 2 - n_attackers]

        candidate_indices.append(all_indices[indices[0].cpu().numpy()])
        all_indices = np.delete(all_indices, indices[0].cpu().numpy())
        candidates = (
            remaining_updates[indices[0]][None, :]
            if not len(candidates)
            else torch.cat((candidates, remaining_updates[indices[0]][None, :]), 0)
        )
        remaining_updates = torch.cat(
            (remaining_updates[: indices[0]], remaining_updates[indices[0] + 1 :]), 0
        )
        if not multi_k:
            break
    # print(len(remaining_updates))
    aggregate = torch.mean(candidates, dim=0)
    return aggregate, np.array(candidate_indices)


def fang_attack(
    param_updates: torch.Tensor,
    param_global: torch.Tensor,
    deviation: torch.Tensor,
    n_attackers: int,
):

    lamda = _fang_attack_compute_lambda(param_updates, param_global, n_attackers)

    threshold = 1e-5
    mal_update = []

    while lamda > threshold:
        mal_update = -lamda * deviation
        mal_updates = torch.stack([mal_update] * n_attackers)
        mal_updates = torch.cat((mal_updates, param_updates), 0)

        # print(mal_updates.shape, n_attackers)
        agg_grads, krum_candidate = _fang_attack_multi_krum(
            mal_updates, n_attackers, multi_k=False
        )
        if krum_candidate < n_attackers:
            # print('successful lamda is ', lamda)
            return mal_update
        else:
            mal_update = []

        lamda *= 0.5

    if not len(mal_update):
        mal_update = param_global - lamda * deviation

    return mal_update


def aggregation_tailored_attack(
    param_updates: torch.Tensor,
    param_global: torch.Tensor,
    n_attackers,
    dev_type="sign",
):

    if dev_type == "unit_vec":
        deviation = param_global / torch.norm(
            param_global
        )  # unit vector, dir opp to good dir
    elif dev_type == "sign":
        deviation = torch.sign(param_global)
    elif dev_type == "std":
        deviation = torch.std(param_updates, 0)

    lamda = torch.Tensor(
        [20.0]
    ).cuda()  # compute_lambda_our(all_updates, model_re, n_attackers)

    threshold_diff = 1e-5
    lamda_fail = lamda
    lamda_succ = 0

    while torch.abs(lamda_succ - lamda) > threshold_diff:
        mal_update = param_global - lamda * deviation
        mal_updates = torch.stack([mal_update] * n_attackers)
        mal_updates = torch.cat((mal_updates, param_updates), 0)

        agg_grads, krum_candidate = _fang_attack_multi_krum(
            mal_updates, n_attackers, multi_k=True
        )

        if np.sum(krum_candidate < n_attackers) == n_attackers:
            # print('successful lamda is ', lamda)
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2

        lamda_fail = lamda_fail / 2

    mal_update = param_global - lamda_succ * deviation

    return mal_update


def scaling_attack(model_update: torch.nn.Module):
    params_list = list(model_update.parameters())
    num_params = len(params_list)

    for param in params_list:
        param.data *= num_params
    return model_update


def alie_attack(model_update: torch.nn.Module, epsilon=0.1):
    with torch.no_grad():
        for param in model_update.parameters():
            param.add_(torch.randn(param.size(), device=param.device) * epsilon)
    return model_update



def cosine_sim(a, b):
    if a.norm() == 0 or b.norm() == 0:
        return torch.tensor(0.0, device=a.device)
    return F.cosine_similarity(a.view(1, -1), b.view(1, -1)).item()


def flatten_param(tensor):
    return tensor.detach().view(-1)


def front_layer_attack(
 old_model,
    new_model,
    global_model,
    config={
        "scale": 0.8,                    # 前层慢性参数漂移强度
        "attack_indices": list(range(0, 8)),         # 不包含最后一层
        "rank_ratio": 0.3,               # 低秩子空间比例
        "reuse_basis": False,              # 跨轮复用攻击方向
        "last_layer_align_ratio": 0.98,   # 最后一层 cosine 对齐强度
    }
):
    """
    Parameter-space front-layer attack for FedAvg-style aggregation.
    """

    device = next(new_model.parameters()).device
    crafted_model = copy.deepcopy(new_model)

    old_params = [p.detach().clone().to(device) for p in old_model.parameters()]
    new_params = [p.detach().clone().to(device) for p in new_model.parameters()]
    global_params = [p.detach().clone().to(device) for p in global_model.parameters()]
    crafted_params = [p for p in crafted_model.parameters()]

    num_layers = len(new_params)
    last_idx = num_layers - 1

    # ============================================================
    # 1. 初始化跨轮前层攻击子空间（低秩、可复用）
    # ============================================================
    if not hasattr(front_layer_attack, "attack_basis"):
        front_layer_attack.attack_basis = {}

    # ============================================================
    # 2. 前层：参数空间慢性漂移（FedAvg 不敏感方向）
    # ============================================================
    for idx in config["attack_indices"]:
        if idx < 0 or idx >= last_idx:
            continue

        # 参数差分
        local_delta = new_params[idx] - old_params[idx]
        global_delta = global_params[idx] - old_params[idx]

        if local_delta.norm() < 1e-8 or global_delta.norm() < 1e-8:
            continue

        # ---- Step 1: 去 global 对齐（消除 benign 主方向）----
        proj = torch.sum(local_delta * global_delta) / (
            global_delta.norm() ** 2 + 1e-8
        )
        orth_delta = local_delta - proj * global_delta
        if orth_delta.norm() < 1e-8:
            continue

        flat = orth_delta.view(-1)

        # ---- Step 2: 构建 / 复用低秩攻击子空间 ----
        if idx not in front_layer_attack.attack_basis or not config["reuse_basis"]:
            k = max(1, int(flat.numel() * config["rank_ratio"]))

            rand_dir = torch.randn_like(flat)
            rand_dir = rand_dir / (rand_dir.norm() + 1e-8)

            # 与 benign 更新正交
            proj = torch.dot(rand_dir, flat) * flat / (flat.norm()**2 + 1e-8)
            drift_dir = rand_dir - proj
            drift_dir = drift_dir / (drift_dir.norm() + 1e-8)

            # 稀疏化（低秩）
            _, topk = torch.topk(drift_dir.abs(), k)
            mask = torch.zeros_like(drift_dir)
            mask[topk] = 1.0
            drift_dir = (drift_dir * mask).detach()

            front_layer_attack.attack_basis[idx] = drift_dir.clone()

        drift_dir = front_layer_attack.attack_basis[idx].to(device)
        drift_delta = drift_dir.view_as(local_delta)

        # ---- Step 3: 参数级慢性漂移（直接拉 new 参数）----
        crafted_params[idx].data = (
            new_params[idx]
            + config["scale"] * drift_delta.to(crafted_params[idx].dtype)
        )

    # ============================================================
    # 3. 最后一层：强制 cosine 对齐（绕 similarity filter）
    # ============================================================
    new_last = new_params[last_idx]
    global_last = global_params[last_idx]

    if new_last.norm() > 1e-8 and global_last.norm() > 1e-8:
        cos = torch.sum(new_last * global_last) / (
            new_last.norm() * global_last.norm() + 1e-8
        )

        # 强制参数方向接近 global
        crafted_params[last_idx].data = (
            config["last_layer_align_ratio"] * global_last
            + (1 - config["last_layer_align_ratio"]) * new_last
        ).to(crafted_params[last_idx].dtype)

    # ============================================================
    # 4. 其余层：保持 benign 行为
    # ============================================================
    for i in range(num_layers):
        if i in config["attack_indices"] or i == last_idx:
            continue
        crafted_params[i].data = new_params[i]


    return crafted_model

'''
def front_layer_attack(
    old_model,
    new_model,
    global_model,
    config={
        "scale": 0.18,                      # 前层慢性漂移强度
        "attack_indices": [2, 3],            # ❗只攻击语义前层
        "rank_ratio": 0.15,
        "reuse_basis": True,
        "semantic_keep_ratio": 0.2,          # ❗保留 20% 语义分量
        "last_layer_scale_ratio": 0.35,       # ❗最后一层更强
    }
):
    device = next(new_model.parameters()).device
    crafted_model = copy.deepcopy(new_model)

    old_params = [p.detach().clone().to(device) for p in old_model.parameters()]
    new_params = [p.detach().clone().to(device) for p in new_model.parameters()]
    global_params = [p.detach().clone().to(device) for p in global_model.parameters()]
    crafted_params = [p for p in crafted_model.parameters()]

    updates = [
        (new_params[i] - old_params[i]).detach().clone()
        for i in range(len(new_params))
    ]

    last_idx = len(updates) - 1

    # ================= 前层：语义慢性漂移 =================
    if not hasattr(front_layer_attack, "attack_basis"):
        front_layer_attack.attack_basis = {}

    for idx in config["attack_indices"]:
        local_delta = updates[idx]
        global_delta = global_params[idx] - old_params[idx]

        if local_delta.norm() < 1e-8 or global_delta.norm() < 1e-8:
            continue

        flat = local_delta.view(-1)

        # --- 构建 / 复用方向 ---
        if idx not in front_layer_attack.attack_basis or not config["reuse_basis"]:
            k = max(1, int(flat.numel() * config["rank_ratio"]))

            rand_dir = torch.randn_like(flat)
            rand_dir = rand_dir / (rand_dir.norm() + 1e-8)

            # ❗不是完全正交，保留语义破坏
            proj = torch.dot(rand_dir, flat) * flat / (flat.norm()**2 + 1e-8)
            drift_dir = rand_dir - (1 - config["semantic_keep_ratio"]) * proj
            drift_dir = drift_dir / (drift_dir.norm() + 1e-8)

            _, topk = torch.topk(drift_dir.abs(), k)
            mask = torch.zeros_like(drift_dir)
            mask[topk] = 1.0
            drift_dir = (drift_dir * mask).detach()

            front_layer_attack.attack_basis[idx] = drift_dir.clone()

        drift_delta = front_layer_attack.attack_basis[idx].view_as(local_delta)
        updates[idx] = local_delta + config["scale"] * drift_delta

    # ================= 最后一层：沿决策边界扰动 =================
    local_delta = updates[last_idx]
    global_delta = global_params[last_idx] - old_params[last_idx]

    if local_delta.norm() > 1e-8 and global_delta.norm() > 1e-8:
        flat_l = local_delta.view(-1)
        flat_g = global_delta.view(-1)

        # ❗决策边界方向（local ⟂ global）
        proj = torch.dot(flat_l, flat_g) * flat_g / (flat_g.norm()**2 + 1e-8)
        boundary_dir = flat_l - proj
        boundary_dir = boundary_dir / (boundary_dir.norm() + 1e-8)

        updates[last_idx] = (
            local_delta
            + config["last_layer_scale_ratio"] * boundary_dir.view_as(local_delta)
        )

    # ================= 构造模型 =================
    for i in range(len(crafted_params)):
        crafted_params[i].data = (
            old_params[i] + updates[i].to(crafted_params[i].dtype)
        )

    return crafted_model, {}
'''


def gradient_shift_attack(model_update: torch.nn.Module,
                          global_model: torch.nn.Module,
                          shift_scale: float = 0.03):


    for u_param, g_param in zip(model_update.parameters(), global_model.parameters()):
        g = g_param.data.clone().to(u_param.device)
        norm = torch.norm(g)
        if norm < 1e-6:
            direction = torch.sign(g)
        else:
            direction = g / (norm + 1e-6)
        # in-place copy 确保写回
        u_param.data.copy_(g + shift_scale * direction)
    return model_update



def badnets(old_model, new_model, global_model, align_ratio=0.98):

    crafted_model = copy.deepcopy(new_model)

    new_state = new_model.state_dict()
    global_state = global_model.state_dict()
    crafted_state = crafted_model.state_dict()

    # 找到最后一层的 key（通常是 classifier.weight / classifier.bias）
    last_keys = [k for k in new_state.keys() if "weight" in k or "bias" in k][-2:]

    for k in last_keys:
        crafted_state[k] = (
            align_ratio * global_state[k] +
            (1 - align_ratio) * new_state[k]
        )

    crafted_model.load_state_dict(crafted_state)
    return crafted_model

def pgd_attack(
    old_model,
    new_model,
    global_model,
    step_size=0.1,
    pgd_steps=5,
    target_cos=0.98,
):
    crafted_model = copy.deepcopy(new_model)

    old_state = old_model.state_dict()
    new_state = new_model.state_dict()
    global_state = global_model.state_dict()
    crafted_state = crafted_model.state_dict()

    # ====== 1. 计算 benign 更新方向 ======
    delta_global = {k: global_state[k] - old_state[k] for k in new_state.keys()}
    delta_att = {k: new_state[k] - global_state[k] for k in new_state.keys()}

    # ====== 找到最后一层（你贴的逻辑） ======
    last_keys = [k for k in new_state.keys() if "weight" in k or "bias" in k][-2:]
    front_keys = [k for k in new_state.keys() if k not in last_keys]

    # ====== 工具函数 ======
    def flatten(d):
        return torch.cat([d[k].flatten() for k in d])

    def cos(a, b):
        return torch.sum(a * b) / (a.norm() * b.norm() + 1e-8)

    flat_global = flatten(delta_global)

    # ====== 2. PGD 破坏（只动前层） ======
    for _ in range(pgd_steps):
        for k in front_keys:
            noise = torch.randn_like(delta_att[k])
            delta_att[k] = delta_att[k] + step_size * noise

        # 投影到 benign 方向（保持相似度）
        flat_att = flatten(delta_att)
        if cos(flat_att, flat_global) < target_cos:
            for k in delta_att.keys():
                delta_att[k] = (
                    target_cos * delta_global[k] +
                    (1 - target_cos) * delta_att[k]
                )

    # ====== 3. 最后一层强对齐（你贴的逻辑） ======
    for k in last_keys:
        crafted_state[k] = (
            target_cos * global_state[k] +
            (1 - target_cos) * new_state[k]
        )
        delta_att[k] = crafted_state[k] - global_state[k]

    # ====== 4. 构造最终模型 ======
    for k in crafted_state.keys():
        if k not in last_keys:
            crafted_state[k] = global_state[k] + delta_att[k]

    crafted_model.load_state_dict(crafted_state)
    return crafted_model


def low_rank_attack(global_model,worst_direction):
    if worst_direction is None:
        return copy.deepcopy(global_model)

    crafted_model = copy.deepcopy(global_model).to(next(global_model.parameters()).device)

    flat = worst_direction.to(next(crafted_model.parameters()).device)
    idx = 0

    with torch.no_grad():
        for p in crafted_model.parameters():
            numel = p.numel()
            p.add_(flat[idx:idx+numel].view_as(p))
            idx += numel

    return crafted_model


