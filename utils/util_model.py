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


def scaling_attack2(model_update: torch.nn.Module):
    params_list = list(model_update.parameters())
    num_params = len(params_list)
    for param in params_list:
        param.data *= num_params
    return model_update



def scaling_attack1(
    local_model: torch.nn.Module,
    global_model: torch.nn.Module,
    fusion: str, 
    round_client_list: list,
    attacker_list: list,
    client_data_loader: dict,
    gamma_override: float = None,
) -> torch.nn.Module:
    num_selected_clients = len(round_client_list)
    num_attackers = max(1, len(attacker_list))
    
    if gamma_override is not None:
        gamma = float(gamma_override)
    else:
        gamma = num_selected_clients / num_attackers

    attacked_model = copy.deepcopy(local_model)

    with torch.no_grad():
        global_state = global_model.state_dict()
        local_state = attacked_model.state_dict()

        for key in local_state.keys():
            delta = local_state[key] - global_state[key]
            local_state[key] = global_state[key] + gamma * delta

        attacked_model.load_state_dict(local_state)

    logger.info(
        f"[scaling_attack] gamma={gamma}, attackers={attacker_list}, selected={round_client_list}"
    )
    return attacked_model



def compute_dataset_size(loader):
    # ✅ 如果是 tuple（如 train/test）
    if isinstance(loader, (list, tuple)):
        loader = loader[0]   # 默认取 train_loader

    # ✅ 标准 DataLoader
    if hasattr(loader, "dataset"):
        return len(loader.dataset)

    # ✅ fallback（极端情况）
    total = 0
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            total += len(batch[0])
        else:
            total += len(batch)
    return total

def clip_tensor(tensor, clip_norm):
    norm = torch.norm(tensor)
    if norm > clip_norm:
        tensor = tensor * (clip_norm / norm)
    return tensor

def estimate_safe_gamma(delta, reference_norm, max_scale=10.0):
    """
    根据当前更新幅度估计一个“不会被明显检测”的 gamma
    """
    delta_norm = torch.norm(delta)
    if delta_norm == 0:
        return 1.0
    gamma = reference_norm / delta_norm
    gamma = min(gamma, max_scale)
    return gamma


def scaling_attack(
    local_model: torch.nn.Module,
    global_model: torch.nn.Module,
    fusion: str,
    round_client_list: list,
    attacker_list: list,
    client_data_loader: dict,
    gamma_override: float = None,
    clip_norm: float = None,
    noise_std: float = 0.0,
    max_scale: float = 10.0,
) -> torch.nn.Module:
    num_selected_clients = len(round_client_list)
    num_attackers = max(1, len(attacker_list))

    # 1️⃣ 基础 gamma
    if gamma_override is not None:
        base_gamma = float(gamma_override)
    else:
        total_samples = sum(compute_dataset_size(client_data_loader[pid]) for pid in round_client_list)
        malicious_samples = sum(compute_dataset_size(client_data_loader[pid]) for pid in attacker_list)
        malicious_samples = max(1, malicious_samples)
        base_gamma = total_samples / malicious_samples

    # 2️⃣ 攻击方向
    direction = -1.0 if fusion not in ["cos_defense", "dual_defense"] else +1.0

    attacked_model = copy.deepcopy(local_model)

    with torch.no_grad():
        global_state = global_model.state_dict()
        local_state = attacked_model.state_dict()

        # 3️⃣ 计算整体 delta 范数分布
        deltas = []
        for key in local_state.keys():
            deltas.append(torch.norm(local_state[key] - global_state[key]))
        median_norm = torch.median(torch.tensor(deltas))

        for key in local_state.keys():
            delta = local_state[key] - global_state[key]

            # 裁剪
            if clip_norm is not None:
                delta = clip_tensor(delta, clip_norm)

            # 自适应 gamma
            reference_norm = torch.norm(global_state[key])
            adaptive_gamma = reference_norm / (median_norm + 1e-6)
            gamma = min(base_gamma, adaptive_gamma, max_scale)

            # 噪声
            if noise_std > 0:
                delta = delta + torch.randn_like(delta) * noise_std

            # 更新
            local_state[key] = global_state[key] + direction * gamma * delta

        attacked_model.load_state_dict(local_state)

    print(f"[smart_scaling_attack] fusion={fusion}, base_gamma={base_gamma:.3f}, direction={direction}, attackers={attacker_list}")
    return attacked_model


def alie_attack(model_update: torch.nn.Module, epsilon=1.0):
    with torch.no_grad():
        for param in model_update.parameters():
            param.add_(torch.randn(param.size(), device=param.device) * epsilon)
    return model_update


def badnets(old_model, new_model, global_model, align_ratio=0.98):
    crafted_model = copy.deepcopy(new_model)
    new_state = new_model.state_dict()
    global_state = global_model.state_dict()
    crafted_state = crafted_model.state_dict()

    last_keys = [k for k in new_state.keys() if "weight" in k or "bias" in k][-2:]
    for k in last_keys:
        crafted_state[k] = (
            align_ratio * global_state[k] +
            (1 - align_ratio) * new_state[k]
        )
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


