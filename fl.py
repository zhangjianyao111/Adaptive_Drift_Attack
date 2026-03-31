from abc import ABC, abstractmethod
import time
import copy
import random
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from utils.util_sys import get_available_device, intersection_of_lists
from utils.util_data import get_client_data_loader
from utils.util_data import get_global_test_data_loader
from utils.util_model import get_client_model
from utils.util_model import (
    ipm_attack_craft_model,
    scaling_attack,
    alie_attack,
    front_layer_attack,
    gradient_shift_attack,
    badnets,
    pgd_attack,
    low_rank_attack,
)
from utils.util_model import get_server_model
from utils.util_fusion import (
    fusion_avg,
    fusion_clipping_median,
    fusion_cos_defense,
    fusion_fedavg,
    fusion_krum,
    fusion_median,
    fusion_trimmed_mean,
    fusion_dual_defense,
    fusion_dual_defense2,
    fusion_dual_defense3,
    drift_defense,
)
from utils.util_logger import logger
from utils.backdoor_dataset import BackdoorDataset
from utils.util_data import LabelFlipDataset


    # ============================================
    # 这里放 compute_benign_subspace
    # ============================================
def compute_benign_subspace(client_updates, k=5):
        """
        client_updates: list of flattened benign updates (each is a 1D tensor)
        k: number of principal components to keep
        """
        X = torch.stack(client_updates)  # shape: [num_clients, dim]
        X = X - X.mean(dim=0, keepdim=True)

        # PCA via SVD
        U, S, Vt = torch.linalg.svd(X, full_matrices=False)
        subspace = Vt[:k]  # top-k principal directions
        return subspace  # shape: [k, dim]

class SimulationFL(ABC):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.device = config.get("device", None)

        self.num_clients = config.get("num_clients", 5)
        self.dataset = config.get("dataset", "mnist")
        self.fusion = config.get("fusion", "fedavg")
        self.partion_type = config.get("partition_type", "noniid")
        self.partion_dirichlet_beta = config.get("partition_dirichlet_beta", 0.25)
        self.dir_data = config.get("dir_data", "./data/")

        self.training_round = config.get("training_round", 10)
        self.local_epochs = config.get("local_epochs", 1)
        self.optimizer = config.get("optimizer", "sgd")
        self.learning_rate = config.get("learning_rate", 0.01)
        self.batch_size = config.get("batch_size", 64)
        self.regularization = config.get("regularization", 1e-5)

        self.attacker_ratio = config.get("attacker_ratio", 0.0)
        self.attacker_strategy = config.get("attacker_strategy", None)
        self.attacker_list = []
        self.attack_start_round = config.get("attack_start_round", -1)
        self.epsilon = config.get("epsilon", None)

        self.metrics = {}
        self.tensorboard = config.get("tensorboard", None)

        self.benign_subspace = None
        self.worst_direction = None
        self.attack_radius = config.get("attack_radius", 1.0)
 # ===== 每个客户端自己的历史更新，用于 local trajectory PCA =====
        self.client_update_history = {}   # key: client_id, value: list[1D tensor]

        # =========================
        # number of classes (for loss / attack / defense)
        # =========================
        if self.dataset in ["mnist", "fmnist", "cifar10", "svhn"]:
            self.num_classes = 10
        elif self.dataset in ["cifar100"]:
            self.num_classes = 100
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")



    def init_seed(self) -> None:
        if self.seed is not None and self.seed > 0:
            logger.info("setting up the seed as {}".format(self.seed))
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
            random.seed(self.seed)
        else:
            logger.info("no seed is set")

    def init_data(self) -> None:
        self.client_data_loader = get_client_data_loader(
            self.dataset,
            self.dir_data,
            self.num_clients,
            self.partion_type,
            self.partion_dirichlet_beta,
            self.batch_size,


        )
        self.server_test_data_loader = get_global_test_data_loader(
            self.dataset, self.dir_data, self.batch_size
        )

    def init_model(self) -> None:
        self.client_model = get_client_model(
            self.dataset, self.num_clients, self.device
        )
        self.server_model = get_server_model(self.dataset, self.device)


    #每轮参与训练的客户端数量
    def init_client_per_round(self) -> None:
        num_client_per_round = min(self.num_clients, 10)   #修改此处就能修改最终参与训练的客户端数量
        client_list_all = [i for i in range(self.num_clients)]
        round_client_list = []
        if num_client_per_round != self.num_clients:
            for _ in range(self.training_round):
                _client_list = random.sample(client_list_all, num_client_per_round)
                _client_list.sort()
                round_client_list.append(_client_list)
        else:
            for _ in range(self.training_round):
                round_client_list.append(client_list_all)
        self.round_client_list = round_client_list

    def init_attacker(self) -> None:
        if (
            self.attacker_strategy is not None
            and self.attacker_strategy != "none"
            and self.attacker_ratio > 0
        ):
            logger.info(
                "attacker env is set  with strategy: {} and ratio: {}".format(
                    self.attacker_strategy, self.attacker_ratio
                )
            )
            size_attackers = int(self.attacker_ratio * self.num_clients)
            self.attacker_list = random.sample(range(self.num_clients), size_attackers)
            logger.info("attacker list: {}".format(self.attacker_list))

    def init_device(self) -> None:
        self.device = get_available_device()

    def start(self):
        self.init_device()
        self.init_attacker()
        self.init_client_per_round()
        self.init_data()
        self.init_model()  

        logger.info("start the FL simulation")

        time_start = time.perf_counter()

        for _round_idx in range(self.training_round):
            logger.info(f"start training round {_round_idx}")

            # 保存上一轮的全局模型（用于计算真正的更新方向）
            prev_global_state = copy.deepcopy(self.server_model.state_dict())
            self.prev_global_model = copy.deepcopy(self.server_model)


            self.metrics[_round_idx] = {"time": None, "parties": {}, "server": {}}

            # simulate query each client
            server_model_params = self.server_model.state_dict()
            round_client_list = self.round_client_list[_round_idx]
            round_client_models = {
                pid: self.client_model[pid] for pid in round_client_list
            }

            last_round_attackers = (
                intersection_of_lists(
                    self.round_client_list[_round_idx - 1], self.attacker_list
                )
                if _round_idx >= 1
                else intersection_of_lists(
                    self.round_client_list[_round_idx], self.attacker_list
                )
            )
            for _pid, _model in round_client_models.items():
                # part of hyper guard defense mechanism
                if (
                    _pid not in last_round_attackers
                    and _round_idx >= self.attack_start_round
                    and (
                        self.attacker_strategy.startswith("model_poisoning")
                        and self.attacker_strategy != "model_poisoning_ipm"
                    )
                    and self.fusion == "dual_defense"
                ):
                    fused_params = copy.deepcopy(server_model_params)
                    for param_key in server_model_params.keys():
                        fused_params[param_key] = torch.clamp(
                            fused_params[param_key], -0.2, 0.2
                        )
                    _model.load_state_dict(fused_params)
                else:
                    _model.load_state_dict(server_model_params)

            if (
                self.attacker_strategy is not None
                and self.attacker_strategy != "none"
                and self.attacker_ratio > 0
                and self.attack_start_round <= _round_idx - 1
            ):
                self.attacker_list = random.sample(
                    round_client_list, int(self.attacker_ratio * len(round_client_list))
                )
                logger.info(f"round {_round_idx} attackers: {self.attacker_list}")
            else:
                logger.info(f"no attack at the round {_round_idx}")

            # simulate local training in parallel
            model_dict = {}
            for _client_id in round_client_list:
                logger.info(f"start client {_client_id} training")
                model_client, eval_metrics = self.client_local_train(
                    _round_idx, _client_id, round_client_models[_client_id]
                )
                model_dict[_client_id] = model_client
                logger.info(f"end client {_client_id} training")

                # RECORD PARTY METRICS
                logger.info(f"client {_client_id} evaluation metrics: {eval_metrics}")
                self.metrics[_round_idx]["parties"][_client_id] = eval_metrics

            # ====== 只做一件事：记录每个客户端本轮的更新，供“自己以后做 PCA”用 ======
            for pid in round_client_list:
                new_state = model_dict[pid].state_dict()
                delta = []
                for k in new_state.keys():
                    delta.append((new_state[k] - prev_global_state[k]).flatten())
                delta_flat = torch.cat(delta)

                if pid not in self.client_update_history:
                    self.client_update_history[pid] = []
                self.client_update_history[pid].append(delta_flat.detach().cpu())

            # simulate aggregation
            aggregated_params = self.aggregate_model(_round_idx, model_dict)
            self.server_model.load_state_dict(aggregated_params)

            # RECORD GLOBAL METRICS
            criterion = nn.CrossEntropyLoss().to(self.device)
            _, _test_acc = self.model_evaluate(
                self.server_model, self.server_test_data_loader, criterion
            )
            self.metrics[_round_idx]["server"]["test_acc"] = _test_acc
            logger.info(f"global side -  test accuracy: {_test_acc}")

            # ===== 计算后门攻击成功率 ASR（仅在使用 badnets 策略时启用）=====
            if self.attacker_strategy == "badnets":
                from torch.utils.data import DataLoader

                clean_test_dataset = self.server_test_data_loader.dataset

                # ===== BadNets 测试参数（与训练保持一致）=====
                TARGET_LABEL = 1
                TRIGGER_SIZE = 3
                # ============================================
                backdoor_testset = BackdoorDataset(
                    clean_test_dataset,
                    target_label=TARGET_LABEL,
                    poison_ratio=1.0,   # 测试时所有样本都加触发器，不要修改，保持1.0
                    trigger_size=TRIGGER_SIZE,
                )
                backdoor_test_loader = DataLoader(
                    backdoor_testset,
                    batch_size=self.batch_size,
                    shuffle=False,
                )

                _, asr = self.model_evaluate(self.server_model, backdoor_test_loader, criterion)
                self.metrics[_round_idx]["server"]["asr"] = asr
                logger.info(f"global side - test accuracy: {_test_acc}, ASR: {asr}")
            # ============================================================

            self.tensorboard.add_scalar(
                "{}-{} - Server Test Acc".format(
                    self.dataset,
                    self.fusion,
                ),
                _test_acc,
                _round_idx,
            )
            self.tensorboard.flush()

            time_round_end = time.perf_counter()
            self.metrics[_round_idx]["time"] = time_round_end - time_start

        logger.info("end the FL simulation")
        logger.info("summarization - simulation metrics: {}".format(self.metrics))
        self.tensorboard.close()

    def client_local_train(
        self, round_idx: int, client_id: int, client_model: nn.Module
    ) -> None:

        logger.info(f"client {client_id} start local training ...")
        train_data_loader, test_data_loader = self.client_data_loader[client_id]

    # ===== BadNets 后门攻击（数据层）=====
        if (
            self.attacker_strategy == "badnets"
            and client_id in self.attacker_list
            and round_idx >= self.attack_start_round
        ):
            logger.warning(f"[ATTACKER] client {client_id} using BadNets backdoor attack")

            # ====== 所有 BadNets 参数都写在这里 ======
            TARGET_LABEL = 1          # ★ 目标标签（改这里）
            POISON_RATIO = 0.3        # ★ 投毒比例（改这里）
            TRIGGER_SIZE = 3          # ★ 触发器大小（改这里）
            # =========================================

            base_dataset = train_data_loader.dataset  # 原始本地训练集（可能是 Subset）

            poisoned_dataset = BackdoorDataset(
                base_dataset,
                target_label=TARGET_LABEL,  # 统一改成目标标签
                poison_ratio=POISON_RATIO,                # 你可以调，比如 0.1 / 0.2 / 0.3
                trigger_size=TRIGGER_SIZE                  # 触发器大小
            )
            img, label = poisoned_dataset[0]
            print("Trigger region:")
            print(img[:, -5:, -5:])

            print("poison indices example:", list(poisoned_dataset.poison_indices)[:10])

            train_data_loader = torch.utils.data.DataLoader(
                poisoned_dataset,
                batch_size=train_data_loader.batch_size,
                shuffle=True,
                num_workers=train_data_loader.num_workers,
                drop_last=False,
            )
        else:
            logger.info(f"[BENIGN] client {client_id} clean training")
        # ===================================

 # ===== 标签攻击（Dataset 层）=====
        if (
            self.attacker_strategy == "label_flipping"
            and client_id in self.attacker_list
            and round_idx >= self.attack_start_round
        ):
            base_dataset = train_data_loader.dataset

            poisoned_dataset = LabelFlipDataset(
                base_dataset=base_dataset,
                num_classes=10,      # MNIST 
                flip_ratio=0.40,      #翻转程度  
                mode="cyclic",
                target_label=0,
                source_label=7, # 源类别 target_label=1, # 目标类别
                dynamic=True, # 每次取样动态翻转
                seed=round_idx
            )

            train_data_loader = torch.utils.data.DataLoader(
                poisoned_dataset,
                batch_size=train_data_loader.batch_size,
                shuffle=True,
                num_workers=train_data_loader.num_workers,
                drop_last=False
            )
        # ================================

        model = client_model.to(self.device)
        model.train()
        criterion = nn.CrossEntropyLoss().to(self.device)
        optimizer = self.get_optimizer(model)      
        
        for _epoch in range(self.local_epochs):
            train_loss_lst = []
            epoch_correct = 0
            epoch_total = 0

            _size_total_data = len(train_data_loader.dataset)
            _size_batch = len(train_data_loader)

            for _batch_idx, (_data, _target) in enumerate(train_data_loader):
                data, target = _data.to(self.device), _target.to(self.device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                

                optimizer.step()

                train_loss_lst.append(loss.item())
                _, predicted = torch.max(output.data, 1)
                epoch_total += target.size(0)
                epoch_correct += (predicted == target.data).sum().item()

            epoch_train_acc = epoch_correct / epoch_total * 100
            epoch_avg_loss = np.mean(train_loss_lst)


        if (
            self.attacker_strategy == "model_poisoning_ipm"
            and client_id in self.attacker_list
            and round_idx >= self.attack_start_round
        ):
            logger.info(f"client {client_id} is attacker, start poisoning model")
            crafted_model = ipm_attack_craft_model(
                self.server_model.to(self.device), model.to(self.device)
            )
            _, _test_acc = self.model_evaluate(
                crafted_model, test_data_loader, criterion
            )
            _train_loss, _ = self.model_evaluate(
                crafted_model, train_data_loader, criterion
            )
            return crafted_model, {"train_loss": _train_loss, "test_acc": _test_acc}
        elif (
            self.attacker_strategy == "model_poisoning_scaling"
            and client_id in self.attacker_list
            and round_idx >= self.attack_start_round
        ):
            logger.info(f"client {client_id} is attacker, start poisoning model")
            crafted_model = scaling_attack(model.to(self.device))
            _, _test_acc = self.model_evaluate(
                crafted_model, test_data_loader, criterion
            )
            _train_loss, _ = self.model_evaluate(
                crafted_model, train_data_loader, criterion
            )
            return crafted_model, {"train_loss": _train_loss, "test_acc": _test_acc}
        elif (
            self.attacker_strategy == "model_poisoning_alie"
            and client_id in self.attacker_list
            and round_idx >= self.attack_start_round
        ):
            logger.info(f"client {client_id} is attacker, start poisoning model")
            crafted_model = alie_attack(model.to(self.device))
            _, _test_acc = self.model_evaluate(
                crafted_model, test_data_loader, criterion
            )
            _train_loss, _ = self.model_evaluate(
                crafted_model, train_data_loader, criterion
            )
            return crafted_model, {"train_loss": _train_loss, "test_acc": _test_acc}
        elif (
            self.attacker_strategy == "front_layer_attack"
            and client_id in self.attacker_list
            and round_idx >= self.attack_start_round
            ):
            logger.info(f"client {client_id} is attacker, start stealth front-layer attack")
            old_model = copy.deepcopy(self.server_model).to(self.device)   # 训练前模型
            new_model = model.to(self.device)                              # 客户端训练后的模型
            crafted_model = front_layer_attack(old_model=old_model,new_model=new_model,global_model=self.server_model)
            _, _test_acc = self.model_evaluate(crafted_model, test_data_loader, criterion)
            _train_loss, _ = self.model_evaluate(crafted_model, train_data_loader, criterion)
            return crafted_model, {"train_loss": _train_loss, "test_acc": _test_acc}
        elif (
            self.attacker_strategy == "gradient_shift"
            and client_id in self.attacker_list
            and round_idx >= self.attack_start_round
        ):
            logger.info(f"client {client_id} is attacker, start gradient shifting attack")
            crafted_model = gradient_shift_attack(
                model.to(self.device),
                self.server_model.to(self.device),   # 需要传入上一轮的全局模型 
                shift_scale=0.03,
            )
            _, _test_acc = self.model_evaluate(
                crafted_model, test_data_loader, criterion
            )
            _train_loss, _ = self.model_evaluate(
                crafted_model, train_data_loader, criterion
            )
            return crafted_model, {"train_loss": _train_loss, "test_acc": _test_acc}
        elif (
            self.attacker_strategy == "badnets"
            and client_id in self.attacker_list
            and round_idx >= self.attack_start_round
        ):
            logger.warning(f"[ATTACKER] client {client_id} using BadNets + model poisoning")

            old_model = copy.deepcopy(self.server_model).to(self.device)
            new_model = model.to(self.device)

            crafted_model = badnets(
                old_model=old_model,
                new_model=new_model,
                global_model=self.server_model,
                align_ratio=0.98,   # ★ 可调
            )

            _test_loss, _test_acc = self.model_evaluate(crafted_model, test_data_loader, criterion)
            _train_loss, _ = self.model_evaluate(crafted_model, train_data_loader, criterion)

            return crafted_model, {"train_loss": _train_loss, "test_acc": _test_acc}
        elif (
            self.attacker_strategy == "pgd_attack"
            and client_id in self.attacker_list
            and round_idx >= self.attack_start_round
        ):
            logger.info(f"client {client_id} is attacker, start PGD-similarity model poisoning")

         
            new_model = model.to(self.device)
            old_model = copy.deepcopy(self.server_model).to(self.device)
            global_model = self.server_model.to(self.device)

            crafted_model = pgd_attack(
                old_model=old_model,
                new_model=new_model,
                global_model=global_model,
                step_size=0.1,
                pgd_steps=5,
                target_cos=0.95,
            )

            _test_loss, _test_acc = self.model_evaluate(crafted_model, test_data_loader, criterion)
            _train_loss, _ = self.model_evaluate(crafted_model, train_data_loader, criterion)
            return crafted_model, {"train_loss": _train_loss, "test_acc": _test_acc}
        elif (
            self.attacker_strategy == "low_rank_attack"
            and client_id in self.attacker_list
            and round_idx >= self.attack_start_round
        ):
            logger.info(f"client {client_id} is attacker, start low-rank subspace poisoning")

            # ========== 1) 先确保有足够的历史更新 ==========
            history = self.client_update_history.get(client_id, [])
            logger.info(f"[ATTACKER {client_id}] history length = {len(history)}")
            if len(history) < 2:
                logger.warning(f"[ATTACKER {client_id}] history too short ({len(history)}), fallback to benign update")
                crafted_model = model.to(self.device)
            else:
                # ========== 2) 用自己的历史更新做 PCA（local trajectory PCA）==========
                local_updates = [h.to(self.device) for h in history]
                local_subspace = compute_benign_subspace(local_updates, k=min(10, len(local_updates)))

                # 打印 PCA 奇异值（能量分布）
                U, S, Vt = torch.linalg.svd(torch.stack(local_updates) - torch.stack(local_updates).mean(0), full_matrices=False)
                logger.info(f"[ATTACKER {client_id}] PCA singular values: {S[:5].cpu().numpy()}")


                # ========== 3) 用“上一轮全局模型 + 本地数据”估计一个梯度方向 ==========
                approx_model = copy.deepcopy(self.prev_global_model).to(self.device)
                approx_model.train()
                approx_model.zero_grad()

                criterion = nn.CrossEntropyLoss().to(self.device)

                # 取5批本地数据来近似 global loss 的梯度
                data_iter = iter(train_data_loader)
                g_accum = None
                for _ in range(5):
                    try:
                        data_batch, target_batch = next(data_iter)
                    except StopIteration:
                        data_iter = iter(train_data_loader)
                        data_batch, target_batch = next(data_iter)

                    data_batch = data_batch.to(self.device)
                    target_batch = target_batch.to(self.device)

                    output = approx_model(data_batch)
                    loss = criterion(output, target_batch)
                    loss.backward()

                    grads = []
                    for p in approx_model.parameters():
                        if p.grad is not None:
                            grads.append(p.grad.view(-1))
                    g = torch.cat(grads).detach()

                    if g_accum is None:
                        g_accum = g
                    else:
                        g_accum += g
                    approx_model.zero_grad()
                g_local = g_accum / 5  
                
                logger.info(f"[ATTACKER {client_id}] g_local norm = {g_local.norm().item():.4f}")

                # ========== 4) 把梯度投影到 local PCA 子空间，得到局部最坏方向 ==========
                B = local_subspace.to(self.device)        # [k, dim]
                coeff = torch.matmul(B, g_local)          # [k]
                #proj = torch.matmul(coeff, B)             # [dim]
                # -------- 新做法：只选取最优的3个方向 --------
                topk = min(5, coeff.size(0))
                top_indices = torch.topk(torch.abs(coeff), k=topk).indices
                proj = torch.matmul(coeff[top_indices], B[top_indices])  # [dim]
                logger.info(f"[ATTACKER {client_id}] projected grad norm = {proj.norm().item():.4f}")

                if proj.norm() < 1e-12:
                    logger.warning(f"[ATTACKER {client_id}] projected grad ~ 0, fallback to benign update")
                    crafted_model = model.to(self.device)
                else:
                    #R = self.attack_radius 
                    # 使用当前近似梯度的 norm 作为参考
                    benign_like_norm = g_local.norm().item()
                    alpha = 1.1  # 可调节，范围在 0.5–2 之间
                    R = benign_like_norm * alpha
                    
                    logger.info(f"[ATTACKER {client_id}] benign_like_norm = {benign_like_norm:.4f}, R = {R:.4f}")
                    worst_direction_local = proj / proj.norm() * R
                    
                    logger.info(f"[ATTACKER {client_id}] worst_direction_local norm = {worst_direction_local.norm().item():.4f}")

                    # ========== 5) 沿着局部最坏方向，从上一轮全局模型出发构造 poisoned model ==========
                    crafted_model = low_rank_attack(
                        global_model=self.prev_global_model.to(self.device), # 以上一轮全局模型为基点
                        worst_direction=worst_direction_local
                    )

            # ====== 评估 ======
            _test_loss, _test_acc = self.model_evaluate(
                crafted_model, test_data_loader, criterion
            )
            _train_loss, _ = self.model_evaluate(
                crafted_model, train_data_loader, criterion
            )

            return crafted_model, {
                "train_loss": _train_loss,
                "test_acc": _test_acc,
            }

        else:
            _test_loss, _test_acc = self.model_evaluate(
                model, test_data_loader, criterion
            )
            _train_loss, _train_acc = self.model_evaluate(
                model, train_data_loader, criterion
            )
            return model, {"train_loss": _train_loss, "test_acc": _test_acc}

    def _batch_records_debug(
        self,
        epoch: int,
        batch_idx: int,
        size_total_data: int,
        size_data: int,
        size_batch: int,
        loss: Any,
    ) -> None:
        if batch_idx % 10 == 0:
            logger.debug(
                "train epoch: {} [{}/{} ({:.0f}%)]\t training loss: {:.6f}".format(
                    epoch,
                    batch_idx * size_data,
                    size_total_data,
                    100.0 * batch_idx / size_batch,
                    loss.item(),
                )
            )

    def get_optimizer(self, model: nn.Module) -> optim.Optimizer:
        if self.optimizer == "adam":
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=self.learning_rate,
                weight_decay=self.regularization,
            )
        elif self.optimizer == "amsgrad":
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=self.learning_rate,
                weight_decay=self.regularization,
                amsgrad=True,
            )
        elif self.optimizer == "sgd":
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.regularization,
            )
        return optimizer

    def model_evaluate(
        self,
        model: nn.Module,
        data_loader: data.DataLoader,
        criterion: nn.CrossEntropyLoss,
    ) -> tuple:
        model.eval()

        loss = 0
        correct = 0
        with torch.no_grad():
            model.to(self.device)
            for _data, _targets in data_loader:
                data, targets = _data.to(self.device), _targets.to(self.device)
                outputs = model(data)
                loss += criterion(outputs, targets).item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == targets).sum().item()

        loss /= len(data_loader)
        accuracy = 100.0 * correct / len(data_loader.dataset)
        return loss, accuracy

    def aggregate_model(self, round_idx: int, model_updates: dict) -> Dict[str, Any]:
        logger.info("start model aggregation...fusion method: {}".format(self.fusion))

        if self.fusion == "average":
            average_params = fusion_avg(model_updates)
            return average_params
        elif self.fusion == "fedavg":
            data_sizes = {
                p_id: sum(len(batch[0]) for batch in self.client_data_loader[p_id][0])
                for p_id in self.round_client_list[round_idx]
            }
            logger.debug("data sizes: {}".format(data_sizes))
            weighted_avg_params = fusion_fedavg(model_updates, data_sizes)
            return weighted_avg_params
        elif self.fusion == "krum":
            # max_expected_adversaries = int(self.attacker_ratio * self.num_clients)
            max_expected_adversaries = int(self.attacker_ratio * len(model_updates))
            krum_params = fusion_krum(
                model_updates, max_expected_adversaries, self.device
            )
            return krum_params
        elif self.fusion == "median":
            median_params = fusion_median(model_updates, device=self.device)
            return median_params
        elif self.fusion == "clipping_median":
            median_clipping_params = fusion_clipping_median(
                model_updates, clipping_threshold=0.1, device=self.device
            )
            return median_clipping_params
        elif self.fusion == "trimmed_mean":
            trimmed_mean_params = fusion_trimmed_mean(
                model_updates, trimmed_ratio=0.1, device=self.device
            )
            return trimmed_mean_params
        elif self.fusion == "cos_defense":
            weighted_params = fusion_cos_defense(self.server_model, model_updates)
            return weighted_params
        elif self.fusion == "dual_defense":
            logger.info("start hyper-guard fusion with epsilon {}".format(self.epsilon))
            lst_round_attackers = intersection_of_lists(
                list(model_updates.keys()), self.attacker_list
            )
            logger.info(f"round {round_idx} attackers: {lst_round_attackers}")
            data_sizes = {
                p_id: sum(len(batch[0]) for batch in self.client_data_loader[p_id][0])
                for p_id in self.round_client_list[round_idx]
            }
            fused_params = fusion_dual_defense(
                self.server_model,
                model_updates,
                data_sizes,
                epsilon=self.epsilon,
            )
            return fused_params
        elif self.fusion == "dual_defense2":
            logger.info("start hyper-guard fusion with epsilon {}".format(self.epsilon))
            lst_round_attackers = intersection_of_lists(
                list(model_updates.keys()), self.attacker_list
            )
            logger.info(f"round {round_idx} attackers: {lst_round_attackers}")
            data_sizes = {
                p_id: sum(len(batch[0]) for batch in self.client_data_loader[p_id][0])
                for p_id in self.round_client_list[round_idx]
            }
            fused_params = fusion_dual_defense2(
                self.server_model,
                model_updates,
                data_sizes,
                epsilon=self.epsilon,
            )
            return fused_params
        elif self.fusion == "dual_defense3":
            logger.info("start hyper-guard fusion with epsilon {}".format(self.epsilon))
            lst_round_attackers = intersection_of_lists(
                list(model_updates.keys()), self.attacker_list
            )
            logger.info(f"round {round_idx} attackers: {lst_round_attackers}")
            data_sizes = {
                p_id: sum(len(batch[0]) for batch in self.client_data_loader[p_id][0])
                for p_id in self.round_client_list[round_idx]
            }
            fused_params = fusion_dual_defense3(
                self.server_model,
                model_updates,
                data_sizes,
                epsilon=self.epsilon,
            )
            return fused_params
        elif self.fusion == "drift_defense":
            logger.info("start enhanced drift defense with epsilon {}".format(self.epsilon))
            
            # 打印攻击者列表（仅标识，不泄露更新内容）
            lst_round_attackers = intersection_of_lists(
                list(model_updates.keys()), self.attacker_list
            )
            logger.info(f"round {round_idx} attackers: {lst_round_attackers}")

            # 计算每个客户端的数据量
            data_sizes = {
                p_id: sum(len(batch[0]) for batch in self.client_data_loader[p_id][0])
                for p_id in self.round_client_list[round_idx]
            }

            # 调用 drift_defense
            fused_params = drift_defense(
                self.server_model,
                model_updates,
                data_sizes,
                epsilon=self.epsilon,
            )
            # ===== 调试信息=====
            # 只打印服务器端的整体统计，不打印单个客户端的明文
            try:
                logger.info("[DEBUG] drift_defense finished.")
                logger.info(f"[DEBUG] round {round_idx} fusion method: drift_defense")
            except Exception as e:
                logger.warning(f"[DEBUG] drift_defense debug info failed: {e}")
            return fused_params
        else:
            raise ValueError("Invalid fusion method")

