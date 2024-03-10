import torch
from torch import optim
from tqdm import tqdm
import numpy as np
from typing import List, Tuple, Union, Dict

from .interaction_utils import generate_all_masks, generate_subset_masks, generate_reverse_subset_masks, \
    generate_set_with_intersection_masks


def get_reward2Iand_mat(dim):
    '''
    The transformation matrix (containing 0, 1, -1's) from reward to and-interaction (Harsanyi)
    :param dim: the input dimension n
    :return: a matrix, with shape 2^n * 2^n
    '''
    all_masks = torch.BoolTensor(generate_all_masks(dim))
    n_masks, _ = all_masks.shape
    mat = []
    for i in range(n_masks):
        mask_S = all_masks[i]
        row = torch.zeros(n_masks)
        # ===============================================================================================
        # Note: I(S) = \sum_{L\subseteq S} (-1)^{s-l} v(L)
        mask_Ls, L_indices = generate_subset_masks(mask_S, all_masks)
        L_indices = (L_indices == True).nonzero(as_tuple=False)
        assert mask_Ls.shape[0] == L_indices.shape[0]
        row[L_indices] = torch.pow(-1., mask_S.sum() - mask_Ls.sum(dim=1)).unsqueeze(1)
        # ===============================================================================================
        mat.append(row.clone())
    mat = torch.stack(mat).float()
    return mat


def get_reward2Ior_mat(dim):
    '''
    The transformation matrix (containing 0, 1, -1's) from reward to or-interaction
    :param dim: the input dimension n
    :return: a matrix, with shape 2^n * 2^n
    '''
    all_masks = torch.BoolTensor(generate_all_masks(dim))
    n_masks, _ = all_masks.shape
    mat = []
    for i in range(n_masks):
        mask_S = all_masks[i]
        row = torch.zeros(n_masks)
        # ===============================================================================================
        # Note: I(S) = -\sum_{L\subseteq S} (-1)^{s+(n-l)-n} v(N\L) if S is not empty
        if mask_S.sum() == 0:
            row[i] = 1.
        else:

            mask_NLs, NL_indices = generate_reverse_subset_masks(mask_S, all_masks)  # L \in N. L \neqsubset S
            NL_indices = (NL_indices == True).nonzero(as_tuple=False)
            assert mask_NLs.shape[0] == NL_indices.shape[0]
            row[NL_indices] = - torch.pow(-1., mask_S.sum() + mask_NLs.sum(dim=1) + dim).unsqueeze(1)
        # ================================================================================================
        mat.append(row.clone())
    mat = torch.stack(mat).float()
    return mat


def get_Iand2reward_mat(dim):
    all_masks = torch.BoolTensor(generate_all_masks(dim))
    n_masks, _ = all_masks.shape
    mat = []
    for i in range(n_masks):
        mask_S = all_masks[i]
        row = torch.zeros(n_masks)
        # ================================================================================================
        # Note: v(S) = \sum_{L\subseteq S} I(S)
        mask_Ls, L_indices = generate_subset_masks(mask_S, all_masks)
        row[L_indices] = 1.
        # ================================================================================================
        mat.append(row.clone())
    mat = torch.stack(mat).float()
    return mat


def get_Ior2reward_mat(dim):
    all_masks = torch.BoolTensor(generate_all_masks(dim))
    n_masks, _ = all_masks.shape
    mat = []
    mask_empty = torch.zeros(dim).bool()
    _, empty_indice = generate_subset_masks(mask_empty, all_masks)
    for i in range(n_masks):
        mask_S = all_masks[i]
        row = torch.zeros(n_masks)
        # ================================================================================================
        # Note: v(S) = I(\emptyset) + \sum_{L: L\union S\neq \emptyset} I(S)
        row[empty_indice] = 1.
        mask_Ls, L_indices = generate_set_with_intersection_masks(mask_S, all_masks)
        row[L_indices] = 1.
        # ================================================================================================
        mat.append(row.clone())
    mat = torch.stack(mat).float()
    return mat


# ==============================================
#     For sparsifying and-or interactions
# ==============================================
def l1_on_given_dim(vector: torch.Tensor, indices: List) -> torch.Tensor:
    assert len(vector.shape) == 1
    strength = torch.abs(vector)
    return torch.sum(strength[indices])


def generate_ckpt_id_list(niter: int, nckpt: int) -> List:
    ckpt_id_list = list(range(niter))[::max(1, niter // nckpt)]
    # force the last iteration to be a checkpoint
    if niter - 1 not in ckpt_id_list:
        ckpt_id_list.append(niter - 1)
    return ckpt_id_list


def _train_p_q(
        rewards: torch.Tensor,
        lr: float,
        niter: int,
        qbound: Union[float, torch.Tensor],
        reward2Iand: torch.Tensor = None,
        reward2Ior: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict, Dict]:
    device = rewards.device
    n_dim = int(np.log2(rewards.numel()))
    if reward2Iand is None:
        reward2Iand = get_reward2Iand_mat(n_dim).to(device)
    if reward2Ior is None:
        reward2Ior = get_reward2Ior_mat(n_dim).to(device)

    log_lr = np.log10(lr)
    eta_list = np.logspace(log_lr, log_lr - 1, niter)

    p = torch.zeros_like(rewards).requires_grad_(True)
    q = torch.zeros_like(rewards).requires_grad_(True)
    optimizer = optim.SGD([p, q], lr=0.0, momentum=0.9)

    losses = {"loss": []}
    progresses = {"I_and": [], "I_or": []}
    ckpt_id_list = generate_ckpt_id_list(niter=niter, nckpt=20)

    pbar = tqdm(range(niter), desc="Optimizing pq", ncols=100)
    for it in pbar:
        q.data = torch.max(torch.min(q.data, qbound), -qbound)
        Iand_p = torch.matmul(reward2Iand, 0.5 * (rewards + q) + p).squeeze()
        Ior_p = torch.matmul(reward2Ior, 0.5 * (rewards + q) - p).squeeze()

        # Equation (4) in paper
        loss = torch.sum(torch.abs(Iand_p)) + torch.sum(torch.abs(Ior_p))
        losses["loss"].append(loss.item())

        if it + 1 < niter:
            optimizer.zero_grad()
            optimizer.param_groups[0]["lr"] = eta_list[it]
            loss.backward()
            optimizer.step()

        if it in ckpt_id_list:
            progresses["I_and"].append(Iand_p.detach().cpu().numpy())
            progresses["I_or"].append(Ior_p.detach().cpu().numpy())
            pbar.set_postfix_str(f"loss={loss.item():.4f}")

    return p.detach(), q.detach(), losses, progresses



def _train_p_q_joint_model(
        rewards: torch.Tensor,
        lr: float,
        niter: int,
        alpha: float,
        qbound: Union[float, torch.Tensor],
        pbound: Union[float, torch.Tensor],
        reward2Iand: torch.Tensor = None,
        reward2Ior: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict, Dict]:

    rewards_1 = rewards[0]
    rewards_2 = rewards[1]
    qbound_1 = qbound[0]
    qbound_2 = qbound[1]
    pbound_1 = pbound[0]
    pbound_2 = pbound[1]
    device = rewards_1.device

    n_dim = int(np.log2(rewards_1.numel()))
    if reward2Iand is None:
        reward2Iand = get_reward2Iand_mat(n_dim).to(device)
    if reward2Ior is None:
        reward2Ior = get_reward2Ior_mat(n_dim).to(device)


    p_share = torch.zeros_like(rewards_1).requires_grad_(True)   # common decomposition shared by all DNNs in the paper
    p_1 = torch.zeros_like(rewards_1).requires_grad_(True)       # the decomposition specific to the first DNN
    p_2 = torch.zeros_like(rewards_1).requires_grad_(True)       # the decomposition specific to the second DNN
    q_1 = torch.zeros_like(rewards_1).requires_grad_(True)       # the error term (tiny noisy signals) for the first DNN
    q_2 = torch.zeros_like(rewards_1).requires_grad_(True)       # the error term (tiny noisy signals) for the second DNN
    optimizer = optim.SGD([p_share, p_1, p_2, q_1, q_2], lr=lr, momentum=0.9)

    losses = {"loss": []}
    progresses = {"I_and_1": [], "I_or_1": [], "I_and_2": [], "I_or_2": []}
    ckpt_id_list = generate_ckpt_id_list(niter=niter, nckpt=20)

    pbar = tqdm(range(niter), desc="Optimizing pq", ncols=100, leave=False, position=0)
    for it in pbar:
        q_1.data = torch.max(torch.min(q_1.data, qbound_1), -qbound_1)
        q_2.data = torch.max(torch.min(q_2.data, qbound_2), -qbound_2)
        p_1.data = torch.max(torch.min(p_1.data, pbound_1), -pbound_1)
        p_2.data = torch.max(torch.min(p_2.data, pbound_2), -pbound_2)

        Iand_p_1 = torch.matmul(reward2Iand, 0.5 * (rewards_1 + q_1) + p_share + p_1).squeeze()
        Ior_p_1 = torch.matmul(reward2Ior, 0.5 * (rewards_1 + q_1) - p_share - p_1).squeeze()
        Iand_p_2 = torch.matmul(reward2Iand, 0.5 * (rewards_2 + q_2) + p_share + p_2).squeeze()
        Ior_p_2 = torch.matmul(reward2Ior, 0.5 * (rewards_2 + q_2) - p_share - p_2).squeeze()

        # concat I_and and I_or
        I_1 = torch.cat((Iand_p_1.unsqueeze(dim=0), Ior_p_1.unsqueeze(dim=0)), dim=1)
        I_2 = torch.cat((Iand_p_2.unsqueeze(dim=0), Ior_p_2.unsqueeze(dim=0)), dim=1)
        and_or_matrix = torch.cat((I_1, I_2), dim=0)
        max_abs, _ = torch.max(torch.abs(and_or_matrix), dim=0)
        min_abs, _ = torch.min(torch.abs(and_or_matrix), dim=0)

        # Equation (6) in the paper
        v = (1-alpha) * max_abs + alpha * torch.sum(torch.abs(and_or_matrix), dim=0)
        loss = torch.sum(v)
        losses["loss"].append(loss.item())

        if it + 1 < niter:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if it in ckpt_id_list:
            progresses["I_and_1"].append(Iand_p_1.detach().cpu().numpy())
            progresses["I_or_1"].append(Ior_p_1.detach().cpu().numpy())
            progresses["I_and_2"].append(Iand_p_2.detach().cpu().numpy())
            progresses["I_or_2"].append(Ior_p_2.detach().cpu().numpy())
            pbar.set_postfix_str(f"loss={loss.item():.4f}")


    return p_share.detach(), p_1.detach(), p_2.detach(), q_1.detach(), q_2.detach(), losses, progresses

