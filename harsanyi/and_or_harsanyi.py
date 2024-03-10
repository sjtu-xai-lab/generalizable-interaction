import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn

from typing import Union, Tuple, Callable

from .and_or_harsanyi_utils import get_reward2Iand_mat, get_reward2Ior_mat, _train_p_q,  _train_p_q_joint_model
from .interaction_utils import calculate_all_subset_outputs, calculate_output_empty, calculate_output_N, get_reward


class AndOrHarsanyi(object):
    def __init__(
            self,
            interaction_type: str,
            model: Union[nn.Module, Callable],
            selected_dim: Union[None, str],
            x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            baseline: torch.Tensor,
            y: Union[torch.Tensor, int, None],
            all_players: Union[None, tuple, list] = None,
            background: Union[None, tuple, list] = None,
            mask_input_fn: Callable = None,
            calc_bs: int = None,
            verbose: int = 1
    ):

        self.interaction_type = interaction_type
        if interaction_type == 'traditional':
            self.model = model
            self.baseline = baseline
            self.input = x
        elif interaction_type == 'generalizable':
            self.model_1 = model[0]
            self.model_2 = model[1]
            self.baseline_1 = baseline[0]
            self.baseline_2 = baseline[1]
            self.input_1 = (x[0], x[1], x[2], x[3], x[4])
            self.input_2 = (x[5], x[1], x[2], x[3], x[4])

        self.selected_dim = selected_dim
        self.target = y
        self.device = x[0].device
        self.verbose = verbose
        self.all_players = all_players


        if background is None:
            background = []
        self.background = background  # background variables
        self.mask_input_fn = mask_input_fn
        self.calc_bs = calc_bs

        if all_players is not None:
            self.n_players = len(all_players)

        # calculate coefficients for all interactions
        self.reward2Iand = get_reward2Iand_mat(self.n_players).to(self.device)
        self.reward2Ior = get_reward2Ior_mat(self.n_players).to(self.device)

        # calculate the model output V(N) and V(\emptyset)
        if interaction_type == 'traditional':
            with torch.no_grad():
                self.output_empty = calculate_output_empty(
                    model=self.model, input=self.input, baseline=self.baseline,
                    all_players=self.all_players, background=self.background,
                    mask_input_fn=self.mask_input_fn, verbose=self.verbose
                )
                self.output_N = calculate_output_N(
                    model=self.model, input=self.input, baseline=self.baseline,
                    all_players=self.all_players, background=self.background,
                    mask_input_fn=self.mask_input_fn, verbose=self.verbose
                )
            if self.selected_dim.endswith("-v0"):
                self.v0 = get_reward(self.output_empty, self.selected_dim[:-3], gt=y)
            else:
                self.v0 = 0
            self.v_N = get_reward(self.output_N, self.selected_dim, gt=y, v0=self.v0)
            self.v_empty = get_reward(self.output_empty, self.selected_dim, gt=y, v0=self.v0)

        elif interaction_type == 'generalizable':
            with torch.no_grad():
                self.output_empty_1 = calculate_output_empty(
                    model=self.model_1, input=self.input_1, baseline=self.baseline_1,
                    all_players=self.all_players, background=self.background,
                    mask_input_fn=self.mask_input_fn, verbose=self.verbose
                )
                self.output_N_1 = calculate_output_N(
                    model=self.model_1, input=self.input_1, baseline=self.baseline_1,
                    all_players=self.all_players, background=self.background,
                    mask_input_fn=self.mask_input_fn, verbose=self.verbose
                )
                self.output_empty_2 = calculate_output_empty(
                    model=self.model_2, input=self.input_2, baseline=self.baseline_2,
                    all_players=self.all_players, background=self.background,
                    mask_input_fn=self.mask_input_fn, verbose=self.verbose
                )
                self.output_N_2 = calculate_output_N(
                    model=self.model_2, input=self.input_2, baseline=self.baseline_2,
                    all_players=self.all_players, background=self.background,
                    mask_input_fn=self.mask_input_fn, verbose=self.verbose
                )
            if self.selected_dim.endswith("-v0"):
                self.v0_1 = get_reward(self.output_empty_1, self.selected_dim[:-3], gt=y)
                self.v0_2 = get_reward(self.output_empty_2, self.selected_dim[:-3], gt=y)
            else:
                self.v0_1 = 0
                self.v0_2 = 0
            self.v_N_1 = get_reward(self.output_N_1, self.selected_dim, gt=y, v0=self.v0_1)
            self.v_empty_1 = get_reward(self.output_empty_1, self.selected_dim, gt=y, v0=self.v0_1)
            self.v_N_2 = get_reward(self.output_N_2, self.selected_dim, gt=y, v0=self.v0_2)
            self.v_empty_2 = get_reward(self.output_empty_2, self.selected_dim, gt=y, v0=self.v0_2)


    def attribute(self):
        # get original And-Or interactions
        if self.interaction_type == 'traditional':
            with torch.no_grad():
                self.masks, self.outputs = calculate_all_subset_outputs(
                    model=self.model, input=self.input, baseline=self.baseline,
                    all_players=self.all_players, background=self.background,
                    mask_input_fn=self.mask_input_fn, calc_bs=self.calc_bs,
                    verbose=self.verbose
                )
            self.rewards = get_reward(self.outputs, self.selected_dim, gt=self.target, v0=self.v0)
            self.Iand = torch.matmul(self.reward2Iand, self.rewards)
            self.Ior = torch.matmul(self.reward2Ior, self.rewards)

        elif self.interaction_type == 'generalizable':
            with torch.no_grad():
                self.masks, self.outputs_1 = calculate_all_subset_outputs(
                    model=self.model_1, input=self.input_1, baseline=self.baseline_1,
                    all_players=self.all_players, background=self.background,
                    mask_input_fn=self.mask_input_fn, calc_bs=self.calc_bs,
                    verbose=self.verbose
                )
                _, self.outputs_2 = calculate_all_subset_outputs(
                    model=self.model_2, input=self.input_2, baseline=self.baseline_2,
                    all_players=self.all_players, background=self.background,
                    mask_input_fn=self.mask_input_fn, calc_bs=self.calc_bs,
                    verbose=self.verbose
                )
            self.rewards_1 = get_reward(self.outputs_1, self.selected_dim, gt=self.target, v0=self.v0_1)
            self.Iand_1 = torch.matmul(self.reward2Iand, self.rewards_1)
            self.Ior_1 = torch.matmul(self.reward2Ior, self.rewards_1)
            self.rewards_2 = get_reward(self.outputs_2, self.selected_dim, gt=self.target, v0=self.v0_2)
            self.Iand_2 = torch.matmul(self.reward2Iand, self.rewards_2)
            self.Ior_2 = torch.matmul(self.reward2Ior, self.rewards_2)

    def get_masks(self):
        return self.masks


class AndOrHarsanyiSparsifier(object):
    def __init__(
            self,
            calculator: AndOrHarsanyi,
            interaction_type: str,
            lr: float,
            niter: int,
            qthres: float = None,
            pthres: float = None,
            qstd: str = None,
            alpha: float = None,
            average_model_output=0,
    ):
        self.calculator = calculator
        self.interaction_type = interaction_type
        self.pthres = pthres
        self.qthres = qthres
        self.qstd = qstd
        self.lr = lr
        self.niter = niter
        self.alpha = alpha

        if self.interaction_type == 'traditional':
            self.average_model_output = average_model_output
        elif self.interaction_type == 'generalizable':
            self.average_model_output_1 = average_model_output[0]
            self.average_model_output_2 = average_model_output[1]

        self.p = None  # refers to the parameter $\gamma$ in the paper
        self.q = None  # refers to the parameter $\epsilon$ in the paper
        self.q_bound = None
        self.p_bound = None

    # initialize the parameter $\epsilon$ in the paper
    def _init_q_bound(self):
        self.standard = None
        if self.qstd == "vN-v0":
            if self.interaction_type == 'traditional':
                standard = self.calculator.v_N - self.calculator.v_empty
                self.standard = torch.abs(standard)
                self.q_bound = self.qthres * self.standard

            elif self.interaction_type == 'generalizable':
                standard_1 = self.calculator.v_N_1 - self.calculator.v_empty_1
                standard_2 = self.calculator.v_N_2 - self.calculator.v_empty_2
                self.standard_1 = torch.abs(standard_1)
                self.q_bound_1 = self.qthres * self.standard_1
                self.standard_2 = torch.abs(standard_2)
                self.q_bound_2 = self.qthres * self.standard_2
        return

    # initialize the parameter $\gamma$ in the paper
    def _init_p_bound(self):
        if self.interaction_type == 'traditional':
            self.p_bound = self.pthres * torch.tensor([[self.average_model_output]],
                                                  device=self.calculator.device)

        elif self.interaction_type == 'generalizable':
            self.p_bound_1 = self.pthres * torch.tensor([[self.average_model_output_1]],
                                                        device=self.calculator.device)
            self.p_bound_2 = self.pthres * torch.tensor([[self.average_model_output_2]],
                                                        device=self.calculator.device)


    def get_vN_vEmpty(self):
        if self.interaction_type == 'traditional':
            return self.calculator.v_N - self.calculator.v_empty
        elif self.interaction_type == 'generalizable':
            return (self.calculator.v_N_1 - self.calculator.v_empty_1), \
                   (self.calculator.v_N_2 - self.calculator.v_empty_2)


    def sparsify(self, verbose_folder=None):
        # initialize the parameter q ($\epsilon$) and p ($\gamma$) in the paper
        self._init_q_bound()
        self._init_p_bound()


        # train p ($\gamma$) and q ($\epsilon$) in the paper
        if self.interaction_type == 'traditional':
            print("not model_joint")
            p, q, losses, progresses = _train_p_q(
                rewards=self.calculator.rewards,
                lr=self.lr, niter=self.niter,
                qbound=self.q_bound,
                reward2Iand=self.calculator.reward2Iand,
                reward2Ior=self.calculator.reward2Ior
            )
            self.p = p.clone()
            self.q = q.clone()

        elif self.interaction_type == 'generalizable':
            p_share, p_1, p_2, q_1, q_2, losses, progresses = _train_p_q_joint_model(
                rewards=[self.calculator.rewards_1, self.calculator.rewards_2],
                lr=self.lr, niter=self.niter, alpha=self.alpha,
                qbound=[self.q_bound_1, self.q_bound_2],
                pbound=[self.p_bound_1, self.p_bound_2],
                reward2Iand=self.calculator.reward2Iand,
                reward2Ior=self.calculator.reward2Ior
            )
            self.p_share = p_share.clone()
            self.p_1 = p_1.clone()
            self.p_2 = p_2.clone()
            self.q_1 = q_1.clone()
            self.q_2 = q_2.clone()

        # calculate And-Or interactions
        self._calculate_interaction()


    def _calculate_interaction(self):
        if self.interaction_type == 'traditional':
            rewards = self.calculator.rewards
            self.Iand = torch.matmul(self.calculator.reward2Iand,
                                     0.5 * (rewards + self.q) + self.p).detach()
            self.Ior = torch.matmul(self.calculator.reward2Ior,
                                    0.5 * (rewards + self.q) - self.p).detach()

        elif self.interaction_type == 'generalizable':
            rewards_1 = self.calculator.rewards_1
            self.Iand_1 = torch.matmul(self.calculator.reward2Iand,
                                     0.5 * (rewards_1 + self.q_1) + self.p_share + self.p_1).detach()
            self.Ior_1 = torch.matmul(self.calculator.reward2Ior,
                                    0.5 * (rewards_1 + self.q_1) - self.p_share - self.p_1).detach()

            rewards_2 = self.calculator.rewards_2
            self.Iand_2 = torch.matmul(self.calculator.reward2Iand,
                                       0.5 * (rewards_2 + self.q_2) + self.p_share + self.p_2).detach()
            self.Ior_2 = torch.matmul(self.calculator.reward2Ior,
                                              0.5 * (rewards_2 + self.q_2) - self.p_share - self.p_2).detach()


    def save(self, save_folder):
        os.makedirs(save_folder, exist_ok=True)
        if self.interaction_type == 'traditional':
            Iand = self.Iand.cpu().numpy()
            Ior = self.Ior.cpu().numpy()
            np.save(osp.join(save_folder, "Iand.npy"), Iand)
            np.save(osp.join(save_folder, "Ior.npy"), Ior)

            p = self.p.cpu().numpy()
            np.save(osp.join(save_folder, "p.npy"), p)
            q = self.q.cpu().numpy()
            np.save(osp.join(save_folder, "q.npy"), q)

        elif self.interaction_type == 'generalizable':
            Iand_1 = self.Iand_1.cpu().numpy()
            Ior_1 = self.Ior_1.cpu().numpy()
            Iand_2 = self.Iand_2.cpu().numpy()
            Ior_2 = self.Ior_2.cpu().numpy()
            np.save(osp.join(save_folder, "Iand_1.npy"), Iand_1)
            np.save(osp.join(save_folder, "Ior_1.npy"), Ior_1)
            np.save(osp.join(save_folder, "Iand_2.npy"), Iand_2)
            np.save(osp.join(save_folder, "Ior_2.npy"), Ior_2)

            p_share = self.p_share.cpu().numpy()
            np.save(osp.join(save_folder, "p_share.npy"), p_share)
            p_1 = self.p_1.cpu().numpy()
            np.save(osp.join(save_folder, "p_1.npy"), p_1)
            p_2 = self.p_2.cpu().numpy()
            np.save(osp.join(save_folder, "p_2.npy"), p_2)
            q_1 = self.q_1.cpu().numpy()
            np.save(osp.join(save_folder, "q_1.npy"), q_1)
            q_2 = self.q_2.cpu().numpy()
            np.save(osp.join(save_folder, "q_2.npy"), q_2)

    def get_interaction(self):
        if self.interaction_type == 'traditional':
            return self.Iand, self.Ior
        elif self.interaction_type == 'generalizable':
            return self.Iand_1, self.Ior_1, self.Iand_2, self.Ior_2

    def get_masks(self):
        return self.calculator.masks
