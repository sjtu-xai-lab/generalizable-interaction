import numpy as np
import torch
import os


def mean_baseline(X_train):
    '''
    Use **mean** value in each attribute dimension as the reference value
    :param X_train: data sampled from p_data, torch.Size([n_samples, n_attributes])
    :return: baseline value in each dimension, torch.Size([1, n_attributes,])
    '''
    baseline = torch.mean(X_train, dim=0, keepdim=True)
    return baseline


def zero_baseline(X_train):
    '''
    Use **zero** value in each attribute dimension as the reference value
    :param X_train: data sampled from p_data, torch.Size([n_samples, n_attributes])
    :return: baseline value in each dimension, torch.Size([1, n_attributes,])
    '''
    baseline = torch.zeros_like(X_train[:1])
    return baseline


def damp_baseline(X_train, lamb_damp):
    assert X_train.shape[0] == 1
    assert 0 <= lamb_damp <= 1
    return X_train * lamb_damp


def center_baseline(point_cloud):
    """
    Use center point as the baseline value for point cloud data
    :param point_cloud: with shape [1, 3, n_points]
    :return:
    """
    return torch.mean(point_cloud, dim=(0, 2), keepdim=True)

def token_baseline(X_train, baseline_config):
    value_dict = {'PAD':0,
                  'UNK':100,
                  'CLS':101,
                  'SEP':102,
                  'MASK':103}
    baseline = torch.ones_like(X_train[:1])
    baseline = value_dict[baseline_config] * baseline
    return baseline


def get_baseline_value(model_str, model_dir, device, interaction_type, baseline_value_type='pretrained'):
    if baseline_value_type == 'pretrained':
        baseline_path = os.path.join(model_dir, 'baseline_values', interaction_type, f'bert-{model_str}-baseline-value.npy')
        baseline = np.load(baseline_path)[-1]
        baseline = torch.from_numpy(baseline).unsqueeze(0).to(device)
    return baseline

