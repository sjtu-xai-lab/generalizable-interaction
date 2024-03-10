from matplotlib import pyplot as plt
import torch
import numpy as np
import os
import re
import os.path as osp
from datetime import date
import matplotlib.ticker as ticker
from utils import get_inds


device = "cuda:2"

def draw_bar_chart_4(data1_pos, data1_neg, data2_pos, data2_neg, title, xlabel, ylabel, filename, label='mean',
                     cnt=False,
                     vmax=None, vmin=None):

    x = np.arange(0, len(data1_pos))
    aw = 3
    fs = 20

    plt.rc('axes', linewidth=aw)

    plt.rcParams['xtick.major.size'] = 7
    plt.rcParams['ytick.major.size'] = 7
    plt.rcParams['xtick.major.width'] = aw
    plt.rcParams['ytick.major.width'] = aw
    plt.rcParams['font.size'] = fs
    ax = plt.gca()

    plt.bar(x, data1_pos, width=0.4, label=f'{label[0]}', color='crimson')
    plt.bar(x, data1_neg, width=0.4, label=f'{label[0]}', color='royalblue')

    plt.bar(x + 0.4, data2_pos, width=0.4, label=f'{label[1]}', color='coral')
    plt.bar(x + 0.4, data2_neg, width=0.4, label=f'{label[1]}', color='seagreen')

    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for tick in plt.gca().yaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')

    plt.xticks(x + 0.2, list(x))
    
    if title:
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
    if vmax is not None:
        plt.ylim([vmin, vmax])

    if not cnt:
        plt.yscale('symlog')


    if filename:
        plt.savefig(filename + '.pdf')
    else:
        plt.show()
    plt.close()

def get_shared_concept_value_by_order(interaction_0, interaction_1, masks, threshold=0.03, data_path=None,count = False, model_thresholds = None, player_num  = 10):
    shared_strength_list_0_pos = np.zeros(player_num + 1)
    shared_strength_list_0_neg = np.zeros(player_num + 1)

    shared_strength_list_1_pos = np.zeros(player_num + 1)
    shared_strength_list_1_neg = np.zeros(player_num + 1)

    if data_path is None:
        thres_0 = threshold * np.max(np.abs(interaction_0))
        thres_1 = threshold * np.max(np.abs(interaction_1))
    else:
        thres_0 = threshold * model_thresholds[data_path[0]]
        thres_1 = threshold * model_thresholds[data_path[1]]
    salient_indices_0 = np.where(np.abs(interaction_0) > thres_0)[0]
    pattern_masks_0 = masks[salient_indices_0]

    salient_indices_1 = np.where(np.abs(interaction_1) > thres_1)[0]
    pattern_masks_1 = masks[salient_indices_1]

    for salient_indice_0 in salient_indices_0:
        for salient_indice_1 in salient_indices_1:
            pattern_0 = masks[salient_indice_0]
            pattern_1 = masks[salient_indice_1]
            if np.all(pattern_0 == pattern_1):
                if count:
                    if interaction_0[salient_indice_0] > 0:
                        shared_strength_list_0_pos[pattern_0.sum()] += 1
                    if interaction_0[salient_indice_0] <= 0:
                        shared_strength_list_0_neg[pattern_0.sum()] -= 1
                    if interaction_1[salient_indice_1] > 0:
                        shared_strength_list_1_pos[pattern_1.sum()] += 1
                    if interaction_1[salient_indice_1] <= 0:
                        shared_strength_list_1_neg[pattern_1.sum()] -= 1
                    break
                else:
                    if interaction_0[salient_indice_0] > 0:
                        shared_strength_list_0_pos[pattern_0.sum()] += interaction_0[salient_indice_0]
                    if interaction_0[salient_indice_0] <= 0:
                        shared_strength_list_0_neg[pattern_0.sum()] += interaction_0[salient_indice_0]
                    if interaction_1[salient_indice_1] > 0:
                        shared_strength_list_1_pos[pattern_1.sum()] += interaction_1[salient_indice_1]
                    if interaction_1[salient_indice_1] <= 0:
                        shared_strength_list_1_neg[pattern_1.sum()] += interaction_1[salient_indice_1]
                    break

    return shared_strength_list_0_pos, shared_strength_list_0_neg, shared_strength_list_1_pos, shared_strength_list_1_neg


def get_concept_value_by_order(interaction, masks, threshold=0.03, data_path=None,count = False, model_thresholds = None, player_num = 10):
    positive_value_list = np.zeros(player_num + 1)
    negative_value_list = np.zeros(player_num + 1)
    if data_path is None:
        thres = threshold * np.max(np.abs(interaction))
    else:
        thres = threshold * model_thresholds[data_path]
        # print(data_path, thres)
    salient_indices = np.where(np.abs(interaction) > thres)[0]

    for indice in salient_indices:
        order = masks[indice].sum()
        if interaction[indice] > 0:
            if count:
                positive_value_list[order] += 1
            else:
                positive_value_list[order] += interaction[indice]
        else:
            if count:
                negative_value_list[order] -= 1
            else:
                negative_value_list[order] += interaction[indice]
    return positive_value_list, negative_value_list

def is_in_marginal(pattern, pattern_1):
    '''
    pattern's length >= 5: more than 5 True in pattern
    is in marginal, with marginal distance = 1
    pattern is:  [False  True  True  True False  True  True  True  True  True]
    pattern_1 is:  [False  True  True  True False  True  True  True  True False]
    function: judge whether pattern and pattern_1 are marginally 'same' ,
        if pattern 1 and pattern_1 only differs in one element,
            here element refers to the index of pattern or pattern_1 where is 'True', 'False' should be ignored
        then pattern and pattern_1 are marginally 'same'
    '''
    count_diff = 0
    for i in range(len(pattern)):
        if pattern[i] and pattern_1[i]:
            continue
        elif pattern[i] or pattern_1[i]:
            count_diff += 1
        if count_diff > 1:
            return False
    return True


def draw_concept_by_order_pos_neg_andor(sample_inds, label, datapath, masks, models=['llama', 'bert_large'],
                                        model_name_print=['LLaMA', 'OPT-1.3B'], vmax=None, vmin=None, r=0.1, cnt=False, save = False, model_thresholds = None, dataset = "sst2", player_num = 10):
    """_summary_

    Args:
        sample_inds (_type_): _description_
        label (_type_): _description_
        datapath (_type_): _description_
        masks (_type_): _description_
        type (_type_): _description_
        models (list, optional): _description_. Defaults to ['llama', 'bert_large'].
        model_name_print (list, optional): _description_. Defaults to ['LLaMA', 'OPT-1.3B'].
        vmax (_type_, optional): _description_. Defaults to None.
        vmin (_type_, optional): _description_. Defaults to None.
        dataset (str, optional): _description_. Defaults to "sst2",  could be ["sst2", "mnist", "squad"]
    """
    
    if dataset not in ["sst2", "mnist", "squad"]:
        raise NotImplementedError('Please set dataset to one of "sst2", "mnist", "squad"!')
    
    average_strength_list_0_pos = np.zeros(player_num + 1)
    average_strength_list_0_neg = np.zeros(player_num + 1)

    average_strength_list_1_pos = np.zeros(player_num + 1)
    average_strength_list_1_neg = np.zeros(player_num + 1)

    average_shared_strength_list_0_pos = np.zeros(player_num + 1)
    average_shared_strength_list_0_neg = np.zeros(player_num + 1)

    average_shared_strength_list_1_pos = np.zeros(player_num + 1)
    average_shared_strength_list_1_neg = np.zeros(player_num + 1)

    for ind in sample_inds:
        if dataset == "sst2":
            if 'joint' in label:
                # if osp.exists(datapath + '/class_0' + f"/sample_{int(ind):>05d}"):
                if osp.exists(datapath + '/class_0' + f"/sample_{int(ind)}"):
                    class_name = '/class_0'
                else:
                    class_name = '/class_1'
                # save_folder = datapath + class_name + f"/sample_{int(ind):>05d}"
                save_folder = datapath + class_name + f"/sample_{int(ind)}"

                I_0_and = torch.load(osp.join(save_folder, f"I_and_1.pth")).cpu().numpy()
                I_1_and = torch.load(osp.join(save_folder, f"I_and_2.pth")).cpu().numpy()
                I_0_or = torch.load(osp.join(save_folder, f"I_or_1.pth")).cpu().numpy()
                I_1_or = torch.load(osp.join(save_folder, f"I_or_2.pth")).cpu().numpy()

            else:
                if osp.exists(datapath[0] + '/class_0' + f"/sample_{int(ind)}"):
                    class_name = '/class_0'
                else:
                    class_name = '/class_1'
                save_folder = datapath[0] + class_name + f"/sample_{int(ind)}"
                save_folder_1 = datapath[1] + class_name + f"/sample_{int(ind)}"
                                
                I_0_and = torch.load(
                    osp.join(save_folder, f'I_and.pth'), map_location=device).cpu().numpy()
                I_0_or = torch.load(
                    osp.join(save_folder, f'I_or.pth'), map_location=device).cpu().numpy()
                I_1_and = torch.load(
                    osp.join(save_folder_1, f'I_and.pth'),map_location=device).cpu().numpy()
                I_1_or = torch.load(
                    osp.join(save_folder_1, f'I_or.pth'),map_location=device).cpu().numpy()
                
        elif dataset == "mnist":
            if 'joint' in label:
                save_folder = datapath + f"/sample_{int(ind):>05d}"

                I_0_and = torch.load(osp.join(save_folder, f"I_and.pth")).cpu().numpy()
                I_1_and = torch.load(osp.join(save_folder, f"I_and_1.pth")).cpu().numpy()
                I_0_or = torch.load(osp.join(save_folder, f"I_or.pth")).cpu().numpy()
                I_1_or = torch.load(osp.join(save_folder, f"I_or_1.pth")).cpu().numpy()

            else:
                save_folder = datapath[0] + f"/sample_{int(ind):>05d}"
                save_folder_1 = datapath[1] + f"/sample_{int(ind):>05d}"
                I_0_and = torch.load(save_folder + f'/I_and.pth').cpu().numpy()
                I_0_or = torch.load(save_folder + f'/I_or.pth').cpu().numpy()
                I_1_and = torch.load((save_folder_1 + f'/I_and.pth')).cpu().numpy()
                I_1_or = torch.load(osp.join(save_folder_1 +  f'/I_or.pth')).cpu().numpy()
        elif dataset == "squad":
            if 'joint' in label:
                if osp.exists(datapath + f"/sample{int(ind)}"):
                    save_folder = datapath + f"/sample{int(ind)}/"
                else:
                    save_folder = datapath + f"/sample-{int(ind)}"
                I_0_and = np.load(osp.join(save_folder, models[0], f"Iand.npy"))
                I_1_and = np.load(osp.join(save_folder, models[1], f'Iand.npy'))
                I_0_or = np.load(osp.join(save_folder, models[0], f"Ior.npy"))
                I_1_or = np.load(osp.join(save_folder, models[1], f'Ior.npy'))

            else:
                if osp.exists(datapath[0] + f"/sample{int(ind)}"):
                    I_0_and = np.load(osp.join(datapath[0], f'sample{int(ind)}', f'Iand.npy'))
                    I_0_or = np.load(osp.join(datapath[0], f'sample{int(ind)}', f'Ior.npy'))
                else:
                    I_0_and = np.load(osp.join(datapath[0], f'sample-{int(ind)}', f'Iand.npy'))
                    I_0_or = np.load(osp.join(datapath[0], f'sample-{int(ind)}', f'Ior.npy'))
                I_1_and = np.load(osp.join(datapath[1], f'sample{int(ind)}', f'Iand.npy'))
                I_1_or = np.load(osp.join(datapath[1], f'sample{int(ind)}', f'Ior.npy'))

        pos_and, neg_and = get_concept_value_by_order(I_0_and, masks, data_path=models[0],count=cnt, model_thresholds=model_thresholds, player_num = player_num)

        pos_or, neg_or = get_concept_value_by_order(I_0_or, masks, data_path=models[0],count=cnt, model_thresholds=model_thresholds, player_num = player_num)

        average_strength_list_0_pos += pos_and + pos_or
        average_strength_list_0_neg += neg_and + neg_or

        pos_1_and, neg_1_and = get_concept_value_by_order(I_1_and, masks, data_path=models[1],count=cnt, model_thresholds=model_thresholds, player_num = player_num)
        pos_1_or, neg_1_or = get_concept_value_by_order(I_1_or, masks, data_path=models[1],count=cnt, model_thresholds=model_thresholds, player_num = player_num)
        average_strength_list_1_pos += pos_1_and + pos_1_or
        average_strength_list_1_neg += neg_1_and + neg_1_or

        l1_and, l2_and, l3_and, l4_and = get_shared_concept_value_by_order(I_0_and, I_1_and, masks,
                                                                            data_path=models,count=cnt, model_thresholds=model_thresholds, player_num = player_num)
        l1_or, l2_or, l3_or, l4_or = get_shared_concept_value_by_order(I_0_or, I_1_or, masks, data_path=models,count=cnt, model_thresholds=model_thresholds, player_num = player_num)

        average_shared_strength_list_0_pos += l1_and + l1_or
        average_shared_strength_list_0_neg += l2_and + l2_or
        average_shared_strength_list_1_pos += l3_and + l3_or
        average_shared_strength_list_1_neg += l4_and + l4_or

    average_strength_list_0_pos = average_strength_list_0_pos / len(sample_inds)
    average_strength_list_0_neg = average_strength_list_0_neg / len(sample_inds)
    average_strength_list_1_pos = average_strength_list_1_pos / len(sample_inds)
    average_strength_list_1_neg = average_strength_list_1_neg / len(sample_inds)

    average_shared_strength_list_0_pos = average_shared_strength_list_0_pos / len(sample_inds)
    average_shared_strength_list_0_neg = average_shared_strength_list_0_neg / len(sample_inds)

    average_shared_strength_list_1_pos = average_shared_strength_list_1_pos / len(sample_inds)

    average_shared_strength_list_1_neg = average_shared_strength_list_1_neg / len(sample_inds)

    if save:
        today = date.today()
        today = today.strftime("%b-%d")
        if 'individual' in label:
            save_path = f'./{today}/individual'
            os.makedirs(save_path, exist_ok=True)
        else:
            save_path = f'./{today}/{models[0]}_{models[1]}'
            os.makedirs(save_path, exist_ok=True)
                
        pic_results_path_0 = f'{save_path}/{models[0]}_{label}_{type}_strength'
        pic_results_path_1 = f'{save_path}/{models[1]}_{label}_{type}_strength'
    else:
        pic_results_path_0 = None
        pic_results_path_1 = None

    suffix = ""
    if 'joint' in label:
        suffix = "(ours)"
    else:
        suffix = "(traditional)"
        
    type = 'andor'
    draw_bar_chart_4(average_strength_list_0_pos, average_strength_list_0_neg, average_shared_strength_list_0_pos,
                     average_shared_strength_list_0_neg, f'{model_name_print[0]} {suffix}',
                     'Order of Concept',
                     'Sum of Interaction Values', pic_results_path_0,
                     label=['all', 'share'], vmax=vmax, vmin=vmin)
    draw_bar_chart_4(average_strength_list_1_pos, average_strength_list_1_neg, average_shared_strength_list_1_pos,
                     average_shared_strength_list_1_neg, f'{model_name_print[1]} {suffix}',
                     'Order of Concept',
                     'Sum of Interaction Values', pic_results_path_1,
                     label=['all', 'share'], vmax=vmax, vmin=vmin)

def get_shared_concept_value_by_order_no_threshold(interaction_0, interaction_1, masks, r=0.1, data_path=None,
                                                   count=False, player_num = 10):
    shared_strength_list_0_pos = np.zeros(player_num + 1)
    shared_strength_list_0_neg = np.zeros(player_num + 1)

    # print(r)
    shared_strength_list_1_pos = np.zeros(player_num + 1)
    shared_strength_list_1_neg = np.zeros(player_num + 1)

    num = len(masks) * r
    salient_indices_0 = np.argsort(-np.abs(interaction_0))[:int(num)]

    salient_indices_1 = np.argsort(-np.abs(interaction_1))[:int(num)]

    for salient_indice_0 in salient_indices_0:
        for salient_indice_1 in salient_indices_1:
            pattern_0 = masks[salient_indice_0]
            pattern_1 = masks[salient_indice_1]
            if np.all(pattern_0 == pattern_1):
                if count:
                    if interaction_0[salient_indice_0] > 0:
                        shared_strength_list_0_pos[pattern_0.sum()] += 1
                    if interaction_0[salient_indice_0] <= 0:
                        shared_strength_list_0_neg[pattern_0.sum()] -= 1
                    if interaction_1[salient_indice_1] > 0:
                        shared_strength_list_1_pos[pattern_1.sum()] += 1
                    if interaction_1[salient_indice_1] <= 0:
                        shared_strength_list_1_neg[pattern_1.sum()] -= 1
                    break
                else:
                    if interaction_0[salient_indice_0] > 0:
                        shared_strength_list_0_pos[pattern_0.sum()] += interaction_0[salient_indice_0]
                    if interaction_0[salient_indice_0] <= 0:
                        shared_strength_list_0_neg[pattern_0.sum()] += interaction_0[salient_indice_0]
                    if interaction_1[salient_indice_1] > 0:
                        shared_strength_list_1_pos[pattern_1.sum()] += interaction_1[salient_indice_1]
                    if interaction_1[salient_indice_1] <= 0:
                        shared_strength_list_1_neg[pattern_1.sum()] += interaction_1[salient_indice_1]
                    break

    return shared_strength_list_0_pos, shared_strength_list_0_neg, shared_strength_list_1_pos, shared_strength_list_1_neg


def get_concept_value_by_order_no_threshold(interaction, masks, data_path=None, r=0.1, count=False,player_num = 10):
    positive_value_list = np.zeros(player_num + 1)
    negative_value_list = np.zeros(player_num + 1)
    num = len(masks) * r
    salient_indices = np.argsort(-np.abs(interaction))[0:int(num)]
    for indice in salient_indices:
        order = masks[indice].sum()
        if interaction[indice] > 0:
            if count:
                positive_value_list[order] += 1
            else:
                positive_value_list[order] += interaction[indice]
        else:
            if count:
                negative_value_list[order] -= 1
            else:
                negative_value_list[order] += interaction[indice]
    return positive_value_list, negative_value_list


def draw_concept_by_order_pos_neg_andor_no_threshold(sample_inds, label, datapath, masks,
                                                     models=['llama', 'bert_large'],
                                                     model_name_print=['LLaMA', 'OPT-1.3B'], vmax=None, vmin=None, cnt=False, r=0.1,player_num = 10):
    average_strength_list_0_pos = np.zeros(player_num + 1)
    average_strength_list_0_neg = np.zeros(player_num + 1)

    average_strength_list_1_pos = np.zeros(player_num + 1)
    average_strength_list_1_neg = np.zeros(player_num + 1)

    average_shared_strength_list_0_pos = np.zeros(player_num + 1)
    average_shared_strength_list_0_neg = np.zeros(player_num + 1)

    average_shared_strength_list_1_pos = np.zeros(player_num + 1)
    average_shared_strength_list_1_neg = np.zeros(player_num + 1)

    for ind in sample_inds:
        if 'joint' in label:
            if osp.exists(datapath + '/class_0' + f"/sample_{int(ind):>05d}"):
                class_name = '/class_0'
            else:
                class_name = '/class_1'
            save_folder = datapath + class_name + f"/sample_{int(ind):>05d}"
            I_0_and = torch.load(osp.join(save_folder, f"I_and_1.pth")).cpu().numpy()
            I_1_and = torch.load(osp.join(save_folder, f"I_and_2.pth")).cpu().numpy()
            I_0_or = torch.load(osp.join(save_folder, f"I_or_1.pth")).cpu().numpy()
            I_1_or = torch.load(osp.join(save_folder, f"I_or_2.pth")).cpu().numpy()

        else:
            if osp.exists(datapath[0] + '/class_0' + f"/sample_{int(ind):>05d}"):
                class_name = '/class_0'
            else:
                class_name = '/class_1'
            I_0_and = torch.load(
                osp.join(datapath[0], class_name, f"/sample_{int(ind):>05d}", f'I_and_1.pth')).cpu().numpy()
            I_0_or = torch.load(
                osp.join(datapath[0], class_name, f"/sample_{int(ind):>05d}", f'I_or_1.pth')).cpu().numpy()
            I_1_and = torch.load(
                osp.join(datapath[1], class_name, f"/sample_{int(ind):>05d}", f'I_and_1.pth')).cpu().numpy()
            I_1_or = torch.load(
                osp.join(datapath[1], class_name, f"/sample_{int(ind):>05d}", f'I_or_1.pth')).cpu().numpy()

        if 'joint' in label:
            pos_and, neg_and = get_concept_value_by_order_no_threshold(I_0_and, masks, data_path=models[0], count=cnt,
                                                                       r=r)
            pos_or, neg_or = get_concept_value_by_order_no_threshold(I_0_or, masks, data_path=models[0], count=cnt,
                                                                     r=r)
            average_strength_list_0_pos += pos_and + pos_or
            average_strength_list_0_neg += neg_and + neg_or

            pos_1_and, neg_1_and = get_concept_value_by_order_no_threshold(I_1_and, masks, data_path=models[1],
                                                                           count=cnt,
                                                                           r=r)
            pos_1_or, neg_1_or = get_concept_value_by_order_no_threshold(I_1_or, masks, data_path=models[1], count=cnt,
                                                                         r=r)
            average_strength_list_1_pos += pos_1_and + pos_1_or
            average_strength_list_1_neg += neg_1_and + neg_1_or


            l1_and, l2_and, l3_and, l4_and = get_shared_concept_value_by_order_no_threshold(I_0_and, I_1_and, masks,
                                                                                            data_path=models, count=cnt,
                                                                                            r=r)
            l1_or, l2_or, l3_or, l4_or = get_shared_concept_value_by_order_no_threshold(I_0_or, I_1_or, masks,
                                                                                        data_path=models, count=cnt,
                                                                                        r=r)
        else:
            pos_and, neg_and = get_concept_value_by_order_no_threshold(I_0_and, masks, data_path=datapath[0])
            pos_or, neg_or = get_concept_value_by_order_no_threshold(I_0_or, masks, data_path=datapath[0])
            average_strength_list_0_pos += pos_and + pos_or
            average_strength_list_0_neg += neg_and + neg_or

            pos_1_and, neg_1_and = get_concept_value_by_order_no_threshold(I_1_and, masks, data_path=datapath[1])
            pos_1_or, neg_1_or = get_concept_value_by_order_no_threshold(I_1_or, masks, data_path=datapath[1])
            average_strength_list_1_pos += pos_1_and + pos_1_or
            average_strength_list_1_neg += neg_1_and + neg_1_or

            l1_and, l2_and, l3_and, l4_and = get_shared_concept_value_by_order_no_threshold(I_0_and, I_1_and, masks,
                                                                                            data_path=datapath)
            l1_or, l2_or, l3_or, l4_or = get_shared_concept_value_by_order_no_threshold(I_0_or, I_1_or, masks,
                                                                                        data_path=datapath)

        average_shared_strength_list_0_pos += l1_and + l1_or
        average_shared_strength_list_0_neg += l2_and + l2_or
        average_shared_strength_list_1_pos += l3_and + l3_or
        average_shared_strength_list_1_neg += l4_and + l4_or

    average_strength_list_0_pos = average_strength_list_0_pos / len(sample_inds)
    average_strength_list_0_neg = average_strength_list_0_neg / len(sample_inds)
    average_strength_list_1_pos = average_strength_list_1_pos / len(sample_inds)
    average_strength_list_1_neg = average_strength_list_1_neg / len(sample_inds)

    average_shared_strength_list_0_pos = average_shared_strength_list_0_pos / len(sample_inds)
    average_shared_strength_list_0_neg = average_shared_strength_list_0_neg / len(sample_inds)

    average_shared_strength_list_1_pos = average_shared_strength_list_1_pos / len(sample_inds)

    average_shared_strength_list_1_neg = average_shared_strength_list_1_neg / len(sample_inds)

    today = date.today()
    today = today.strftime("%b-%d")
    if 'individual' in label:
        save_path = f'./{today}/individual'
        os.makedirs(save_path, exist_ok=True)
    else:
        save_path = f'./{today}/{models[0]}_{models[1]}'
        os.makedirs(save_path, exist_ok=True)

    type = 'andor'
    draw_bar_chart_4(average_strength_list_0_pos, average_strength_list_0_neg, average_shared_strength_list_0_pos,
                     average_shared_strength_list_0_neg, f'{model_name_print[0]}',
                     'Order of Concept',
                     'Sum of Interaction Values', f'{save_path}/{models[0]}_{label}_{type}_strength',
                     label=['all', 'share'], vmax=vmax, vmin=vmin)
    draw_bar_chart_4(average_strength_list_1_pos, average_strength_list_1_neg, average_shared_strength_list_1_pos,
                     average_shared_strength_list_1_neg, f'{model_name_print[1]}',
                     'Order of Concept',
                     'Sum of Interaction Values', f'{save_path}/{models[1]}_{label}_{type}_strength',
                     label=['all', 'share'], vmax=vmax, vmin=vmin)


def draw_concept_by_order_pos_neg(sample_inds, label, datapath, masks, type, models=['llama', 'bert_large'],
                                  model_name_print=['LLaMA', 'OPT-1.3B'], vmax=None, vmin=None, cnt=False, r=0.1, save = False, dataset = "sst2",player_num = 10):
    
    """_summary_

    Args:
        sample_inds (_type_): _description_
        label (_type_): _description_
        datapath (_type_): _description_
        masks (_type_): _description_
        type (_type_): _description_
        models (list, optional): _description_. Defaults to ['llama', 'bert_large'].
        model_name_print (list, optional): _description_. Defaults to ['LLaMA', 'OPT-1.3B'].
        vmax (_type_, optional): _description_. Defaults to None.
        vmin (_type_, optional): _description_. Defaults to None.
        dataset (str, optional): _description_. Defaults to "sst2",  could be ["sst2", "mnist", "squad"]
    """
    
    if dataset not in ["sst2", "mnist", "squad"]:
        raise NotImplementedError('Please set dataset to one of "sst2", "mnist", "squad"!')
    
    average_strength_list_0_pos = np.zeros(player_num + 1)
    average_strength_list_0_neg = np.zeros(player_num + 1)

    average_strength_list_1_pos = np.zeros(player_num + 1)
    average_strength_list_1_neg = np.zeros(player_num + 1)

    average_shared_strength_list_0_pos = np.zeros(player_num + 1)
    average_shared_strength_list_0_neg = np.zeros(player_num + 1)

    average_shared_strength_list_1_pos = np.zeros(player_num + 1)
    average_shared_strength_list_1_neg = np.zeros(player_num + 1)

    for ind in sample_inds:
        if dataset == "sst2":
            if 'joint' in label:
                if osp.exists(datapath + '/class_0' + f"/sample_{int(ind):>05d}"):
                    class_name = '/class_0'
                else:
                    class_name = '/class_1'
                save_folder = datapath + class_name + f"/sample_{int(ind):>05d}"

                
                I_0 = torch.load(osp.join(save_folder, f"I_{type}.pth")).cpu().numpy()
                I_1 = torch.load(osp.join(save_folder, f"I_{type}_1.pth")).cpu().numpy()
            else:
                if osp.exists(datapath[0] + '/class_0' + f"/sample_{int(ind):>05d}"):
                    class_name = '/class_0'
                else:
                    class_name = '/class_1'
                I_0 = torch.load(
                    osp.join(datapath[0], class_name, f"/sample_{int(ind):>05d}", f'I_{type}.pth')).cpu().numpy()
                I_1 = torch.load(
                    osp.join(datapath[1], class_name, f"/sample_{int(ind):>05d}", f'I_{type}.pth')).cpu().numpy()
                
        elif dataset == "mnist":
            if 'joint' in label:
                if osp.exists(datapath + f"/sample{int(ind)}"):
                    save_folder = datapath + f"/sample{int(ind)}/"
                else:
                    save_folder = datapath + f"/sample-{int(ind)}"
                I_0 = np.load(osp.join(save_folder, models[0], f"I{type}.npy"))
                I_1 = np.load(osp.join(save_folder, models[1], f'I{type}.npy'))

            else:
                if osp.exists(datapath[0] + f"/sample{int(ind)}"):
                    I_0 = np.load(osp.join(datapath[0], f'sample{int(ind)}', f'I{type}.npy'))
                else:
                    I_0 = np.load(osp.join(datapath[0], f'sample-{int(ind)}', f'I{type}.npy'))
                I_1 = np.load(osp.join(datapath[1], f'sample{int(ind)}', f'I{type}.npy'))
            
        print("=====================sample{}_{}".format(ind, type))
        if 'joint' in label:

            pos, neg = get_concept_value_by_order_no_threshold(I_0, masks, data_path=models[0], count=cnt, r=r)
            average_strength_list_0_pos += pos
            average_strength_list_0_neg += neg

            pos, neg = get_concept_value_by_order_no_threshold(I_1, masks, data_path=models[1], count=cnt, r=r)
            average_strength_list_1_pos += pos
            average_strength_list_1_neg += neg

            l1, l2, l3, l4 = get_shared_concept_value_by_order_no_threshold(I_0, I_1, masks, data_path=models,
                                                                            count=cnt, r=r)
        else:
            pos, neg = get_concept_value_by_order(I_0, masks, data_path=datapath[0], model_thresholds=model_thresholds)
            average_strength_list_0_pos += pos
            average_strength_list_0_neg += neg

            pos, neg = get_concept_value_by_order(I_1, masks, data_path=datapath[1], model_thresholds=model_thresholds)
            average_strength_list_1_pos += pos
            average_strength_list_1_neg += neg

            l1, l2, l3, l4 = get_shared_concept_value_by_order(I_0, I_1, masks, data_path=datapath, model_thresholds=model_thresholds)

        average_shared_strength_list_0_pos += l1
        average_shared_strength_list_0_neg += l2
        average_shared_strength_list_1_pos += l3
        average_shared_strength_list_1_neg += l4

    average_strength_list_0_pos = average_strength_list_0_pos / len(sample_inds)
    average_strength_list_0_neg = average_strength_list_0_neg / len(sample_inds)
    average_strength_list_1_pos = average_strength_list_1_pos / len(sample_inds)
    average_strength_list_1_neg = average_strength_list_1_neg / len(sample_inds)

    average_shared_strength_list_0_pos = average_shared_strength_list_0_pos / len(sample_inds)
    average_shared_strength_list_0_neg = average_shared_strength_list_0_neg / len(sample_inds)

    average_shared_strength_list_1_pos = average_shared_strength_list_1_pos / len(sample_inds)

    average_shared_strength_list_1_neg = average_shared_strength_list_1_neg / len(sample_inds)


    if save:
        today = date.today()
        today = today.strftime("%b-%d")
        if 'individual' in label:
            save_path = f'./{today}/individual'
            os.makedirs(save_path, exist_ok=True)
        else:
            save_path = f'./{today}/{models[0]}_{models[1]}'
            os.makedirs(save_path, exist_ok=True)
        
        pic_results_path_0 = f'{save_path}/{models[0]}_{label}_{type}_strength'
        pic_results_path_1 = f'{save_path}/{models[1]}_{label}_{type}_strength'
    else:
        pic_results_path_0 = None
        pic_results_path_1 = None
        
    if "joint" in label:
        suffix = "(joint)"
    else:
        suffix = "(individual)"
    draw_bar_chart_4(average_strength_list_0_pos, average_strength_list_0_neg, average_shared_strength_list_0_pos,
                    average_shared_strength_list_0_neg, f'{model_name_print[0]} {suffix}',
                    'Order of Concept',
                    'Sum of Interaction Values',pic_results_path_0 ,
                    label=['all', 'share'], vmax=vmax, vmin=vmin, cnt=cnt)
    draw_bar_chart_4(average_strength_list_1_pos, average_strength_list_1_neg, average_shared_strength_list_1_pos,
                    average_shared_strength_list_1_neg, f'{model_name_print[1]} {suffix}',
                    'Order of Concept',
                    'Sum of Interaction Values',pic_results_path_1,
                    label=['all', 'share'], vmax=vmax, vmin=vmin, cnt=cnt)
    


