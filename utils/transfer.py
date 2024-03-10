import sys
import os
import numpy as np
import pandas as pd
import torch
import os.path as osp
from utils import *
from tqdm import tqdm
import json
import re
import os.path as osp

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def cal_jaccard(set1, set2):
    set1 = set(set1)
    set2 = set(set2)
    # print("set1: ", set1)
    # print("set2: ", set2)
    intersect = set1.intersection(set2)
    # print("intersect: ", intersect)
    union = set1.union(set2)
    # print("union: ", union)
    if len(union) == 0:
        return 0, 0, 0
    else:
        return len(intersect), len(set1), len(set2)


def transferability(set1, set2):
    set1 = set(set1)
    set2 = set(set2)
    intersect = set1.intersection(set2)
    # print("intersect", intersect)
    return len(intersect)


def get_transferability(data_path, inds, salient_fixed_num, string, output_path, dataset = "sst2", model_name = ('llama', 'opt')):
    datasheet = pd.DataFrame(
        columns=["Sample", "And Interset", "And percent", "Or Interset", "Or percent"])

    for ind in tqdm(inds, desc='ind', total=len(inds)):
        if dataset == "sst2":
            if 'disjoint' not in string:
                if osp.exists(data_path + '/class_0' + f"/sample_{int(ind):>05d}"):
                    class_name = '/class_0'
                else:
                    class_name = '/class_1'
                # save_folder = data_path + class_name + f"/sample_{int(ind):>05d}"
                save_folder = data_path + class_name + f"/sample_{int(ind)}"
                Iand = np.abs(torch.load(osp.join(save_folder, "I_and_1.pth"), map_location=device).cpu().numpy())
                Iand_1 = np.abs(torch.load(osp.join(save_folder, "I_and_2.pth"), map_location=device).cpu().numpy())
                Ior = np.abs(torch.load(osp.join(save_folder, "I_or_1.pth"), map_location=device).cpu().numpy())
                Ior_1 = np.abs(torch.load(osp.join(save_folder, "I_or_2.pth"), map_location=device).cpu().numpy())
            else:
                if osp.exists(data_path[0] + '/class_0' + f"/sample_{int(ind):>05d}"):
                    class_name = '/class_0'
                else:
                    class_name = '/class_1'
                save_folder = data_path[0] + class_name + f"/sample_{int(ind)}"
                save_folder_1 = data_path[1] + class_name + f"/sample_{int(ind)}"
                Iand = np.abs(torch.load(osp.join(save_folder, "I_and_1.pth"), map_location=device).cpu().numpy())
                Iand_1 = np.abs(torch.load(osp.join(save_folder_1, "I_and_1.pth"), map_location=device).cpu().numpy())
                Ior = np.abs(torch.load(osp.join(save_folder, "I_or_1.pth"), map_location=device).cpu().numpy())
                Ior_1 = np.abs(torch.load(osp.join(save_folder_1, "I_or_1.pth"), map_location=device).cpu().numpy())
        elif dataset == "mnist":
            if 'disjoint' not in string:
                save_folder = data_path + f"/sample_{str(int(ind)).rjust(5, '0')}"
                Iand = torch.load(osp.join(save_folder, f"I_and.pth")).cpu().numpy()
                Iand_1 = torch.load(osp.join(save_folder, f'I_and_1.pth')).cpu().numpy()
                Ior = torch.load(osp.join(save_folder, f"I_or.pth")).cpu().numpy()
                Ior_1 = torch.load(osp.join(save_folder, f'I_or_1.pth')).cpu().numpy()
            else:
                save_folder = data_path[0] + f"/sample_{str(int(ind)).rjust(5, '0')}"
                save_folder_1 = data_path[1] + f"/sample_{str(int(ind)).rjust(5, '0')}"
                Iand = np.abs(torch.load(osp.join(save_folder, f"I_and.pth")).cpu().numpy())
                Iand_1 = np.abs(torch.load(osp.join(save_folder_1, f'I_and.pth')).cpu().numpy())
                Ior = np.abs(torch.load(osp.join(save_folder, f"I_or.pth")).cpu().numpy())
                Ior_1 = np.abs(torch.load(osp.join(save_folder_1, f'I_or.pth')).cpu().numpy())
        elif dataset == "squad":
            if 'disjoint' not in string:
                if osp.exists(data_path + f"/sample-{str(int(ind))}"):
                    save_folder = data_path + f"/sample-{str(int(ind))}"
                else:
                    save_folder = data_path + f"/sample{str(int(ind))}"


                # print("path:", save_folder)

                Iand = np.load(osp.join(save_folder,model_name[0], f"Iand.npy"))
                Iand_1 = np.load(osp.join(save_folder, model_name[1],f'Iand.npy'))
                Ior = np.load(osp.join(save_folder, model_name[0],f"Ior.npy"))
                Ior_1 = np.load(osp.join(save_folder,model_name[1], f'Ior.npy'))
            else:
                if osp.exists(data_path[0] + f"/sample-{str(int(ind))}"):
                    save_folder = data_path[0] + f"/sample-{str(int(ind))}"
                else:
                    save_folder = data_path[0] + f"/sample{str(int(ind))}"

                if osp.exists(data_path[1] + f"/sample-{str(int(ind))}"):
                    save_folder_1 = data_path[1] + f"/sample-{str(int(ind))}"
                else:
                    save_folder_1 = data_path[1] + f"/sample{str(int(ind))}"

                # print("path 1:", save_folder, "path 2:", save_folder_1)

                Iand = np.abs(np.load(osp.join(save_folder, f"Iand.npy")))
                Iand_1 = np.abs(np.load(osp.join(save_folder_1, f'Iand.npy')))
                Ior = np.abs(np.load(osp.join(save_folder, f"Ior.npy")))
                Ior_1 = np.abs(np.load(osp.join(save_folder_1, f'Ior.npy')))

        else:
            raise NotImplementedError('Please set dataset to one of "sst2", "mnist", "squad"!')

        # get the descending index of abstract value of interactions 
        sal_and_1 = np.argsort(np.abs(Iand))[::-1]  # np.argsort(np.abs(a))[::-1]:
        sal_and_2 = np.argsort(np.abs(Iand_1))[::-1]
        sal_or_1 = np.argsort(np.abs(Ior))[::-1]
        sal_or_2 = np.argsort(np.abs(Ior_1))[::-1]

        # Compare the set of interactions with the first N=salient_fixed_num with the largest absolute values
        and_common = transferability(sal_and_1[:salient_fixed_num], sal_and_2[:salient_fixed_num])
        or_common = transferability(sal_or_1[:salient_fixed_num], sal_or_2[:salient_fixed_num])


        and_percent = and_common / float(salient_fixed_num)
        or_percent = or_common / float(salient_fixed_num)

        item = {
            "Sample": int(ind),
            "And Interset": int(and_common),
            "And percent": and_percent,
            "Or Interset": int(or_common),
            "Or percent": or_percent
        }
        datasheet = datasheet.append(item, ignore_index=True)

    if output_path:
        if not osp.exists(osp.join(output_path)):
            os.makedirs(osp.join(output_path))
        datasheet.to_csv(osp.join(output_path,string), index=False)
    return datasheet

def get_transferability_for_baseline(data_path, inds, salient_fixed_num, string, output_path = None, add_or=False, prefix = False):
    """
    this function is used for baseline ablation experiment
    :param data_path:
    :param inds:
    :param salient_fixed_num:
    :param string:
    :param output_path:
    :return:
    """
    datasheet = pd.DataFrame(
        columns=["Sample", "And Interset", "And percent"])

    for ind in tqdm(inds, desc='ind', total=len(inds)):
        if 'disjoint' not in string:

            if osp.exists(data_path + '/class_0' + f"/sample_{int(ind)}"):
                class_name = '/class_0'
            else:
                class_name = '/class_1'

            save_folder = data_path + class_name + f"/sample_{int(ind)}"
            Iand = np.abs(torch.load(osp.join(save_folder, "I_and_1.pth"), map_location=device).cpu().numpy())
            Iand_1 = np.abs(torch.load(osp.join(save_folder, "I_and_2.pth"), map_location=device).cpu().numpy())
            if add_or:
                Ior = np.abs(torch.load(osp.join(save_folder, "I_or_1.pth"), map_location=device).cpu().numpy())
                Ior_1 = np.abs(torch.load(osp.join(save_folder, "I_or_2.pth"), map_location=device).cpu().numpy())

        else:
            if osp.exists(data_path[0] + '/class_0' + f"/sample_{int(ind):>05d}") or osp.exists(data_path[0] + '/class_0' + f"/sample_{int(ind)}"):
                class_name = '/class_0'
            else:
                class_name = '/class_1'
                
            if prefix:
                save_folder = data_path[0] + class_name + f"/sample_{int(ind):>05d}"
                save_folder_1 = data_path[1] + class_name + f"/sample_{int(ind):>05d}"
            else:
                save_folder = data_path[0] + class_name + f"/sample_{int(ind)}"
                save_folder_1 = data_path[1] + class_name + f"/sample_{int(ind)}"


            Iand = np.abs(torch.load(osp.join(save_folder, "I_and.pth"), map_location=device).cpu().numpy())
            Iand_1 = np.abs(torch.load(osp.join(save_folder_1, "I_and.pth"), map_location=device).cpu().numpy())
            if add_or:
                Ior = np.abs(torch.load(osp.join(save_folder, "I_or.pth"), map_location=device).cpu().numpy())
                Ior_1 = np.abs(torch.load(osp.join(save_folder_1, "I_or.pth"), map_location=device).cpu().numpy())

        # squeeze Iand to one dim
        Iand = np.squeeze(Iand)
        Iand_1 = np.squeeze(Iand_1)
    
        # get the index of interactions in descending order of its abs value
        sal_and_1 = np.argsort(np.abs(Iand))[::-1]  
        sal_and_2 = np.argsort(np.abs(Iand_1))[::-1]
        if add_or:
            sal_or_1 = np.argsort(np.abs(Ior))[::-1]
            sal_or_2 = np.argsort(np.abs(Ior_1))[::-1]

        # Compare the set of interactions with the first N=salient_fixed_num with the largest absolute values
        and_common = transferability(sal_and_1[:salient_fixed_num], sal_and_2[:salient_fixed_num])
        if add_or:
            or_common = transferability(sal_or_1[:salient_fixed_num], sal_or_2[:salient_fixed_num])

        and_percent = and_common / float(salient_fixed_num)
        if add_or:
            or_percent = or_common / float(salient_fixed_num)
  

        if add_or:
            item = {
                "Sample": int(ind),
                "And Interset": int(and_common),
                "And percent": and_percent,
                "Or Interset": int(or_common),
                "Or percent": or_percent
            }
        else:
            item = {
                "Sample": int(ind),
                "And Interset": int(and_common),
                "And percent": and_percent,
            }
        datasheet = datasheet.append(item, ignore_index=True)

    if output_path:
        if not osp.exists(osp.join(output_path)):
            os.makedirs(osp.join(output_path))
        datasheet.to_csv(osp.join(output_path,string), index=False)
    return datasheet


if __name__ == "__main__":
    pass
