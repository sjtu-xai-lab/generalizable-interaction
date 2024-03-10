from matplotlib import pyplot as plt
import torch
from tqdm import tqdm
import os.path as osp
import os
import numpy as np
from .util import normalization
import matplotlib.ticker as ticker

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

##this two function are for base and large model to show on the same picture
def plot_sparsify_joint(data_path, inds, labels=None, save_path=None, title=None):
    interaction = []
    interaction_1 = []

    for ind in tqdm(inds, desc='ind', total=len(inds)):
        if osp.exists(data_path + '/class_0' + f"/sample_{int(ind):>05d}"):
            save_folder = data_path + '/class_0' + f"/sample_{int(ind):>05d}"
            I_and = torch.load(osp.join(save_folder, "I_and_1.pth"), map_location=device)
            I_or = torch.load(osp.join(save_folder, "I_or_1.pth"), map_location=device)
            I_and_1 = torch.load(osp.join(save_folder, "I_and_2.pth"), map_location=device)
            I_or_1 = torch.load(osp.join(save_folder, "I_or_2.pth"), map_location=device)
        else:
            save_folder = data_path + '/class_1' + f"/sample_{int(ind):>05d}"
            I_and = torch.load(osp.join(save_folder, "I_and_1.pth"), map_location=device)
            I_or = torch.load(osp.join(save_folder, "I_or_1.pth"), map_location=device)
            I_and_1 = torch.load(osp.join(save_folder, "I_and_2.pth"), map_location=device)
            I_or_1 = torch.load(osp.join(save_folder, "I_or_2.pth"), map_location=device)
        I_and = np.abs(I_and.cpu().numpy())
        I_or = np.abs(I_or.cpu().numpy())
        I_and_1 = np.abs(I_and_1.cpu().numpy())
        I_or_1 = np.abs(I_or_1.cpu().numpy())

        interaction = list(I_and) + list(I_or) + interaction
        interaction_1 = list(I_and_1) + list(I_or_1) + interaction_1

    interaction = normalization(sorted(interaction, reverse=True))
    interaction_1 = normalization(sorted(interaction_1, reverse=True))

    plt.figure(figsize=(10, 5))

    plt.plot(interaction, label=labels[0])
    plt.plot(interaction_1, label=labels[1])
    plt.legend()
    plt.xlabel('index')
    plt.ylabel('normalized interaction')
    plt.title(title)

    plt.savefig(save_path)


def plot_sparsify_disjoint(data_paths, inds, labels=None, save_path=None, title=None):
    interaction = []
    interaction_1 = []
    data_path = data_paths[0]
    data_path_1 = data_paths[1]

    for ind in tqdm(inds, desc='ind', total=len(inds)):
        ## model 1
        if osp.exists(data_path + '/class_0' + f"/sample_{int(ind):>05d}"):
            save_folder = data_path + '/class_0' + f"/sample_{int(ind):>05d}"
            I_and = torch.load(osp.join(save_folder, "I_and_1.pth"), map_location=device)
            I_or = torch.load(osp.join(save_folder, "I_or_1.pth"), map_location=device)
        else:
            save_folder = data_path + '/class_1' + f"/sample_{int(ind):>05d}"
            I_and = torch.load(osp.join(save_folder, "I_and_1.pth"), map_location=device)
            I_or = torch.load(osp.join(save_folder, "I_or_1.pth"), map_location=device)
        ## model 2
        if osp.exists(data_path_1 + '/class_0' + f"/sample_{int(ind):>05d}"):
            save_folder_1 = data_path_1 + '/class_0' + f"/sample_{int(ind):>05d}"
            I_and_1 = torch.load(osp.join(save_folder_1, "I_and_2.pth"), map_location=device)
            I_or_1 = torch.load(osp.join(save_folder_1, "I_or_2.pth"), map_location=device)
        else:
            save_folder_1 = data_path_1 + '/class_1' + f"/sample_{int(ind):>05d}"
            I_and_1 = torch.load(osp.join(save_folder_1, "I_and_2.pth"), map_location=device)
            I_or_1 = torch.load(osp.join(save_folder_1, "I_or_2.pth"), map_location=device)

        I_and = np.abs(I_and.cpu().numpy())
        I_or = np.abs(I_or.cpu().numpy())
        I_and_1 = np.abs(I_and_1.cpu().numpy())
        I_or_1 = np.abs(I_or_1.cpu().numpy())

        interaction = list(I_and) + list(I_or) + interaction
        interaction_1 = list(I_and_1) + list(I_or_1) + interaction_1

    interaction = normalization(sorted(interaction, reverse=True))
    interaction_1 = normalization(sorted(interaction_1, reverse=True))

    plt.figure(figsize=(10, 5))
    plt.rcParams['font.size'] = 14.5
    plt.rcParams['xtick.labelsize'] = 13
    plt.rcParams['ytick.labelsize'] = 13
    plt.plot(interaction, label=labels[0])
    plt.plot(interaction_1, label=labels[1])
    plt.legend()
    plt.xlabel('index')

    plt.ylabel('normalized interaction')
    plt.title(title)

    plt.savefig(save_path)

## contrast between joint and disjoint on the same model
def plot_sparsify_contrast(joint_path, disjoint_path, inds, labels=None, save_path=None, title=None, is_base=True, is_llama = True,
                           log=True, dataset = "sst2"):
    
    if dataset not in ["sst2", "mnist", "squad"]:
        raise NotImplementedError(' Please choose from "sst2", "mnist", "squad" ')
    
    joint_interaction = []
    disjoint_interaction = []

    for ind in tqdm(inds, desc='ind', total=len(inds)):
        
        if dataset == "sst2":
            if osp.exists(disjoint_path + '/class_0' + f"/sample_{int(ind):>05d}"):
                class_name = '/class_0'
            else:
                class_name = '/class_1'

            ## disjoint case
            save_folder = disjoint_path + class_name + f"/sample_{int(ind):>05d}"
            I_and_disjoint = torch.load(osp.join(save_folder, "I_and_1.pth"), map_location=device)
            I_or_disjoint = torch.load(osp.join(save_folder, "I_or_1.pth"), map_location=device)

            ## joint case
            save_folder_1 = joint_path + class_name + f"/sample_{int(ind):>05d}"
            if is_base:
                I_and_joint = torch.load(osp.join(save_folder_1, "I_and_1.pth"), map_location=device)
                I_or_joint = torch.load(osp.join(save_folder_1, "I_or_1.pth"), map_location=device)
            else:
                I_and_joint = torch.load(osp.join(save_folder_1, "I_and_2.pth"), map_location=device)
                I_or_joint = torch.load(osp.join(save_folder_1, "I_or_2.pth"), map_location=device)
                
            I_and_disjoint = np.abs(I_and_disjoint.cpu().numpy())
            I_or_disjoint = np.abs(I_or_disjoint.cpu().numpy())
            I_and_joint = np.abs(I_and_joint.cpu().numpy())
            I_or_joint = np.abs(I_or_joint.cpu().numpy())
            
        elif dataset == "mnist":
            
            ## disjoint case
            save_folder = disjoint_path + f"/sample_{int(ind):>05d}"
            I_and_disjoint = torch.load(osp.join(save_folder, "I_and.pth"), map_location=device)
            I_or_disjoint = torch.load(osp.join(save_folder, "I_or.pth"), map_location=device)

            ## joint case

            save_folder_1 = joint_path + f"/sample_{int(ind):>05d}"
            if is_base:
                I_and_joint = torch.load(osp.join(save_folder_1, "I_and.pth"), map_location=device)
                I_or_joint = torch.load(osp.join(save_folder_1, "I_or.pth"), map_location=device)
            else:
                I_and_joint = torch.load(osp.join(save_folder_1, "I_and_1.pth"), map_location=device)
                I_or_joint = torch.load(osp.join(save_folder_1, "I_or_1.pth"), map_location=device)
            I_and_disjoint = np.abs(I_and_disjoint.cpu().numpy())
            I_or_disjoint = np.abs(I_or_disjoint.cpu().numpy())
            I_and_joint = np.abs(I_and_joint.cpu().numpy())
            I_or_joint = np.abs(I_or_joint.cpu().numpy())
                
        elif dataset == "squad":
            ## disjoint case
            if os.path.exists(disjoint_path + f"/sample-{int(ind)}"):
                save_folder = disjoint_path + f"/sample-{int(ind)}"
            else:
                save_folder = disjoint_path + f"/sample{int(ind)}"
            I_and_disjoint = np.abs(np.load(osp.join(save_folder, "Iand.npy")))
            I_or_disjoint = np.abs(np.load(osp.join(save_folder, "Ior.npy")))

            ## joint case
            if is_llama:
                save_folder_1 = joint_path + f"/sample-{int(ind)}" + '/llama'
            else:
                save_folder_1 = joint_path + f"/sample-{int(ind)}" + '/opt'

            I_and_joint = np.abs(np.load(osp.join(save_folder_1, "Iand.npy")))
            I_or_joint = np.abs(np.load(osp.join(save_folder_1, "Ior.npy")))
        
        disjoint_interaction = list(I_and_disjoint) + list(I_or_disjoint) + disjoint_interaction
        joint_interaction = list(I_and_joint) + list(I_or_joint) + joint_interaction

    disjoint_interaction = (sorted(disjoint_interaction, reverse=True))
    joint_interaction = (sorted(joint_interaction, reverse=True))

    plt.figure(figsize=(4, 4))
    aw = 3
    fs = 15
    width = 3.5
    # plt.tick_params(axis='both', which='major', width=3.5, length=10)
    plt.rc('axes', linewidth=aw)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.rcParams['xtick.major.width'] = aw
    plt.rcParams['ytick.major.width'] = aw
    plt.rcParams['xtick.major.size'] = 7
    plt.rcParams['ytick.major.size'] = 7
    plt.rcParams['font.size'] = fs


    plt.plot(joint_interaction, label=labels[0], linewidth=width)
    plt.plot(disjoint_interaction, label=labels[1], linewidth=width)

    Strength_thres_20 = np.max(joint_interaction) * 0.05

    # count how many interactions are larger than Strength_thres_20
    count = 0
    for i in range(len(joint_interaction)):
        if joint_interaction[i] > Strength_thres_20:
            count += 1
    print("there are {} salient interactions (> 1/20 * max I) / {}".format(count, len(joint_interaction)))
    
    plt.axhline(y=Strength_thres_20, color='#7F7F7F', linestyle=':', linewidth=width,marker='.')

    ax = plt.gca()
    ax.set_ylim([0.000001, 100])

    plt.title(title)

    plt.yscale('log')  
    ax.yaxis.set_minor_locator(ticker.NullLocator())

    ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='x')

    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for tick in plt.gca().yaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()
    plt.close()


def plot_sparsify_contrast_baseline(joint_path, baseline_path, inds, traditional_path = None, labels=None, save_path=None, title=None, is_base=True,
                           log=True, add_or = False):
    """
    this function is used for ablation experiment of baseline contrast
    baseline use only AND interaction and no sparsify
    if 10 players are chosen, then its dim is 2^10 = 1024

    :param joint_path:
    :param baseline_path:
    :param inds:
    :param labels:
    :param save_path:
    :param title:
    :param is_base:
    :param log:
    :return:
    """

    joint_interaction = []
    baseline_interaction = []
    traditional_interaction = []

    for ind in tqdm(inds, desc='ind', total=len(inds)):
        if osp.exists(baseline_path + '/class_0' + f"/sample_{int(ind):>05d}"):
            class_name = '/class_0'
        else:
            class_name = '/class_1'

        ## baseline

        save_folder = baseline_path + class_name + f"/sample_{int(ind):>05d}"
        I_and_baseline = torch.load(osp.join(save_folder, "I_and.pth"), map_location=device)
        I_and_baseline = np.abs(I_and_baseline.cpu().numpy())
        
        ## joint case

        # save_folder_1 = joint_path + class_name + f"/sample_{int(ind):>05d}"
        save_folder_1 = joint_path + class_name + f"/sample_{int(ind)}"
        if is_base:
            I_and_joint = torch.load(osp.join(save_folder_1, "I_and_1.pth"), map_location=device)
            if add_or:
                I_or_joint = torch.load(osp.join(save_folder_1, "I_or_1.pth"), map_location=device)
        else:
            I_and_joint = torch.load(osp.join(save_folder_1, "I_and_2.pth"), map_location=device)
            if add_or:
                I_or_joint = torch.load(osp.join(save_folder_1, "I_or_2.pth"), map_location=device)
        I_and_joint = np.abs(I_and_joint.cpu().numpy())
        if add_or:
            I_or_joint = np.abs(I_or_joint.cpu().numpy())
            
        ## traditional(disjoint)
        if traditional_path is not None:
            # save_folder_2 = traditional_path + class_name + f"/sample_{int(ind):>05d}"
            save_folder_2 = traditional_path + class_name + f"/sample_{int(ind)}"

            I_and_traditional = torch.load(osp.join(save_folder_2, "I_and.pth"), map_location=device) #TODO, when traditional model also change Iand -> Iand_1, some modifications should be made here
            I_and_traditional = np.abs(I_and_traditional.cpu().numpy())
            if add_or:
                I_or_traditional = torch.load(osp.join(save_folder_2, "I_or.pth"), map_location=device)
                I_or_traditional = np.abs(I_or_traditional.cpu().numpy())


        if add_or:
            joint_interaction = list(I_and_joint) + list(I_or_joint) + joint_interaction
            baseline_interaction = list(I_and_baseline) + list(I_and_baseline) + baseline_interaction
            if traditional_path is not None:
                traditional_interaction = list(I_and_traditional) + list(I_or_traditional) + traditional_interaction
        else:

            baseline_interaction = list(I_and_baseline) + baseline_interaction
            joint_interaction = list(I_and_joint) + joint_interaction
            if traditional_path is not None:
                traditional_interaction = list(I_and_traditional) + traditional_interaction

    # print(len(interaction))
    baseline_interaction = (sorted(baseline_interaction, reverse=True))
    joint_interaction = (sorted(joint_interaction, reverse=True))
    if traditional_path is not None:
        traditional_interaction = (sorted(traditional_interaction, reverse=True))
    # print(interaction[:10])

    plt.figure(figsize=(4, 4))
    aw = 3
    fs = 15
    width = 3.5
    # plt.tick_params(axis='both', which='major', width=3.5, length=10)
    plt.rc('axes', linewidth=aw)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.rcParams['xtick.major.width'] = aw
    plt.rcParams['ytick.major.width'] = aw
    plt.rcParams['xtick.major.size'] = 7
    plt.rcParams['ytick.major.size'] = 7
    plt.rcParams['font.size'] = fs

    plt.plot(joint_interaction, label=labels[0], linewidth=width)
    
    if traditional_path is not None:
        plt.plot(traditional_interaction, label=labels[2], linewidth=width)

    plt.plot(baseline_interaction, label=labels[1], linewidth=width)

    Strength_thres_20 = np.max(joint_interaction) * 0.05

    # count how many interactions are larger than Strength_thres_20
    count = 0
    for i in range(len(joint_interaction)):
        if joint_interaction[i] > Strength_thres_20:
            count += 1
    print("there are {} salient interactions (> 1/20 * max I) / {}".format(count, len(joint_interaction)))
    

    plt.axhline(y=Strength_thres_20, color='#7F7F7F', linestyle=':', linewidth=width,marker='.')

    ax = plt.gca()
    ax.set_ylim([0.000001, 100])

    plt.yscale('log')  
    ax.yaxis.set_minor_locator(ticker.NullLocator())

    ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='x')

    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for tick in plt.gca().yaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
        
    if title:
        plt.title(title)

    plt.tight_layout()
    if save_path: 
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()
        
    plt.clf()
    plt.close()



def plot_sparsify_contrast_for_different_alphas(joint_paths, alphas, inds, save_path=None, title=None,log=True, is_base=True):
    """
    this function is used for ablation experiment of figure 2 in the paper
    :param joint_path: the path of joint training results
    :param alphas:  alphas is a list of alpha values, same order with joint_path
    :param disjoint_path:
    :param inds:
    :param labels:
    :param save_path:
    :param title:
    :param is_base:
    :param log:
    :return:
    """
    joint_interaction_dict = {}
    for joint_path in joint_paths:
        joint_interaction = []
        for ind in tqdm(inds, desc='ind', total=len(inds)):
            if osp.exists(joint_path + '/class_0' + f"/sample_{int(ind):>05d}"):
                class_name = '/class_0'
            else:
                class_name = '/class_1'
            ## joint case
            save_folder_1 = joint_path + class_name + f"/sample_{int(ind):>05d}"
            if is_base:
                I_and_joint = torch.load(osp.join(save_folder_1, "I_and_1.pth"), map_location=device)
                I_or_joint = torch.load(osp.join(save_folder_1, "I_or_1.pth"), map_location=device)
            else:
                I_and_joint = torch.load(osp.join(save_folder_1, "I_and_2.pth"), map_location=device)
                I_or_joint = torch.load(osp.join(save_folder_1, "I_or_2.pth"), map_location=device)

            I_and_joint = np.abs(I_and_joint.cpu().numpy())
            I_or_joint = np.abs(I_or_joint.cpu().numpy())

            joint_interaction = list(I_and_joint) + list(I_or_joint) + joint_interaction

        joint_interaction = (sorted(joint_interaction, reverse=True))
        joint_interaction_dict[joint_path] = joint_interaction

    plt.figure(figsize=(4, 4))
    aw = 3
    fs = 15
    width = 3.5

    plt.rc('axes', linewidth=aw)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.rcParams['xtick.major.width'] = aw
    plt.rcParams['ytick.major.width'] = aw
    plt.rcParams['xtick.major.size'] = 7
    plt.rcParams['ytick.major.size'] = 7
    plt.rcParams['font.size'] = fs
    all_interaction = []
    for joint_path, alpha in zip(joint_paths, alphas):
        joint_interaction = joint_interaction_dict[joint_path]
        all_interaction = list(joint_interaction) + all_interaction
        plt.plot(joint_interaction, label=f" alpha={alpha}", linewidth=width)

    Strength_thres_20 = np.max(np.abs(all_interaction)) * 0.05
    plt.axhline(y=Strength_thres_20, color='#7F7F7F', linestyle=':', linewidth=width,marker='.')
    # plt.legend(shadow=True,borderpad=1)
    ax = plt.gca()
    ax.set_ylim([0.000001, 100])
    plt.yscale('log')  
    ax.yaxis.set_minor_locator(ticker.NullLocator())

    ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='x')

    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for tick in plt.gca().yaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.cla()
    plt.close()


def plot_normalized_mean_standard(joint_path, disjoint_path, inds, labels=None, save_path=None, title=None,
                                  is_base=True):
    joint_interaction = [[] for i in range(2048)]
    disjoint_interaction = [[] for i in range(2048)]

    for ind in tqdm(inds, desc='ind', total=len(inds)):
        if osp.exists(disjoint_path + '/class_0' + f"/sample_{int(ind):>05d}"):
            class_name = '/class_0'
        else:
            class_name = '/class_1'

        ## disjoint case
        save_folder = disjoint_path + class_name + f"/sample_{int(ind):>05d}"
        I_and_disjoint = list(torch.load(osp.join(save_folder, "I_and_1.pth"), map_location=device).cpu().numpy())
        I_or_disjoint = list(torch.load(osp.join(save_folder, "I_or_1.pth"), map_location=device).cpu().numpy())
        I_disjoint = I_and_disjoint + I_or_disjoint

        ## joint case
        save_folder_1 = joint_path + class_name + f"/sample_{int(ind):>05d}"
        if is_base:
            I_and_joint = list(torch.load(osp.join(save_folder_1, "I_and_1.pth"), map_location=device).cpu().numpy())
            I_or_joint = list(torch.load(osp.join(save_folder_1, "I_or_1.pth"), map_location=device).cpu().numpy())
        else:
            I_and_joint = list(torch.load(osp.join(save_folder_1, "I_and_2.pth"), map_location=device).cpu().numpy())
            I_or_joint = list(torch.load(osp.join(save_folder_1, "I_or_2.pth"), map_location=device).cpu().numpy())
        I_joint = I_and_joint + I_or_joint

        I_disjoint = normalization(np.sort(np.abs(I_disjoint)))[::-1]
        I_joint = normalization(np.sort(np.abs(I_joint)))[::-1]

        for i in range(2048):
            disjoint_interaction[i].append(I_disjoint[i])
            joint_interaction[i].append(I_joint[i])

    disjoint_mean = [np.mean(disjoint_interaction[i]) for i in range(2048)]
    disjoint_std = [np.std(disjoint_interaction[i]) for i in range(2048)]
    joint_mean = [np.mean(joint_interaction[i]) for i in range(2048)]
    joint_std = [np.std(joint_interaction[i]) for i in range(2048)]

    plt.figure(figsize=(4, 3))
    plt.rcParams['font.size'] = 14.5
    plt.rcParams['xtick.labelsize'] = 13
    plt.rcParams['ytick.labelsize'] = 13
    plt.xlabel('index')

    plt.ylabel('interaction strength')
    plt.plot(joint_mean, label=labels[0])
    plt.fill_between(range(2048), np.array(joint_mean) - np.array(joint_std),
                     np.array(joint_mean) + np.array(joint_std), alpha=0.3)
    plt.plot(disjoint_mean, label=labels[1])
    plt.fill_between(range(2048), np.array(disjoint_mean) - np.array(disjoint_std),
                     np.array(disjoint_mean) + np.array(disjoint_std), alpha=0.3)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_single_sparsify_contrast(data_path, inds, label=None, save_path=None, title=None):
    disjoint_interaction = []
    for ind in tqdm(inds, desc='ind', total=len(inds)):
        if osp.exists(data_path + '/class_0' + f"/sample_{int(ind):>05d}"):
            class_name = '/class_0'
        else:
            class_name = '/class_1'

        ## disjoint case
        save_folder = data_path + class_name + f"/sample_{int(ind):>05d}"
        I_and_disjoint = torch.load(osp.join(save_folder, "I_and_1.pth"), map_location=device)
        I_or_disjoint = torch.load(osp.join(save_folder, "I_or_1.pth"), map_location=device)

        I_and_disjoint = np.abs(I_and_disjoint.cpu().numpy())
        I_or_disjoint = np.abs(I_or_disjoint.cpu().numpy())

        disjoint_interaction = list(I_and_disjoint) + list(I_or_disjoint) + disjoint_interaction

    disjoint_interaction = (sorted(disjoint_interaction, reverse=True))

    plt.figure(figsize=(4, 3))
    plt.rcParams['font.size'] = 14.5
    plt.rcParams['xtick.labelsize'] = 13
    plt.rcParams['ytick.labelsize'] = 13

    plt.plot(disjoint_interaction, label=label)
    plt.xlabel('index')
    plt.xscale('log')  
    plt.ylabel('interaction strength')

    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_single_mean_standard(data_path, inds, label=None, save_path=None, title=None, ):
    disjoint_interaction = [[] for i in range(2048)]

    for ind in tqdm(inds, desc='ind', total=len(inds)):
        if osp.exists(data_path + '/class_0' + f"/sample_{int(ind):>05d}"):
            class_name = '/class_0'
        else:
            class_name = '/class_1'

        ## disjoint case
        save_folder = data_path + class_name + f"/sample_{int(ind):>05d}"
        I_and_disjoint = list(torch.load(osp.join(save_folder, "I_and_1.pth"), map_location=device).cpu().numpy())
        I_or_disjoint = list(torch.load(osp.join(save_folder, "I_or_1.pth"), map_location=device).cpu().numpy())
        I_disjoint = I_and_disjoint + I_or_disjoint

        I_disjoint = normalization(np.sort(np.abs(I_disjoint)))[::-1]

        for i in range(2048):
            disjoint_interaction[i].append(I_disjoint[i])

    disjoint_mean = [np.mean(disjoint_interaction[i]) for i in range(2048)]
    disjoint_std = [np.std(disjoint_interaction[i]) for i in range(2048)]

    plt.figure(figsize=(4, 3))
    plt.rcParams['font.size'] = 14.5
    plt.rcParams['xtick.labelsize'] = 13
    plt.rcParams['ytick.labelsize'] = 13
    plt.xlabel('index')

    plt.ylabel('interaction strength')
    plt.plot(disjoint_mean, label=label)
    plt.fill_between(range(2048), np.array(disjoint_mean) - np.array(disjoint_std),
                     np.array(disjoint_mean) + np.array(disjoint_std), alpha=0.3)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


if __name__ == '__main__':
    joint_interaction = [[] for i in range(2048)]
    disjoint_interaction = [[] for i in range(2048)]
