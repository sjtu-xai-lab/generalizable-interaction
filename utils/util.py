import os
import random
import stopwords
import pickle
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from typing import List
import string
import re

from harsanyi.and_or_harsanyi import AndOrHarsanyi, AndOrHarsanyiSparsifier  # AndHarsanyi
from harsanyi.interaction_utils import get_mask_input_func_nlp, flatten, reorganize_and_or_harsanyi

def get_inds(data_path, dataset = "sst2"):
    '''

    :param data_path:
    :return: list of ind in `sample_[inds]`
    '''
    if dataset == "mnist":
        samples = os.listdir(data_path)
        numbers = []
        for ind in samples:
            match = re.search(r'sample_?(\d+)', ind)
            if match:
                number = int(match.group(1))

                numbers.append(int(match.group(1)))
        return numbers
    elif dataset == "sst2":
        class_0 = os.listdir(data_path + '/class_0')
        class_0_inds = [int(name[7:]) for name in class_0]
        class_1 = os.listdir(data_path + '/class_1')
        class_1_inds = [int(name[7:]) for name in class_1]
        return class_0_inds + class_1_inds
    elif dataset == "squad":
        samples = os.listdir(data_path)
        numbers = []
        for ind in samples:
            match = re.search(r'sample-?(\d+)', ind)
            if match:
                number = int(match.group(1))
                numbers.append(int(match.group(1)))
        return numbers
    
def get_masks(playnumber : int):
    rows = 2 ** playnumber
    mask = np.zeros((rows, playnumber), dtype=bool)
    
    for row in range(rows):
        binary_representation = [bool(int(x)) for x in bin(row)[2:].zfill(playnumber)]
        mask[row] = binary_representation
    
    return mask

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def setup_seed(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.


def get_stop_words():
    # get stop words
    alphabet = [s for s in string.ascii_letters]
    digits = [d for d in string.digits]
    stop_words = stopwords.get_stopwords('en')
    stop_words += [',', "'s", "'"]
    stop_words += alphabet
    stop_words += digits
    stop_words = set(stop_words)
    print("The length of stopwords is: ", len(stop_words))
    return stop_words



def get_samples_and_input_variables(args, model, forward_func, baseline, interaction_type,
                                    stop_words, sentences,
                                    seqs, masks, segments, labels,
                                    model_str=''):
    SELECTED_SAMPLE_PATH =  os.path.join(args.output_path, 'selected_sample_indices.pkl')  #"selected_indices_filtered.pkl"
    INPUT_VARIABLE_PATH =  os.path.join(args.output_path, 'input_variable_indices.pkl')  #'inds_to_players_filtered.pkl'
    #MODEL_OUTPUT_OF_SAMPLE_PATH = os.path.join(args.output_path, 'selected_sample_model_output.csv')
    if interaction_type == 'traditional':
        MODEL_OUTPUT_OF_SAMPLE_PATH = os.path.join(args.output_path, interaction_type, model_str)
    elif interaction_type == 'generalizable':
        MODEL_OUTPUT_OF_SAMPLE_PATH = os.path.join(args.output_path, interaction_type)
    os.makedirs(MODEL_OUTPUT_OF_SAMPLE_PATH, exist_ok=True)
    MODEL_OUTPUT_OF_SAMPLE_PATH = os.path.join(MODEL_OUTPUT_OF_SAMPLE_PATH, 'selected_sample_model_output.csv')
    device = next(model.parameters()).device
    print(SELECTED_SAMPLE_PATH, INPUT_VARIABLE_PATH)
    print(MODEL_OUTPUT_OF_SAMPLE_PATH)


    if interaction_type == 'generalizable':
        forward_func_1 = forward_func[0]
        forward_func_2 = forward_func[1]
        baseline_1 = baseline[0]
        baseline_2 = baseline[1]

    # if not exists selected samples and their input variables
    if not (os.path.exists(SELECTED_SAMPLE_PATH) and os.path.exists(INPUT_VARIABLE_PATH)):
        # get correctly classified samples for a model
        with torch.no_grad():
            correct_indices = get_correct_sample_indices(model, seqs, masks, segments, labels, bs=16)

        # select samples (sentences) with more than 20 tokens
        sample_indices = get_sample_indices_each_class(correct_indices, labels, masks,
                                                     n_sample_each_class=2000,
                                                     selected_classes=None)
        # get the (randomly) selected input variables for each sample
        qualified_indices = {0: [], 1: []}
        input_variable_indices = dict()
        for class_id in sample_indices.keys():
            for sample_id in sample_indices[class_id]:
                if interaction_type == 'traditional':
                    selected_players = get_all_players_inds(sentences[sample_id], forward_func.tokenizer, stop_words)
                elif interaction_type == 'generalizable':
                    selected_players = get_all_players_inds(sentences[sample_id], forward_func_1.tokenizer, stop_words)
                
                if selected_players != []:  # selected_players
                    qualified_indices[class_id].append(sample_id)
                    input_variable_indices[sample_id] = selected_players
                else:
                    continue
                
        sample_indices = qualified_indices
        
        # save samples and their input variables
        f_save = open(SELECTED_SAMPLE_PATH, 'wb')
        pickle.dump(sample_indices, f_save)
        f_save.close()

        f_save = open(INPUT_VARIABLE_PATH, 'wb')
        pickle.dump(input_variable_indices, f_save)
        f_save.close()


    # load selected samples and their input variables
    f_read = open(SELECTED_SAMPLE_PATH, 'rb')
    sample_indices = pickle.load(f_read)
    f_read.close()

    f_read = open(INPUT_VARIABLE_PATH, 'rb')
    input_variable_indices = pickle.load(f_read)
    f_read.close()

    SAMPLE_NUM = 100
    sample_indices[0] = sample_indices[0][:SAMPLE_NUM]
    sample_indices[1] = sample_indices[1][:SAMPLE_NUM]

    # if the model output for each sample is not calculated
    if not os.path.exists(MODEL_OUTPUT_OF_SAMPLE_PATH):
        if interaction_type == 'traditional':
            table = pd.DataFrame(columns=['sample_index', 'model'])
        elif interaction_type == 'generalizable':
            table = pd.DataFrame(columns=['sample_index', 'model_1', 'model_2'])
        for class_id in sample_indices.keys():
            for sample_id in tqdm(sample_indices[class_id], desc=f"Class {class_id}",
                                  total=len(sample_indices[class_id]), leave=False, ncols=100, position=0):

                # get each sample
                sample_s = seqs[sample_id].clone().unsqueeze(0).to(device)
                sample_mask = masks[sample_id].clone().unsqueeze(0).to(device)
                sample_segment = segments[sample_id].clone().unsqueeze(0).to(device)
                sentence = sentences[sample_id]
                label = labels[sample_id].clone().unsqueeze(0).to(device)

                with torch.no_grad():
                    if interaction_type == 'traditional':
                        sample_seq = forward_func._get_embedding(input_ids=sample_s)  # [1, 50, 768]

                    elif interaction_type == 'generalizable':
                        sample_seq_1 = forward_func_1._get_embedding(input_ids=sample_s)  # [1, 50, 768]
                        sample_seq_2 = forward_func_2._get_embedding(input_ids=sample_s)  # [1, 50, 1024]


                # calculate the model output V(N)-V(\emptyset) for each sample
                model_output_1 = 0
                model_output_2 = 0
                model_output = 0
                if interaction_type == 'generalizable':
                    model_output = [model_output_1, model_output_2]
                sparsify_kwargs = {"interaction_type": args.interaction,
                                   "qthres": args.sparsify_qthres, "pthres": args.sparsify_pthres,
                                   "qstd": args.sparsify_qstd,
                                   "lr": args.sparsify_lr, "niter": args.sparsify_niter, "alpha": args.alpha,
                                   "average_model_output": model_output}

                if interaction_type == 'traditional':
                    model_output = get_vn_vempty(args, forward_func=forward_func,
                              selected_dim=args.selected_dim,
                              sample=(sample_seq, sample_mask, sample_segment, sentence, label),
                              baseline=baseline,
                              label=label,
                              sparsify_kwargs=sparsify_kwargs,
                              predefine_player=input_variable_indices[sample_id],
                              table=table
                              )
                    print(sample_id, model_output)
                    table = table.append({'sample_index': sample_id, 'model': model_output}, ignore_index=True)
                    print(table)
                    table.to_csv(os.path.join(MODEL_OUTPUT_OF_SAMPLE_PATH))  # (MODEL_OUTPUT_OF_SAMPLE_PATH)
                elif interaction_type == 'generalizable':
                    model_output_1, model_output_2 = get_vn_vempty(args, forward_func=[forward_func_1, forward_func_2],
                                  selected_dim=args.selected_dim,
                                  sample=(sample_seq_1, sample_mask, sample_segment, sentence, label, sample_seq_2),
                                  baseline=[baseline_1, baseline_2],
                                  label=label,
                                  sparsify_kwargs=sparsify_kwargs,
                                  predefine_player=input_variable_indices[sample_id],
                                  table=table
                                  )
                    print(sample_id, model_output_1, model_output_2)
                    table = table.append({'sample_index': sample_id, 'model_1': model_output_1, 'model_2': model_output_2},
                                 ignore_index=True)
                    print(table)
                    table.to_csv(os.path.join(MODEL_OUTPUT_OF_SAMPLE_PATH))


        table.to_csv(MODEL_OUTPUT_OF_SAMPLE_PATH)

    print("the total number of samples: {}".format(len(sample_indices[0]) + len(sample_indices[1])))

    return sample_indices, input_variable_indices



def get_average_model_output(path, interaction_type, model_str=''):
    # MODEL_OUTPUT_OF_SAMPLE_PATH = os.path.join(path, 'selected_sample_model_output.csv')
    if interaction_type == 'traditional':
        MODEL_OUTPUT_OF_SAMPLE_PATH = os.path.join(path, interaction_type, model_str, 'selected_sample_model_output.csv')
    elif interaction_type == 'generalizable':
        MODEL_OUTPUT_OF_SAMPLE_PATH = os.path.join(path, interaction_type, 'selected_sample_model_output.csv')
    print(MODEL_OUTPUT_OF_SAMPLE_PATH)

    df = pd.read_csv(MODEL_OUTPUT_OF_SAMPLE_PATH)
    if interaction_type == 'traditional':
        average_model_output = df['model'].mean()
        return average_model_output
    elif interaction_type == 'generalizable':
        average_model_output_1 = df['model_1'].mean()
        average_model_output_2 = df['model_2'].mean()
        return average_model_output_1, average_model_output_2




def get_correct_sample_indices(model: nn.Module,
                               seqs: torch.Tensor,
                               masks: torch.Tensor,
                               segments: torch.Tensor,
                               labels: torch.Tensor,
                               bs: int = 1):
    """
    Get samples which are correctly classified by the model

    :param net:
    :param seqs:
    :param masks:
    :param segments:
    :param labels:
    :param bs:
    :return:
    """
    device = next(model.parameters()).device
    n_sample = seqs.shape[0]
    n_batch = int(np.ceil(n_sample / bs))
    correct_indices = []
    for i in range(n_batch):
        seq_batch = seqs[i * bs:(i + 1) * bs].clone()
        mask_batch = masks[i * bs:(i + 1) * bs].clone()
        segment_batch = segments[i * bs:(i + 1) * bs].clone()
        y_batch = labels[i * bs:(i + 1) * bs].clone()
        seq_batch, mask_batch, segment_batch, y_batch = seq_batch.to(device), mask_batch.to(device), segment_batch.to(
            device), y_batch.to(device)

        _, _, probabilities = model(seq_batch, mask_batch, segment_batch, y_batch)

        pred_batch = torch.argmax(probabilities, dim=-1)
        correct_indices.append(torch.arange(i * bs, i * bs + seq_batch.shape[0])[pred_batch == y_batch])

    return torch.cat(correct_indices).tolist()


def get_sample_indices_each_class(sample_indices: List, labels: torch.Tensor, masks: torch.Tensor,
                                  n_sample_each_class: int = 200, selected_classes=None):
    """

    :param sample_indices: input correctly classified samples
    :param labels:
    :param masks:
    :param n_sample_each_class:
    :param selected_classes:
    :return:
    """

    if n_sample_each_class is None:
        n_sample_each_class = 200
    if selected_classes is None:
        selected_classes = sorted(torch.unique(labels).tolist())

    slots = {class_id: n_sample_each_class for class_id in torch.unique(labels).tolist()}

    for class_id in slots.keys():
        if class_id not in selected_classes:
            slots[class_id] = 0

    selected = {class_id: [] for class_id in selected_classes}
    for i in sample_indices:
        _label = labels[i].item()  # get class-id: 0 or 1
        if masks[i].sum() > 20:  # choose those sentences whose token > 10
            if slots[_label] > 0:
                slots[_label] -= 1
                selected[_label].append(i)
            if sum(list(slots.values())) == 0:
                break
    return selected


def get_all_players_inds(_sentence, tokenizer, stop_words, max_seq_len=50):
        
    players = tokenizer.tokenize(_sentence)
    players = [player for player in players if player not in stop_words]
    # print("tokenize: ", players)
    # print("\n")
    if len(players) < 10:
        return []
    else:
        all_players = np.arange(1, min(len(players) + 1, max_seq_len))  # exclude the first token [CLS]
        _selected_players = np.random.choice(all_players, 10, replace=False)
        _selected_players = np.sort(_selected_players)
        return _selected_players
    





def get_vn_vempty(args,
                  forward_func, selected_dim,
                  sample, baseline, label,
                  sparsify_kwargs,
                  predefine_player=None, table=None
                  ):
    bs, n_words, hidden_dim = sample[0].shape
    assert bs == 1
    mask_input_fn = get_mask_input_func_nlp()
    if predefine_player is None:
        raise NotImplementedError(
            "we should use predifined players, in order to filter out meaningless token, and be reproducible.")
    else:
        all_players = np.array(predefine_player)


    foreground = list(flatten(all_players))
    indices = np.ones(n_words, dtype=bool)  # n_words = 50, max_seq_len
    indices[foreground] = False
    background = np.arange(n_words)[indices].tolist()

    # calculate interaction
    calculator = AndOrHarsanyi(
        interaction_type=args.interaction,
        model=forward_func, selected_dim=selected_dim,
        x=sample, baseline=baseline, y=label,
        all_players=all_players, background=background,
        mask_input_fn=mask_input_fn, calc_bs=None, verbose=0
    )
    with torch.no_grad():
        calculator.attribute()

    sparsifier = AndOrHarsanyiSparsifier(calculator=calculator, **sparsify_kwargs)

    if args.interaction == 'traditional':
        model_output = sparsifier.get_vN_vEmpty()
        model_output = float(model_output.detach().cpu().numpy())
        return model_output
    elif args.interaction == 'generalizable':
        model_output_1, model_output_2 = sparsifier.get_vN_vEmpty()
        model_output_1 = float(model_output_1.detach().cpu().numpy())
        model_output_2 = float(model_output_2.detach().cpu().numpy())
        return model_output_1, model_output_2
