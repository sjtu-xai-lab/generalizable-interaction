import os
import os.path as osp
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np

from harsanyi.and_or_harsanyi import AndOrHarsanyi, AndOrHarsanyiSparsifier
from harsanyi.interaction_utils import get_mask_input_func_nlp, flatten, reorganize_and_or_harsanyi
from baseline_values.baseline_value import get_baseline_value
from utils.data import get_model_data
from utils.util import setup_seed, get_stop_words, get_samples_and_input_variables, get_average_model_output

global stop_words


class ForwardFunction(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 device, ):
        super(ForwardFunction, self).__init__()

        self.model = model
        self.device = device
        self.tokenizer = model.tokenizer

    def _get_embedding(self, input_ids):
        with torch.no_grad():
            word_embeddings = self.model.bert.bert.embeddings.word_embeddings(input_ids)
        return word_embeddings

    def forward(self, masked_embeddings, batch_seq_masks, segments, labels):
        # using input embeddings instead of input ids
        loss, logits = self.model.bert(attention_mask=batch_seq_masks,
                                       token_type_ids=segments,
                                       labels=labels,
                                       inputs_embeds=masked_embeddings)[:2]

        return logits



def evaluate_single_sample(args,
                            forward_func, selected_dim,
                            sample, baseline, label,
                            sparsify_kwargs, save_folder,
                            predefined_player=None
                            ):
    bs, n_words, hidden_dim = sample[0].shape
    assert bs == 1
    mask_input_fn = get_mask_input_func_nlp()

    # get selected input variables and background variables
    if predefined_player is None:
        raise NotImplementedError("Should use pre-defined players.")
    else:
        all_players = np.array(predefined_player)

    if args.interaction == 'traditional':
        players = [player for player in forward_func.tokenizer.tokenize(sample[3]) if player not in stop_words]
        selected_words = ['[CLS]'] + players
    elif args.interaction == 'generalizable':
        players = [player for player in forward_func[0].tokenizer.tokenize(sample[3]) if player not in stop_words]
        selected_words = ['[CLS]'] + players

    with open(osp.join(save_folder, "sample_and_input_variable.txt"), 'a') as f:
        f.write(f'{np.array(selected_words)[all_players]} \n')

    foreground = list(flatten(all_players))
    indices = np.ones(n_words, dtype=bool)
    indices[foreground] = False
    background = np.arange(n_words)[indices].tolist()

    # calculate AND-OR interactions
    calculator = AndOrHarsanyi(
        interaction_type=args.interaction,
        model=forward_func, selected_dim=selected_dim,
        x=sample, baseline=baseline, y=label,
        all_players=all_players, background=background,
        mask_input_fn=mask_input_fn, calc_bs=None, verbose=0
    )
    with torch.no_grad():
        calculator.attribute()
        masks = calculator.get_masks()
        np.save('masks.npy', masks.cpu().numpy())

    # optimize AND-OR interactions
    sparsifier = AndOrHarsanyiSparsifier(calculator=calculator, **sparsify_kwargs)
    sparsifier.sparsify(verbose_folder=osp.join(save_folder))

    with torch.no_grad():
        if args.interaction == 'traditional':
            I_and, I_or = sparsifier.get_interaction()
            I_and, I_or = reorganize_and_or_harsanyi(masks, I_and, I_or)
            sparsifier.save(save_folder=osp.join(save_folder))
            torch.save(I_and.squeeze(), osp.join(save_folder, "I_and.pth"))
            torch.save(I_or.squeeze(), osp.join(save_folder, "I_or.pth"))

            return torch.cat([I_and, I_or])

        elif args.interaction == 'generalizable':
            I_and_1, I_or_1, I_and_2, I_or_2 = sparsifier.get_interaction()
            I_and_1, I_or_1 = reorganize_and_or_harsanyi(masks, I_and_1, I_or_1)
            I_and_2, I_or_2 = reorganize_and_or_harsanyi(masks, I_and_2, I_or_2)
            sparsifier.save(save_folder=osp.join(save_folder))
            torch.save(I_and_1.squeeze(), osp.join(save_folder, "I_and_1.pth"))
            torch.save(I_or_1.squeeze(), osp.join(save_folder, "I_or_1.pth"))
            torch.save(I_and_2.squeeze(), osp.join(save_folder, "I_and_2.pth"))
            torch.save(I_or_2.squeeze(), osp.join(save_folder, "I_or_2.pth"))

            return torch.cat([I_and_1, I_or_1]), torch.cat([I_and_2, I_or_2])



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='compute interactions (on text datasets)')

    parser.add_argument("--interaction", type=str, default='traditional',
                       help="choose to compute 'traditional' or 'generalizable' interactions in the paper")

    parser.add_argument("--dataset", type=str, default='SST-2')
    parser.add_argument("--model_dir", type=str, default='./pretrained_model/task1/',
                        help="directory of pre-trained models and baseline values")
    parser.add_argument("--model_path", type=str, default='BERT-base.pth.tar',
                        help="only to compute 'traditional' interactions on a single DNN")
    parser.add_argument("--model_path_1", type=str, default='BERT-base.pth.tar',
                        help="only to compute 'generalizable' interactions on two DNNs")
    parser.add_argument("--model_path_2", type=str, default='BERT-large.pth.tar',
                        help="only to compute 'generalizable' interactions on two DNNs")
    parser.add_argument("--output_path", type=str, default='./output/task1/',
                        help="path to save results")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default='cuda:0')

    # baseline value
    parser.add_argument("--baseline", type=str, default="pretrained",
                        help="masking input variables using pre-trained baseline values")
    # function v
    parser.add_argument("--selected-dim", type=str, default="gt-log-odds-v0",
                        help="choose which type of model output to compute interactions. \
                        Default: the log-odds output of the ground-truth label")

    # parameters for computing interactions
    parser.add_argument("--sparsify-qthres", default=0.02, type=float,
                        help="the threshold to bound the magnitude of q ($\epsilon$): q in [-thres*std, thres*std]. "
                             "This should be a float number, commly used: 0.02")
    parser.add_argument("--sparsify-pthres", default=0.5, type=float,
                        help="the threshold to bound the magnitude of p ($\gamma$). "
                             "This should be a float number, commly used: 0.5")
    parser.add_argument("--sparsify-qstd", default="vN-v0", type=str,
                        help="the standard to bound the magnitude of q ($\epsilon$): q in [-thres*std, thres*std]. "
                             "Commonly used: vN-v0")
    parser.add_argument("--sparsify-lr", default=1e-7, type=float,
                        help="the learning rate to learn p ($\gamma$) and q ($\epsilon$). Commonly used: depends.")
    parser.add_argument("--sparsify-niter", default=20000, type=int,
                        help="number of iterations to optimize p ($\gamma$) and q ($\epsilon$). Commonly used: 20000, 50000")
    parser.add_argument("--alpha", default=0.1, type=float,
                        help="the coefficient $\alpha$")

    args = parser.parse_args()

    setup_seed(args.seed)
    device = args.device if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.output_path, exist_ok=True)

    # get stop words
    stop_words = get_stop_words()

    # choose to compute traditional AND-OR interactions
    if args.interaction == 'traditional':
        if 'base' in args.model_path:
            model_str = 'base'
        elif 'large' in args.model_path:
            model_str = 'large'

        # initialize model and dataset
        model, train_loader, test_loader = get_model_data(args, device,
                                                          model_dir=os.path.join(args.model_dir, 'pretrained', f'bert-{model_str}-uncased'),
                                                          finetuned_model_dir=os.path.join(args.model_dir, 'finetuned'),
                                                          finetuned_model_path=args.model_path)
        forward_func = ForwardFunction(model, device)

        # initialize baseline values for masking input variables
        baseline = get_baseline_value(model_str=model_str, model_dir=args.model_dir, device=device,
                                      interaction_type=args.interaction, baseline_value_type=args.baseline)


    # choose to compute generalizable AND-OR interactions
    elif args.interaction == 'generalizable':
        # model 1
        model_str_1 = 'base'
        model_1, train_loader, test_loader = get_model_data(args, device,
                                                          model_dir=os.path.join(args.model_dir, 'pretrained', f'bert-{model_str_1}-uncased'),
                                                          finetuned_model_dir=os.path.join(args.model_dir, 'finetuned'),
                                                          finetuned_model_path=args.model_path_1)
        forward_func_1 = ForwardFunction(model_1, device)
        baseline_1 = get_baseline_value(model_str=model_str_1, model_dir=args.model_dir, device=device,
                                       interaction_type=args.interaction, baseline_value_type=args.baseline)

        # model 2
        model_str_2 = 'large'
        model_2, _, _ = get_model_data(args, device,
                                    model_dir=os.path.join(args.model_dir, 'pretrained', f'bert-{model_str_2}-uncased'),
                                    finetuned_model_dir=os.path.join(args.model_dir, 'finetuned'),
                                    finetuned_model_path=args.model_path_2)
        forward_func_2 = ForwardFunction(model_2, device)
        baseline_2 = get_baseline_value(model_str=model_str_2, model_dir=args.model_dir, device=device,
                                        interaction_type=args.interaction, baseline_value_type=args.baseline)


    # get samples and their input variables
    target_id = 0
    seqs, masks, segments, labels = [], [], [], []
    for id, (batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels, sentences) in enumerate(test_loader):
        seqs, masks, segments, labels = batch_seqs.to(device), batch_seq_masks.to(device), batch_seq_segments.to(
            device), batch_labels.to(device)
        if id == target_id:
            break
    if args.interaction == 'traditional':
        sample_indices, input_variable_indices = get_samples_and_input_variables(args=args, model=model, forward_func=forward_func,
                                                                                 baseline=baseline, interaction_type=args.interaction,
                                                                                 stop_words=stop_words, sentences=sentences,
                                                                                 seqs=seqs, masks=masks, segments=segments, labels=labels,
                                                                                 model_str=model_str)
    elif args.interaction  == 'generalizable':
        sample_indices, input_variable_indices = get_samples_and_input_variables(args=args, model=model_1, forward_func=[forward_func_1, forward_func_2],
                                                                                baseline=[baseline_1, baseline_2], interaction_type=args.interaction,
                                                                                stop_words=stop_words, sentences=sentences,
                                                                                seqs=seqs, masks=masks, segments=segments, labels=labels)
    

    #  evaluate interaction for each sample
    for class_id in sample_indices.keys():
        for sample_id in tqdm(sample_indices[class_id], desc=f"Class {class_id}", total=len(sample_indices[class_id]), leave=False, ncols=100, position=0):
            print(f"\nClass id: {class_id}, Sample id: {sample_id}")

            if args.interaction == 'traditional':
                save_folder = osp.join(args.output_path, args.interaction, model_str, f"class_{class_id}", f"sample_{sample_id}")
            elif args.interaction == 'generalizable':
                save_folder = osp.join(args.output_path, args.interaction, f"class_{class_id}", f"sample_{sample_id}")
            os.makedirs(save_folder, exist_ok=True)
            with open(osp.join(save_folder, "sample_and_input_variable.txt"), 'w') as f:
                f.write(f'{sentences[sample_id]} \n')

            # get each sample
            sample_s = seqs[sample_id].clone().unsqueeze(0).to(device)
            sample_mask = masks[sample_id].clone().unsqueeze(0).to(device)
            sample_segment = segments[sample_id].clone().unsqueeze(0).to(device)
            sentence = sentences[sample_id]
            label = labels[sample_id].clone().unsqueeze(0).to(device)

            with torch.no_grad():
                if args.interaction == 'traditional':
                    sample_seq = forward_func._get_embedding(input_ids=sample_s)
                    # get the average model output for all samples
                    average_model_output = get_average_model_output(path=args.output_path, interaction_type=args.interaction, model_str=model_str)

                elif args.interaction == 'generalizable':
                    sample_seq_1 = forward_func_1._get_embedding(input_ids=sample_s)
                    sample_seq_2 = forward_func_2._get_embedding(input_ids=sample_s)
                    # get the average model output for all samples
                    average_model_output_1, average_model_output_2 = get_average_model_output(path=args.output_path, interaction_type=args.interaction)
                    average_model_output = [average_model_output_1, average_model_output_2]

            sparsify_kwargs = {"interaction_type": args.interaction,
                               "qthres": args.sparsify_qthres, "pthres": args.sparsify_pthres, "qstd": args.sparsify_qstd,
                               "lr": args.sparsify_lr, "niter": args.sparsify_niter, "alpha": args.alpha,
                               "average_model_output": average_model_output}

            print("parameter:", sparsify_kwargs, "\n")

            if args.interaction == 'traditional':
                I_and_or = evaluate_single_sample(args,
                                             forward_func=forward_func,
                                             selected_dim=args.selected_dim,
                                             sample=(sample_seq, sample_mask, sample_segment,
                                                     sentence, label),
                                             baseline=baseline, label=label,
                                             save_folder=save_folder,
                                             sparsify_kwargs=sparsify_kwargs,
                                             predefined_player=input_variable_indices[sample_id]
                                             )
                print(f"Traditional interactions were calculated for sample {sample_id} on the model.")

            elif args.interaction == 'generalizable':
                I_and_or_1, I_and_or_2 = evaluate_single_sample(args,
                                                         forward_func=[forward_func_1, forward_func_2],
                                                         selected_dim=args.selected_dim,
                                                         sample=(sample_seq_1, sample_mask, sample_segment,
                                                             sentence, label, sample_seq_2),
                                                         baseline=[baseline_1, baseline_2], label=label,
                                                         save_folder=save_folder,
                                                         sparsify_kwargs=sparsify_kwargs,
                                                         predefined_player=input_variable_indices[sample_id]
                                                         )
                print(f"Generalizable interactions were calculated for sample {sample_id} on models 1 and 2.")


