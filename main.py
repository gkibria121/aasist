"""
Main script that trains, validates, and evaluates
various models including AASIST.

AASIST
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import argparse
import json
import os
import sys
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import Dict, List, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA
from tqdm import tqdm

from data_utils import (Dataset_ASVspoof2019_train,
                        Dataset_ASVspoof2019_devNeval, genSpoof_list)
from evaluation import calculate_tDCF_EER
from utils import create_optimizer, seed_worker, set_seed, str_to_bool

warnings.filterwarnings("ignore", category=FutureWarning)


def main(args: argparse.Namespace) -> None:
    """
    Main function.
    Trains, validates, and evaluates the ASVspoof detection model.
    """
    # Load experiment configurations
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]
    track = config["track"]
    assert track in ["LA", "PA", "DF"], "Invalid track given"
    if "eval_all_best" not in config:
        config["eval_all_best"] = "True"
    if "freq_aug" not in config:
        config["freq_aug"] = "False"

    # Make experiment reproducible
    set_seed(args.seed, config)

    # Define database related paths
    output_dir = Path(args.output_dir)
    prefix_2019 = "ASVspoof2019.{}".format(track)
    database_path = Path(config["database_path"])
    dev_trial_path = (database_path /
                      "ASVspoof2019_{}_cm_protocols/{}.cm.dev.trl.txt".format(
                          track, prefix_2019))
    eval_trial_path = (
        database_path /
        "ASVspoof2019_{}_cm_protocols/{}.cm.eval.trl.txt".format(
            track, prefix_2019))

    # Define model related paths
    model_tag = "{}_{}_ep{}_bs{}".format(
        track,
        os.path.splitext(os.path.basename(args.config))[0],
        config["num_epochs"], config["batch_size"])
    if args.comment:
        model_tag = model_tag + "_{}".format(args.comment)
    model_tag = output_dir / model_tag
    model_save_path = model_tag / "weights"
    eval_score_path = model_tag / config["eval_output"]
    writer = SummaryWriter(model_tag)
    os.makedirs(model_save_path, exist_ok=True)
    copy(args.config, model_tag / "config.conf")

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))
    if device == "cpu":
        raise ValueError("GPU not detected!")

    # ==================== SAFE OPTIMIZATIONS ====================
    # These optimizations do NOT change numerical results
    print("\n" + "="*60)
    print("SAFE OPTIMIZATIONS ENABLED:")
    print("="*60)
    
    # Enable cuDNN autotuner - finds fastest convolution algorithms
    torch.backends.cudnn.benchmark = True 
    print("="*60 + "\n")
    # ============================================================

    # Define model architecture
    model = get_model(model_config, device)

    # Define dataloaders
    trn_loader, dev_loader, eval_loader = get_loader(
        database_path, args.seed, config)

    # Evaluates pretrained model and exit script
    if args.eval:
        model.load_state_dict(
            torch.load(config["model_path"], map_location=device))
        print("Model loaded : {}".format(config["model_path"]))
        print("Start evaluation...")
        produce_evaluation_file(eval_loader, model, device,
                                eval_score_path, eval_trial_path)
        calculate_tDCF_EER(cm_scores_file=eval_score_path,
                           asv_score_file=database_path /
                           config["asv_score_path"],
                           output_file=model_tag / "t-DCF_EER.txt")
        print("DONE.")
        eval_eer, eval_tdcf, eval_acc_eer, eval_acc_tdcf, eval_max_acc = calculate_tDCF_EER(
            cm_scores_file=eval_score_path,
            asv_score_file=database_path / config["asv_score_path"],
            output_file=model_tag/"loaded_model_t-DCF_EER.txt")
        sys.exit(0)

    # Get optimizer and scheduler
    optim_config["steps_per_epoch"] = len(trn_loader)
    optimizer, scheduler = create_optimizer(model.parameters(), optim_config)
    optimizer_swa = SWA(optimizer)

    best_dev_eer = 1.
    best_eval_eer = 100.
    best_dev_tdcf = 0.05
    best_eval_tdcf = 1.
    n_swa_update = 0  # Number of snapshots of model to use in SWA
    f_log = open(model_tag / "metric_log.txt", "a")
    f_log.write("=" * 5 + "\n")

    # Make directory for metric logging
    metric_path = model_tag / "metrics"
    os.makedirs(metric_path, exist_ok=True)

    # Training with progress bar
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    epoch_pbar = tqdm(range(config["num_epochs"]), 
                      desc="Training Progress", 
                      position=0,
                      ncols=100,
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    
    for epoch in epoch_pbar:
        # Update epoch description
        epoch_pbar.set_description(f"Epoch {epoch+1}/{config['num_epochs']}")
        
        # Train for one epoch
        running_loss = train_epoch(trn_loader, model, optimizer, device,
                                   scheduler, config, epoch)
        
        # Validation with progress info
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config['num_epochs']} - Validating...")
        print(f"{'='*60}")
        
        produce_evaluation_file(dev_loader, model, device,
                                metric_path/"dev_score.txt", dev_trial_path)
        dev_eer, dev_tdcf, dev_acc_eer, dev_acc_tdcf, dev_max_acc = calculate_tDCF_EER(
            cm_scores_file=metric_path/"dev_score.txt",
            asv_score_file=database_path/config["asv_score_path"],
            output_file=metric_path/"dev_t-DCF_EER_{}epo.txt".format(epoch),
            printout=False)
        
        print(f"Loss: {running_loss:.5f} | Dev EER: {dev_eer:.3f}% | Dev t-DCF: {dev_tdcf:.5f}")
        print(f"Dev Acc@EER: {dev_acc_eer:.2f}% | Dev Acc@min-tDCF: {dev_acc_tdcf:.2f}% | Dev Max Acc: {dev_max_acc:.2f}%")
        
        writer.add_scalar("loss", running_loss, epoch)
        writer.add_scalar("dev_eer", dev_eer, epoch)
        writer.add_scalar("dev_tdcf", dev_tdcf, epoch)
        writer.add_scalar("dev_acc_eer", dev_acc_eer, epoch)
        writer.add_scalar("dev_acc_tdcf", dev_acc_tdcf, epoch)
        writer.add_scalar("dev_max_acc", dev_max_acc, epoch)

        best_dev_tdcf = min(dev_tdcf, best_dev_tdcf)
        if best_dev_eer >= dev_eer:
            print(f"🎯 NEW BEST MODEL at epoch {epoch}! EER: {dev_eer:.3f}%")
            best_dev_eer = dev_eer
            torch.save(model.state_dict(),
                       model_save_path / "epoch_{}_{:03.3f}.pth".format(epoch, dev_eer))

            # Do evaluation whenever best model is renewed
            if str_to_bool(config["eval_all_best"]):
                print("Evaluating on eval set...")
                produce_evaluation_file(eval_loader, model, device,
                                        eval_score_path, eval_trial_path)
                eval_eer, eval_tdcf, eval_acc_eer, eval_acc_tdcf, eval_max_acc = calculate_tDCF_EER(
                    cm_scores_file=eval_score_path,
                    asv_score_file=database_path / config["asv_score_path"],
                    output_file=metric_path /
                    "t-DCF_EER_{:03d}epo.txt".format(epoch))

                log_text = "epoch{:03d}, ".format(epoch)
                if eval_eer < best_eval_eer:
                    log_text += "best eer, {:.4f}%".format(eval_eer)
                    best_eval_eer = eval_eer
                if eval_tdcf < best_eval_tdcf:
                    log_text += "best tdcf, {:.4f}".format(eval_tdcf)
                    best_eval_tdcf = eval_tdcf
                    torch.save(model.state_dict(),
                               model_save_path / "best.pth")
                
                # Add accuracy information to log
                log_text += f", acc@eer: {eval_acc_eer:.2f}%, acc@min-tDCF: {eval_acc_tdcf:.2f}%, max_acc: {eval_max_acc:.2f}%"
                
                if len(log_text) > 0:
                    print(log_text)
                    f_log.write(log_text + "\n")

            print(f"Saving epoch {epoch} for SWA")
            optimizer_swa.update_swa()
            n_swa_update += 1
        
        # Update progress bar postfix
        epoch_pbar.set_postfix({
            'loss': f'{running_loss:.4f}',
            'dev_eer': f'{dev_eer:.3f}%',
            'acc': f'{dev_acc_eer:.1f}%',
            'best': f'{best_dev_eer:.3f}%'
        })
        
        writer.add_scalar("best_dev_eer", best_dev_eer, epoch)
        writer.add_scalar("best_dev_tdcf", best_dev_tdcf, epoch)
        print()  # Add spacing between epochs

    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    epoch += 1
    if n_swa_update > 0:
        print("Applying SWA (Stochastic Weight Averaging)...")
        optimizer_swa.swap_swa_sgd()
        optimizer_swa.bn_update(trn_loader, model, device=device)
    
    produce_evaluation_file(eval_loader, model, device, eval_score_path,
                            eval_trial_path)
    eval_eer, eval_tdcf, eval_acc_eer, eval_acc_tdcf, eval_max_acc = calculate_tDCF_EER(
        cm_scores_file=eval_score_path,
        asv_score_file=database_path / config["asv_score_path"],
        output_file=model_tag / "t-DCF_EER.txt")
    
    f_log = open(model_tag / "metric_log.txt", "a")
    f_log.write("=" * 5 + "\n")
    f_log.write("EER: {:.3f}%, min t-DCF: {:.5f}\n".format(eval_eer, eval_tdcf))
    f_log.write("Acc@EER: {:.2f}%, Acc@min-tDCF: {:.2f}%, Max Acc: {:.2f}%\n".format(
        eval_acc_eer, eval_acc_tdcf, eval_max_acc))
    f_log.close()

    torch.save(model.state_dict(),
               model_save_path / "swa.pth")

    if eval_eer <= best_eval_eer:
        best_eval_eer = eval_eer
    if eval_tdcf <= best_eval_tdcf:
        best_eval_tdcf = eval_tdcf
        torch.save(model.state_dict(),
                   model_save_path / "best.pth")
    
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE!")
    print("="*60)
    print(f"Best EER: {best_eval_eer:.3f}%")
    print(f"Best t-DCF: {best_eval_tdcf:.5f}")
    print(f"Final Accuracy @ EER: {eval_acc_eer:.2f}%")
    print(f"Final Accuracy @ min-tDCF: {eval_acc_tdcf:.2f}%")
    print(f"Final Maximum Accuracy: {eval_max_acc:.2f}%")
    print("="*60 + "\n")


def get_model(model_config: Dict, device: torch.device):
    """Define DNN model architecture"""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("Number of model parameters: {}".format(nb_params))

    return model


def get_loader(
        database_path: str,
        seed: int,
        config: dict) -> List[torch.utils.data.DataLoader]:
    """Make PyTorch DataLoaders for train / development / evaluation"""
    track = config["track"]
    prefix_2019 = "ASVspoof2019.{}".format(track)

    trn_database_path = database_path / "ASVspoof2019_{}_train/".format(track)
    dev_database_path = database_path / "ASVspoof2019_{}_dev/".format(track)
    eval_database_path = database_path / "ASVspoof2019_{}_eval/".format(track)

    trn_list_path = (database_path /
                     "ASVspoof2019_{}_cm_protocols/{}.cm.train.trn.txt".format(
                         track, prefix_2019))
    dev_trial_path = (database_path /
                      "ASVspoof2019_{}_cm_protocols/{}.cm.dev.trl.txt".format(
                          track, prefix_2019))
    eval_trial_path = (
        database_path /
        "ASVspoof2019_{}_cm_protocols/{}.cm.eval.trl.txt".format(
            track, prefix_2019))

    d_label_trn, file_train = genSpoof_list(dir_meta=trn_list_path,
                                            is_train=True,
                                            is_eval=False)
    print("Number of training files:", len(file_train))

    train_set = Dataset_ASVspoof2019_train(list_IDs=file_train,
                                           labels=d_label_trn,
                                           base_dir=trn_database_path)
    gen = torch.Generator()
    gen.manual_seed(seed)
    
    # SAFE OPTIMIZATION: Multi-worker data loading with prefetching
    # Does NOT change results, only speeds up data loading
    trn_loader = DataLoader(train_set,
                            batch_size=config["batch_size"],
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True,
                            num_workers=2,  # Parallel data loading
                            prefetch_factor=2,  # Prefetch 2 batches per worker
                            persistent_workers=True,  # Keep workers alive between epochs
                            worker_init_fn=seed_worker,
                            generator=gen)

    _, file_dev = genSpoof_list(dir_meta=dev_trial_path,
                                is_train=False,
                                is_eval=False)
    print("Number of validation files:", len(file_dev))

    dev_set = Dataset_ASVspoof2019_devNeval(list_IDs=file_dev,
                                            base_dir=dev_database_path)
    dev_loader = DataLoader(dev_set,
                            batch_size=config["batch_size"],
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=2,  # Fewer workers for eval
                            prefetch_factor=2)

    file_eval = genSpoof_list(dir_meta=eval_trial_path,
                              is_train=False,
                              is_eval=True)
    eval_set = Dataset_ASVspoof2019_devNeval(list_IDs=file_eval,
                                             base_dir=eval_database_path)
    eval_loader = DataLoader(eval_set,
                             batch_size=config["batch_size"],
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True,
                             num_workers=2,
                             prefetch_factor=2)

    return trn_loader, dev_loader, eval_loader


def produce_evaluation_file(
    data_loader: DataLoader,
    model,
    device: torch.device,
    save_path: str,
    trial_path: str) -> None:
    """Perform evaluation and save the score to a file"""
    model.eval()
    with open(trial_path, "r") as f_trl:
        trial_lines = f_trl.readlines()
    fname_list = []
    score_list = []
    
    # Progress bar for evaluation with better visibility
    eval_pbar = tqdm(data_loader, 
                     desc="Evaluating", 
                     ncols=100,
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    # SAFE OPTIMIZATION: Move torch.no_grad() outside loop
    # Reduces overhead, does NOT change results
    with torch.no_grad():
        for batch_x, utt_id in eval_pbar:
            # SAFE OPTIMIZATION: non_blocking=True allows CPU/GPU parallelism
            # Does NOT change results, only speeds up transfer
            batch_x = batch_x.to(device, non_blocking=True)
            _, batch_out = model(batch_x)
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
            # Add outputs
            fname_list.extend(utt_id)
            score_list.extend(batch_score.tolist())

    assert len(trial_lines) == len(fname_list) == len(score_list)
    with open(save_path, "w") as fh:
        for fn, sco, trl in zip(fname_list, score_list, trial_lines):
            _, utt_id, _, src, key = trl.strip().split(' ')
            assert fn == utt_id
            fh.write("{} {} {} {}\n".format(utt_id, src, key, sco))
    print("Scores saved to {}".format(save_path))


def train_epoch(
    trn_loader: DataLoader,
    model,
    optim: Union[torch.optim.SGD, torch.optim.Adam],
    device: torch.device,
    scheduler: torch.optim.lr_scheduler,
    config: argparse.Namespace,
    epoch: int):
    """Train the model for one epoch"""
    running_loss = 0
    num_total = 0.0
    model.train()

    # Set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    
    # Progress bar for training with better visibility
    train_pbar = tqdm(trn_loader, 
                      desc=f"Training Epoch {epoch+1}", 
                      ncols=100,
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')
    
    for batch_x, batch_y in train_pbar:
        batch_size = batch_x.size(0)
        num_total += batch_size
        
        # SAFE OPTIMIZATION: non_blocking=True for parallel CPU/GPU work
        # Does NOT change results
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.view(-1).type(torch.int64).to(device, non_blocking=True)
        
        _, batch_out = model(batch_x, Freq_aug=str_to_bool(config["freq_aug"]))
        batch_loss = criterion(batch_out, batch_y)
        running_loss += batch_loss.item() * batch_size
        optim.zero_grad()
        batch_loss.backward()
        optim.step()

        if config["optim_config"]["scheduler"] in ["cosine", "keras_decay"]:
            scheduler.step()
        elif scheduler is None:
            pass
        else:
            raise ValueError("scheduler error, got:{}".format(scheduler))
        
        # Update progress bar with current loss and average loss
        avg_loss = running_loss / num_total
        train_pbar.set_postfix({
            'batch_loss': f'{batch_loss.item():.4f}',
            'avg_loss': f'{avg_loss:.4f}'
        })

    running_loss /= num_total
    return running_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASVspoof detection system")
    parser.add_argument("--config",
                        dest="config",
                        type=str,
                        help="configuration file",
                        required=True)
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        help="output directory for results",
        default="./exp_result",
    )
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="random seed (default: 1234)")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="when this flag is given, evaluates given model and exit")
    parser.add_argument("--comment",
                        type=str,
                        default=None,
                        help="comment to describe the saved model")
    parser.add_argument("--eval_model_weights",
                        type=str,
                        default=None,
                        help="directory to the model weight file (can be also given in the config file)")
    main(parser.parse_args())