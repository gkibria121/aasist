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
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA
from tqdm.notebook import tqdm

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
    # Start timing
    total_start_time = time.time()
    
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
    print("\n" + "="*60)
    print("SAFE OPTIMIZATIONS ENABLED:")
    print("="*60) 
    print("="*60 + "\n")
    # ============================================================

    # Define model architecture
    model = get_model(model_config, device)

    # Define dataloaders
    print("Loading datasets...")
    trn_loader, dev_loader, eval_loader = get_loader(
        database_path, args.seed, config)
    print(f"✓ Datasets loaded. Training batches: {len(trn_loader)}")

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

    # Training loop with notebook-style progress
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print(f"Epochs: {config['num_epochs']}")
    print("="*60 + "\n")
    
    # Create main progress bar for epochs
    epoch_pbar = tqdm(range(config["num_epochs"]), desc="📊 Epochs", unit="epoch")
    
    for epoch in epoch_pbar:
        epoch_start_time = time.time()
        
        # Update epoch description
        epoch_pbar.set_description(f"📊 Epoch {epoch+1}/{config['num_epochs']}")
        
        # ========== TRAIN PHASE ==========
        train_start_time = time.time()
        running_loss = train_epoch(trn_loader, model, optimizer, device,
                                   scheduler, config, epoch)
        train_time = time.time() - train_start_time
        
        # ========== VALIDATION PHASE ==========
        val_start_time = time.time()
        
        # Perform full validation
        produce_evaluation_file(dev_loader, model, device,
                                metric_path/"dev_score.txt", dev_trial_path)
        dev_eer, dev_tdcf, dev_acc_eer, dev_acc_tdcf, dev_max_acc = calculate_tDCF_EER(
            cm_scores_file=metric_path/"dev_score.txt",
            asv_score_file=database_path/config["asv_score_path"],
            output_file=metric_path/"dev_t-DCF_EER_{}epo.txt".format(epoch),
            printout=False)
        
        val_time = time.time() - val_start_time
        epoch_time = time.time() - epoch_start_time
        
        # Update progress bar postfix with metrics
        epoch_pbar.set_postfix({
            'Loss': f'{running_loss:.4f}',
            'EER': f'{dev_eer:.2f}%',
            't-DCF': f'{dev_tdcf:.4f}',
            'Time': f'{epoch_time:.0f}s'
        })
        
        writer.add_scalar("loss", running_loss, epoch)
        writer.add_scalar("dev_eer", dev_eer, epoch)
        writer.add_scalar("dev_tdcf", dev_tdcf, epoch)
        writer.add_scalar("dev_acc_eer", dev_acc_eer, epoch)
        writer.add_scalar("dev_acc_tdcf", dev_acc_tdcf, epoch)
        writer.add_scalar("dev_max_acc", dev_max_acc, epoch)

        best_dev_tdcf = min(dev_tdcf, best_dev_tdcf)
        if best_dev_eer >= dev_eer:
            # Update best metrics in progress bar
            epoch_pbar.write(f"\n🎯 NEW BEST! EER: {dev_eer:.3f}% (prev: {best_dev_eer:.3f}%)")
            best_dev_eer = dev_eer
            torch.save(model.state_dict(),
                       model_save_path / "epoch_{}_{:03.3f}.pth".format(epoch, dev_eer))

            # Do evaluation whenever best model is renewed
            if str_to_bool(config["eval_all_best"]):
                epoch_pbar.write("   🔍 Evaluating on eval set...")
                eval_start_time = time.time()
                produce_evaluation_file(eval_loader, model, device,
                                        eval_score_path, eval_trial_path)
                eval_eer, eval_tdcf, eval_acc_eer, eval_acc_tdcf, eval_max_acc = calculate_tDCF_EER(
                    cm_scores_file=eval_score_path,
                    asv_score_file=database_path / config["asv_score_path"],
                    output_file=metric_path /
                    "t-DCF_EER_{:03d}epo.txt".format(epoch))

                eval_time = time.time() - eval_start_time
                
                log_text = f"   📊 Eval: EER={eval_eer:.3f}%, t-DCF={eval_tdcf:.5f}"
                if eval_eer < best_eval_eer:
                    log_text += f" (🏆 NEW BEST EER)"
                    best_eval_eer = eval_eer
                if eval_tdcf < best_eval_tdcf:
                    log_text += f" (🏆 NEW BEST t-DCF)"
                    best_eval_tdcf = eval_tdcf
                    torch.save(model.state_dict(),
                               model_save_path / "best.pth")
                
                log_text += f"\n   🎯 Acc@EER: {eval_acc_eer:.2f}%, Acc@min-tDCF: {eval_acc_tdcf:.2f}%, Max Acc: {eval_max_acc:.2f}%"
                
                epoch_pbar.write(log_text)
                f_log.write(f"epoch{epoch+1:03d}: {log_text}\n")

            epoch_pbar.write(f"   💾 Saved for SWA")
            optimizer_swa.update_swa()
            n_swa_update += 1
        
        writer.add_scalar("best_dev_eer", best_dev_eer, epoch)
        writer.add_scalar("best_dev_tdcf", best_dev_tdcf, epoch)
        
        # Clear memory if needed
        if device == "cuda":
            torch.cuda.empty_cache()
    
    epoch_pbar.close()
    
    print("\n" + "="*60)
    print("🎯 FINAL EVALUATION")
    print("="*60)
    
    if n_swa_update > 0:
        print("🔁 Applying SWA (Stochastic Weight Averaging)...")
        optimizer_swa.swap_swa_sgd()
        optimizer_swa.bn_update(trn_loader, model, device=device)
    
    print("📊 Final evaluation on eval set...")
    eval_start_time = time.time()
    
    # Create final evaluation progress bar
    eval_pbar = tqdm(total=len(eval_loader), desc="🔍 Final Evaluation", unit="batch")
    
    # Monkey patch the produce_evaluation_file to show progress
    original_produce = produce_evaluation_file
    def tracked_produce_evaluation_file(*args, **kwargs):
        # We'll create a custom version for final eval
        model.eval()
        data_loader, model, device, save_path, trial_path = args
        with open(trial_path, "r") as f_trl:
            trial_lines = f_trl.readlines()
        fname_list = []
        score_list = []
        
        with torch.no_grad():
            for batch_x, utt_id in data_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                _, batch_out = model(batch_x)
                batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
                fname_list.extend(utt_id)
                score_list.extend(batch_score.tolist())
                eval_pbar.update(1)
        
        eval_pbar.close()
        
        assert len(trial_lines) == len(fname_list) == len(score_list)
        with open(save_path, "w") as fh:
            for fn, sco, trl in zip(fname_list, score_list, trial_lines):
                _, utt_id, _, src, key = trl.strip().split(' ')
                assert fn == utt_id
                fh.write("{} {} {} {}\n".format(utt_id, src, key, sco))
    
    produce_evaluation_file = tracked_produce_evaluation_file
    produce_evaluation_file(eval_loader, model, device, eval_score_path,
                            eval_trial_path)
    
    eval_eer, eval_tdcf, eval_acc_eer, eval_acc_tdcf, eval_max_acc = calculate_tDCF_EER(
        cm_scores_file=eval_score_path,
        asv_score_file=database_path / config["asv_score_path"],
        output_file=model_tag / "t-DCF_EER.txt")
    eval_time = time.time() - eval_start_time
    
    f_log = open(model_tag / "metric_log.txt", "a")
    f_log.write("=" * 5 + "\n")
    f_log.write("FINAL RESULTS:\n")
    f_log.write(f"EER: {eval_eer:.3f}%, min t-DCF: {eval_tdcf:.5f}\n")
    f_log.write(f"Acc@EER: {eval_acc_eer:.2f}%, Acc@min-tDCF: {eval_acc_tdcf:.2f}%, Max Acc: {eval_max_acc:.2f}%\n")
    f_log.close()

    torch.save(model.state_dict(),
               model_save_path / "swa.pth")

    if eval_eer <= best_eval_eer:
        best_eval_eer = eval_eer
    if eval_tdcf <= best_eval_tdcf:
        best_eval_tdcf = eval_tdcf
        torch.save(model.state_dict(),
                   model_save_path / "best.pth")
    
    total_time = time.time() - total_start_time
    
    # Create final summary with emojis
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE!")
    print("="*60)
    print(f"  🏆 Best EER:                 {best_eval_eer:.3f}%")
    print(f"  🏆 Best t-DCF:               {best_eval_tdcf:.5f}")
    print(f"  📊 Final Accuracy @ EER:     {eval_acc_eer:.2f}%")
    print(f"  📊 Final Accuracy @ min-tDCF:{eval_acc_tdcf:.2f}%")
    print(f"  📊 Final Maximum Accuracy:   {eval_max_acc:.2f}%")
    print(f"  ⏱️  Total Training Time:      {total_time:.1f}s")
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
    trn_loader = DataLoader(train_set,
                            batch_size=config["batch_size"],
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True,
                            num_workers=0,  # Start with 0 to avoid blocking
                            worker_init_fn=seed_worker if config.get("num_workers", 0) > 0 else None,
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
                            num_workers=0)

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
                             num_workers=0)

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
    
    # Notebook-style progress bar for evaluation
    eval_pbar = tqdm(data_loader, desc="🔍 Evaluating", unit="batch")
    
    with torch.no_grad():
        for batch_x, utt_id in eval_pbar:
            batch_x = batch_x.to(device, non_blocking=True)
            _, batch_out = model(batch_x)
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
            fname_list.extend(utt_id)
            score_list.extend(batch_score.tolist())
            
            # Update progress
            eval_pbar.set_postfix({'samples': len(fname_list)})

    assert len(trial_lines) == len(fname_list) == len(score_list)
    with open(save_path, "w") as fh:
        for fn, sco, trl in zip(fname_list, score_list, trial_lines):
            _, utt_id, _, src, key = trl.strip().split(' ')
            assert fn == utt_id
            fh.write("{} {} {} {}\n".format(utt_id, src, key, sco))


def train_epoch(
    trn_loader: DataLoader,
    model,
    optim: Union[torch.optim.SGD, torch.optim.Adam],
    device: torch.device,
    scheduler: torch.optim.lr_scheduler,
    config: argparse.Namespace,
    epoch: int):
    """Train the model for one epoch with notebook-style progress bar"""
    running_loss = 0
    num_total = 0.0
    model.train()

    # Set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    
    # Notebook-style training progress bar
    train_pbar = tqdm(trn_loader, desc="🚀 Training", unit="batch")
    
    for batch_idx, (batch_x, batch_y) in enumerate(train_pbar):
        batch_size = batch_x.size(0)
        num_total += batch_size
        
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
        
        # Update progress bar with rich information
        current_loss = running_loss / num_total
        current_lr = optim.param_groups[0].get("lr", 0)
        
        train_pbar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'lr': f'{current_lr:.6f}' if current_lr > 0 else 'N/A',
            'batch': f'{batch_idx+1}/{len(trn_loader)}'
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