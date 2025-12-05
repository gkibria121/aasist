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
import numpy as np
from collections import Counter, defaultdict
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

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


def analyze_database(config: Dict, database_path: Path, track: str) -> None:
    """
    Comprehensive database analysis function.
    Analyzes dataset statistics, class distributions, augmentation details, etc.
    """
    print("\n" + "="*80)
    print("DATABASE COMPREHENSIVE ANALYSIS")
    print("="*80)
    
    prefix_2019 = "ASVspoof2019.{}".format(track)
    
    # Define all protocol file paths
    protocol_files = {
        'train': database_path / f"ASVspoof2019_{track}_cm_protocols/{prefix_2019}.cm.train.trn.txt",
        'dev': database_path / f"ASVspoof2019_{track}_cm_protocols/{prefix_2019}.cm.dev.trl.txt",
        'eval': database_path / f"ASVspoof2019_{track}_cm_protocols/{prefix_2019}.cm.eval.trl.txt"
    }
    
    # Define audio directories
    audio_dirs = {
        'train': database_path / f"ASVspoof2019_{track}_train/",
        'dev': database_path / f"ASVspoof2019_{track}_dev/",
        'eval': database_path / f"ASVspoof2019_{track}_eval/"
    }
    
    # Check if directories exist
    print("\n📁 DATABASE STRUCTURE ANALYSIS:")
    print("-" * 50)
    for name, path in audio_dirs.items():
        exists = "✅" if path.exists() else "❌"
        print(f"{exists} {name}: {path}")
        if path.exists():
            try:
                # Count audio files with various extensions
                audio_extensions = ['*.flac', '*.wav', '*.mp3']
                num_files = 0
                for ext in audio_extensions:
                    num_files += len(list(path.rglob(ext)))
                print(f"   Audio files: {num_files}")
            except:
                print("   Unable to count files")
    
    print("\n📄 PROTOCOL FILES:")
    print("-" * 50)
    for name, path in protocol_files.items():
        exists = "✅" if path.exists() else "❌"
        print(f"{exists} {name}: {path}")
    
    # Analyze each split
    print("\n📊 DATASET STATISTICS BY SPLIT:")
    print("-" * 50)
    
    all_stats = {}
    for split_name, protocol_path in protocol_files.items():
        if not protocol_path.exists():
            print(f"⚠️  Missing {split_name} protocol file: {protocol_path}")
            continue
            
        print(f"\n🔍 Analyzing {split_name.upper()} set:")
        print("-" * 30)
        
        # Parse protocol file
        stats_dict = analyze_protocol_file(protocol_path, split_name, track)
        
        # Check audio file existence and get durations if possible
        if split_name in audio_dirs and audio_dirs[split_name].exists():
            audio_stats = analyze_audio_files(audio_dirs[split_name], stats_dict['utterances'])
            stats_dict.update(audio_stats)
        
        all_stats[split_name] = stats_dict
        
        # Print summary
        print(f"   Total samples: {stats_dict['total_samples']}")
        print(f"   Bonafide samples: {stats_dict['bonafide_count']} ({stats_dict['bonafide_percentage']:.1f}%)")
        print(f"   Spoof samples: {stats_dict['spoof_count']} ({stats_dict['spoof_percentage']:.1f}%)")
        
        if 'spoof_attacks' in stats_dict and stats_dict['spoof_attacks']:
            print(f"   Spoof attack types: {len(stats_dict['spoof_attacks'])}")
            print("   Attack type distribution:")
            for attack_type, count in stats_dict['spoof_attacks'].most_common(5):
                print(f"      {attack_type}: {count} samples")
        
        if 'audio_duration_stats' in stats_dict and stats_dict['audio_duration_stats']:
            dur_stats = stats_dict['audio_duration_stats']
            print(f"   Audio duration (estimated):")
            print(f"      Min: {dur_stats['min']:.2f}s")
            print(f"      Max: {dur_stats['max']:.2f}s")
            print(f"      Mean: {dur_stats['mean']:.2f}s")
            print(f"      Std: {dur_stats['std']:.2f}s")
        
        if 'missing_files' in stats_dict and stats_dict['missing_files'] > 0:
            print(f"   ⚠️  Missing audio files: {stats_dict['missing_files']}")
    
    # Cross-set analysis
    print("\n🔗 CROSS-SET ANALYSIS:")
    print("-" * 50)
    
    # Check for overlap between sets
    if all(set_name in all_stats for set_name in ['train', 'dev', 'eval']):
        train_speakers = set(all_stats['train']['speakers']) if 'speakers' in all_stats['train'] else set()
        dev_speakers = set(all_stats['dev']['speakers']) if 'speakers' in all_stats['dev'] else set()
        eval_speakers = set(all_stats['eval']['speakers']) if 'speakers' in all_stats['eval'] else set()
        
        train_dev_overlap = train_speakers.intersection(dev_speakers)
        train_eval_overlap = train_speakers.intersection(eval_speakers)
        dev_eval_overlap = dev_speakers.intersection(eval_speakers)
        
        print(f"Speaker overlap analysis:")
        print(f"   Train-Dev overlap: {len(train_dev_overlap)} speakers")
        print(f"   Train-Eval overlap: {len(train_eval_overlap)} speakers")
        print(f"   Dev-Eval overlap: {len(dev_eval_overlap)} speakers")
        
        if len(train_eval_overlap) > 0:
            print(f"   ⚠️  WARNING: {len(train_eval_overlap)} speakers appear in both train and eval sets!")
    
    # Configuration analysis
    print("\n⚙️  CONFIGURATION ANALYSIS:")
    print("-" * 50)
    print(f"Track: {track}")
    print(f"Batch size: {config.get('batch_size', 'Not specified')}")
    print(f"Number of epochs: {config.get('num_epochs', 'Not specified')}")
    
    # Augmentation analysis
    print("\n🔄 AUGMENTATION ANALYSIS:")
    print("-" * 50)
    freq_aug = str_to_bool(config.get("freq_aug", "False"))
    print(f"Frequency augmentation: {'✅ Enabled' if freq_aug else '❌ Disabled'}")
    
    if 'model_config' in config:
        print(f"Model architecture: {config['model_config'].get('architecture', 'Not specified')}")
        if 'specaug' in config['model_config']:
            specaug = config['model_config']['specaug']
            print(f"SpecAugment: ✅ Enabled")
            if isinstance(specaug, dict):
                for key, value in specaug.items():
                    print(f"   {key}: {value}")
    
    # Class imbalance analysis
    print("\n⚖️  CLASS IMBALANCE ANALYSIS:")
    print("-" * 50)
    
    total_bonafide = sum(stats.get('bonafide_count', 0) for stats in all_stats.values())
    total_spoof = sum(stats.get('spoof_count', 0) for stats in all_stats.values())
    total_samples = total_bonafide + total_spoof
    
    if total_samples > 0:
        print(f"Overall dataset:")
        print(f"   Bonafide: {total_bonafide} ({total_bonafide/total_samples*100:.1f}%)")
        print(f"   Spoof: {total_spoof} ({total_spoof/total_samples*100:.1f}%)")
        
        if total_bonafide > 0 and total_spoof > 0:
            imbalance_ratio = max(total_bonafide, total_spoof) / min(total_bonafide, total_spoof)
            print(f"   Class imbalance ratio: {imbalance_ratio:.2f}:1")
            
            if imbalance_ratio > 2:
                print(f"   ⚠️  Significant class imbalance detected!")
        else:
            print(f"   ⚠️  One class has zero samples!")
    
    # Training set specific analysis
    if 'train' in all_stats:
        train_stats = all_stats['train']
        print(f"\n📚 TRAINING SET DETAILS:")
        print("-" * 50)
        
        if 'utterance_lengths' in train_stats and len(train_stats['utterance_lengths']) > 0:
            lengths = train_stats['utterance_lengths']
            print(f"Utterance length statistics:")
            print(f"   Min: {min(lengths):.2f}s")
            print(f"   Max: {max(lengths):.2f}s")
            print(f"   Mean: {np.mean(lengths):.2f}s")
            print(f"   Median: {np.median(lengths):.2f}s")
            
            # Check for very short/long utterances
            short_threshold = 1.0  # seconds
            long_threshold = 10.0  # seconds
            
            short_utts = [l for l in lengths if l < short_threshold]
            long_utts = [l for l in lengths if l > long_threshold]
            
            if short_utts:
                print(f"   ⚠️  {len(short_utts)} utterances shorter than {short_threshold}s")
            if long_utts:
                print(f"   ⚠️  {len(long_utts)} utterances longer than {long_threshold}s")
    
    # Generate visualizations if matplotlib is available
    try:
        print("\n📈 GENERATING VISUALIZATIONS...")
        generate_visualizations(all_stats, track)
        print("✅ Visualizations saved to 'database_analysis' folder")
    except Exception as e:
        print(f"⚠️  Could not generate visualizations: {e}")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    print("-" * 50)
    
    recommendations = []
    
    # Check class balance
    if 'train' in all_stats:
        train_bonafide = all_stats['train'].get('bonafide_count', 0)
        train_spoof = all_stats['train'].get('spoof_count', 0)
        if train_spoof > 0:
            train_imbalance = train_bonafide / train_spoof
            if train_imbalance > 2 or train_imbalance < 0.5:
                recommendations.append("Consider using class weights in loss function due to class imbalance")
    
    # Check dataset sizes
    if 'train' in all_stats and 'dev' in all_stats:
        train_size = all_stats['train'].get('total_samples', 0)
        dev_size = all_stats['dev'].get('total_samples', 0)
        if train_size > 0 and dev_size / train_size < 0.1:
            recommendations.append(f"Development set is small ({dev_size/train_size*100:.1f}% of training). Consider using cross-validation.")
    
    # Check augmentation
    if not freq_aug:
        recommendations.append("Consider enabling frequency augmentation for better generalization")
    
    # Check for missing files
    total_missing = sum(stats.get('missing_files', 0) for stats in all_stats.values())
    if total_missing > 0:
        recommendations.append(f"Found {total_missing} missing audio files. Verify dataset integrity.")
    
    if not recommendations:
        recommendations.append("Dataset looks well-balanced and configured for training.")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


def analyze_protocol_file(protocol_path: Path, split_name: str, track: str) -> Dict:
    """Analyze a protocol file and extract statistics."""
    stats = {
        'total_samples': 0,
        'bonafide_count': 0,
        'spoof_count': 0,
        'speakers': [],
        'utterances': [],
        'spoof_attacks': Counter(),
        'bonafide_percentage': 0,
        'spoof_percentage': 0,
        'missing_files': 0
    }
    
    try:
        with open(protocol_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"⚠️  Protocol file not found: {protocol_path}")
        return stats
    
    stats['total_samples'] = len(lines)
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            # Format: speaker utterance - - label
            speaker_id = parts[0]
            utterance_id = parts[1]
            label = parts[4]
            
            stats['speakers'].append(speaker_id)
            stats['utterances'].append(utterance_id)
            
            if label == 'bonafide':
                stats['bonafide_count'] += 1
            else:
                stats['spoof_count'] += 1
                # Extract attack type (for LA track)
                if track == 'LA' and '-' in utterance_id:
                    try:
                        attack_type = utterance_id.split('-')[1]
                        stats['spoof_attacks'][attack_type] += 1
                    except:
                        stats['spoof_attacks']['unknown'] += 1
                elif track == 'PA':
                    # PA track attack types
                    if 'replay' in label.lower():
                        stats['spoof_attacks']['replay'] += 1
        elif len(parts) == 3:
            # Alternative format for eval sets
            speaker_id = parts[0]
            utterance_id = parts[1]
            label = parts[2]
            
            stats['speakers'].append(speaker_id)
            stats['utterances'].append(utterance_id)
            
            if label == 'bonafide':
                stats['bonafide_count'] += 1
            elif label == 'spoof':
                stats['spoof_count'] += 1
    
    if stats['total_samples'] > 0:
        stats['bonafide_percentage'] = (stats['bonafide_count'] / stats['total_samples']) * 100
        stats['spoof_percentage'] = (stats['spoof_count'] / stats['total_samples']) * 100
    
    stats['speakers'] = list(set(stats['speakers']))
    
    return stats


def analyze_audio_files(audio_dir: Path, utterance_ids: List[str]) -> Dict:
    """Analyze audio files and extract duration information."""
    stats = {
        'audio_duration_stats': None,
        'sample_rates': set(),
        'utterance_lengths': [],
        'missing_files': 0
    }
    
    # Check if librosa is available
    try:
        import librosa
        librosa_available = True
    except ImportError:
        print("   ⚠️  librosa not available, skipping audio duration analysis")
        return stats
    
    durations = []
    
    # Check first N files to get statistics
    max_files_to_check = min(100, len(utterance_ids))
    sample_utterances = utterance_ids[:max_files_to_check]
    
    print(f"   Checking {len(sample_utterances)} audio files for duration...")
    
    for utt_id in tqdm(sample_utterances, desc=f"   Checking audio", leave=False):
        # Try different file extensions
        audio_file = None
        for ext in ['.flac', '.wav']:
            potential_file = audio_dir / f"{utt_id}{ext}"
            if potential_file.exists():
                audio_file = potential_file
                break
        
        if audio_file and audio_file.exists():
            try:
                # Load audio to get duration (only load first 30 seconds to save time)
                y, sr = librosa.load(audio_file, sr=None, mono=True, duration=30)
                duration = librosa.get_duration(y=y, sr=sr)
                durations.append(duration)
                stats['sample_rates'].add(sr)
                stats['utterance_lengths'].append(duration)
            except Exception as e:
                print(f"   ⚠️  Could not load {audio_file}: {e}")
        else:
            stats['missing_files'] += 1
    
    if durations:
        stats['audio_duration_stats'] = {
            'min': min(durations),
            'max': max(durations),
            'mean': np.mean(durations),
            'std': np.std(durations),
            'median': np.median(durations)
        }
    
    return stats


def generate_visualizations(all_stats: Dict, track: str) -> None:
    """Generate visualization plots for dataset analysis."""
    output_dir = Path("database_analysis")
    output_dir.mkdir(exist_ok=True)
    
    # 1. Class distribution across splits
    splits_with_data = [name for name, stats in all_stats.items() 
                       if 'bonafide_count' in stats and 'spoof_count' in stats]
    
    if len(splits_with_data) > 0:
        fig, axes = plt.subplots(1, min(3, len(splits_with_data)), figsize=(15, 5))
        if len(splits_with_data) == 1:
            axes = [axes]
        
        for idx, split_name in enumerate(splits_with_data[:3]):
            stats = all_stats[split_name]
            ax = axes[idx] if len(splits_with_data) > 1 else axes
            labels = ['Bonafide', 'Spoof']
            counts = [stats['bonafide_count'], stats['spoof_count']]
            
            bars = ax.bar(labels, counts, color=['green', 'red'])
            ax.set_title(f'{split_name.upper()} Set Class Distribution')
            ax.set_ylabel('Number of Samples')
            ax.set_xlabel('Class')
            
            # Add count labels on bars
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{count}\n({count/sum(counts)*100:.1f}%)',
                       ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{track}_class_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # 2. Attack type distribution (for LA track)
    if track == 'LA' and 'train' in all_stats:
        train_stats = all_stats['train']
        if 'spoof_attacks' in train_stats and train_stats['spoof_attacks']:
            attack_types = list(train_stats['spoof_attacks'].keys())
            attack_counts = list(train_stats['spoof_attacks'].values())
            
            plt.figure(figsize=(12, 6))
            bars = plt.barh(attack_types, attack_counts, color='steelblue')
            plt.xlabel('Number of Samples')
            plt.title(f'{track} Track - Spoof Attack Type Distribution (Train Set)')
            plt.gca().invert_yaxis()  # Highest count on top
            
            # Add count labels
            for bar, count in zip(bars, attack_counts):
                plt.text(count + max(attack_counts)*0.01, bar.get_y() + bar.get_height()/2,
                        str(count), va='center')
            
            plt.tight_layout()
            plt.savefig(output_dir / f'{track}_attack_distribution.png', dpi=150, bbox_inches='tight')
            plt.close()
    
    # 3. Duration distribution if available
    if 'train' in all_stats and 'utterance_lengths' in all_stats['train']:
        durations = all_stats['train']['utterance_lengths']
        if durations:
            plt.figure(figsize=(10, 6))
            plt.hist(durations, bins=50, alpha=0.7, color='purple', edgecolor='black')
            plt.xlabel('Duration (seconds)')
            plt.ylabel('Frequency')
            plt.title(f'{track} Track - Utterance Duration Distribution (Train Set)')
            
            # Add statistics text
            stats_text = (f'Mean: {np.mean(durations):.2f}s\n'
                         f'Std: {np.std(durations):.2f}s\n'
                         f'Min: {min(durations):.2f}s\n'
                         f'Max: {max(durations):.2f}s\n'
                         f'N: {len(durations)} samples')
            plt.text(0.95, 0.95, stats_text,
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            plt.savefig(output_dir / f'{track}_duration_distribution.png', dpi=150, bbox_inches='tight')
            plt.close()
    
    # 4. Save detailed statistics to CSV
    csv_data = []
    for split_name, stats in all_stats.items():
        row = {
            'split': split_name,
            'total_samples': stats.get('total_samples', 0),
            'bonafide_count': stats.get('bonafide_count', 0),
            'spoof_count': stats.get('spoof_count', 0),
            'bonafide_percentage': stats.get('bonafide_percentage', 0),
            'spoof_percentage': stats.get('spoof_percentage', 0),
            'unique_speakers': len(stats.get('speakers', [])),
            'missing_files': stats.get('missing_files', 0)
        }
        
        # Add duration stats if available
        if 'audio_duration_stats' in stats and stats['audio_duration_stats']:
            dur_stats = stats['audio_duration_stats']
            row.update({
                'duration_min': dur_stats.get('min', 0),
                'duration_max': dur_stats.get('max', 0),
                'duration_mean': dur_stats.get('mean', 0),
                'duration_std': dur_stats.get('std', 0),
                'duration_median': dur_stats.get('median', 0)
            })
        
        csv_data.append(row)
    
    if csv_data:
        df = pd.DataFrame(csv_data)
        df.to_csv(output_dir / f'{track}_dataset_statistics.csv', index=False)
    
    # Create a summary text file
    with open(output_dir / f'{track}_analysis_summary.txt', 'w') as f:
        f.write(f"ASVspoof2019 {track} Track Database Analysis\n")
        f.write("=" * 50 + "\n\n")
        
        for split_name, stats in all_stats.items():
            f.write(f"{split_name.upper()} SET:\n")
            f.write(f"  Total samples: {stats.get('total_samples', 0)}\n")
            f.write(f"  Bonafide: {stats.get('bonafide_count', 0)} ({stats.get('bonafide_percentage', 0):.1f}%)\n")
            f.write(f"  Spoof: {stats.get('spoof_count', 0)} ({stats.get('spoof_percentage', 0):.1f}%)\n")
            f.write(f"  Unique speakers: {len(stats.get('speakers', []))}\n")
            f.write(f"  Missing audio files: {stats.get('missing_files', 0)}\n")
            
            if 'audio_duration_stats' in stats and stats['audio_duration_stats']:
                dur_stats = stats['audio_duration_stats']
                f.write("  Duration statistics:\n")
                f.write(f"    Min: {dur_stats.get('min', 0):.2f}s\n")
                f.write(f"    Max: {dur_stats.get('max', 0):.2f}s\n")
                f.write(f"    Mean: {dur_stats.get('mean', 0):.2f}s\n")
                f.write(f"    Std: {dur_stats.get('std', 0):.2f}s\n")
            
            f.write("\n")


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
    
    # ========== DATABASE ANALYSIS FLAG ==========
    if args.analyze_db:
        analyze_database(config, database_path, track)
        if not args.train_after_analysis:
            print("\nDatabase analysis completed. Exiting.")
            sys.exit(0)
        else:
            print("\nProceeding with training after analysis...\n")
    # ===========================================
    
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

    # Training loop
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print(f"Epochs: {config['num_epochs']}")
    print("="*60 + "\n")
    
    for epoch in range(config["num_epochs"]):
        epoch_start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config['num_epochs']}")
        print(f"{'='*60}")
        
        # ========== TRAIN PHASE ==========
        print("\n🚀 Training...")
        train_start_time = time.time()
        running_loss = train_epoch(trn_loader, model, optimizer, device,
                                   scheduler, config, epoch)
        train_time = time.time() - train_start_time
        
        print(f"   ✓ Training completed in {train_time:.1f}s")
        print(f"   📊 Training Loss: {running_loss:.4f}")
        
        # ========== VALIDATION PHASE ==========
        print("\n📊 Validating...")
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
        
        print(f"   ✓ Validation completed in {val_time:.1f}s")
        print(f"\n📈 Epoch {epoch+1} Summary:")
        print(f"   EER:        {dev_eer:.3f}%")
        print(f"   t-DCF:      {dev_tdcf:.5f}")
        print(f"   Acc@EER:    {dev_acc_eer:.2f}%")
        print(f"   Acc@t-DCF:  {dev_acc_tdcf:.2f}%")
        print(f"   Max Acc:    {dev_max_acc:.2f}%")
        print(f"   Total Time: {epoch_time:.1f}s")
        
        writer.add_scalar("loss", running_loss, epoch)
        writer.add_scalar("dev_eer", dev_eer, epoch)
        writer.add_scalar("dev_tdcf", dev_tdcf, epoch)
        writer.add_scalar("dev_acc_eer", dev_acc_eer, epoch)
        writer.add_scalar("dev_acc_tdcf", dev_acc_tdcf, epoch)
        writer.add_scalar("dev_max_acc", dev_max_acc, epoch)

        best_dev_tdcf = min(dev_tdcf, best_dev_tdcf)
        if best_dev_eer >= dev_eer:
            print(f"\n🎯 NEW BEST MODEL! EER improved from {best_dev_eer:.3f}% to {dev_eer:.3f}%")
            best_dev_eer = dev_eer
            torch.save(model.state_dict(),
                       model_save_path / "epoch_{}_{:03.3f}.pth".format(epoch, dev_eer))

            # Do evaluation whenever best model is renewed
            if str_to_bool(config["eval_all_best"]):
                print("   🔍 Evaluating on eval set...")
                eval_start_time = time.time()
                produce_evaluation_file(eval_loader, model, device,
                                        eval_score_path, eval_trial_path)
                eval_eer, eval_tdcf, eval_acc_eer, eval_acc_tdcf, eval_max_acc = calculate_tDCF_EER(
                    cm_scores_file=eval_score_path,
                    asv_score_file=database_path / config["asv_score_path"],
                    output_file=metric_path /
                    "t-DCF_EER_{:03d}epo.txt".format(epoch))

                eval_time = time.time() - eval_start_time
                
                log_text = f"   📊 Eval Results: EER={eval_eer:.3f}%, t-DCF={eval_tdcf:.5f}"
                if eval_eer < best_eval_eer:
                    log_text += f" (NEW BEST EER)"
                    best_eval_eer = eval_eer
                if eval_tdcf < best_eval_tdcf:
                    log_text += f" (NEW BEST t-DCF)"
                    best_eval_tdcf = eval_tdcf
                    torch.save(model.state_dict(),
                               model_save_path / "best.pth")
                
                log_text += f"\n   🎯 Acc@EER: {eval_acc_eer:.2f}%, Acc@min-tDCF: {eval_acc_tdcf:.2f}%, Max Acc: {eval_max_acc:.2f}%"
                log_text += f"\n   ⏱️  Eval Time: {eval_time:.1f}s"
                
                print(log_text)
                f_log.write(f"epoch{epoch+1:03d}: {log_text}\n")

            print(f"   💾 Saving epoch {epoch+1} for SWA")
            optimizer_swa.update_swa()
            n_swa_update += 1
        
        writer.add_scalar("best_dev_eer", best_dev_eer, epoch)
        writer.add_scalar("best_dev_tdcf", best_dev_tdcf, epoch)
        
        # Clear memory if needed
        if device == "cuda":
            torch.cuda.empty_cache()
    
    print("\n" + "="*60)
    print("🎯 FINAL EVALUATION")
    print("="*60)
    
    if n_swa_update > 0:
        print("🔁 Applying SWA (Stochastic Weight Averaging)...")
        optimizer_swa.swap_swa_sgd()
        optimizer_swa.bn_update(trn_loader, model, device=device)
    
    print("📊 Final evaluation on eval set...")
    eval_start_time = time.time()
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
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE!")
    print("="*60)
    print(f"  Best EER:                    {best_eval_eer:.3f}%")
    print(f"  Best t-DCF:                  {best_eval_tdcf:.5f}")
    print(f"  Final Accuracy @ EER:        {eval_acc_eer:.2f}%")
    print(f"  Final Accuracy @ min-tDCF:   {eval_acc_tdcf:.2f}%")
    print(f"  Final Maximum Accuracy:      {eval_max_acc:.2f}%")
    print(f"  Total Training Time:         {total_time:.1f}s")
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
    
    # Add progress bar for evaluation
    pbar = tqdm(data_loader, desc="Evaluating", leave=False)
    
    with torch.no_grad():
        for batch_x, utt_id in pbar:
            batch_x = batch_x.to(device, non_blocking=True)
            _, batch_out = model(batch_x)
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
            fname_list.extend(utt_id)
            score_list.extend(batch_score.tolist())
            
            # Update progress
            pbar.set_postfix({'samples': len(fname_list)})

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
    """Train the model for one epoch with progress bar"""
    running_loss = 0
    num_total = 0.0
    model.train()

    # Set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    
    # Training progress bar
    train_pbar = tqdm(trn_loader, desc="Training", leave=False)
    
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
        
        # Update progress bar
        current_loss = running_loss / num_total
        train_pbar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'lr': f'{optim.param_groups[0]["lr"]:.6f}' if optim.param_groups[0].get("lr") else 'N/A'
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
    
    # ========== NEW FLAG: DATABASE ANALYSIS ==========
    parser.add_argument("--analyze_db",
                        action="store_true",
                        help="comprehensive database analysis before training")
    parser.add_argument("--train_after_analysis",
                        action="store_true",
                        help="continue training after database analysis")
    # =================================================
    
    # Parse arguments
    args = parser.parse_args()
    
    # Call main function with parsed arguments
    main(args)