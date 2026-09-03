"""
Guess Analysis

Analyzes accuracy and performance of speculative execution by comparing guess
predictions against actual moves and calculating time savings. Use --confidence
for confidence-aware analysis (filters guesses by confidence score and uses
steps_info*_confidence_prediction.json).
"""

import json
import csv
import os
import sys
import yaml
import glob
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional


CONFIDENCE_THRESHOLD = 50.0


def load_step_info(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r") as f:
        return json.load(f)


def analyze_with_truncation(step_info: Dict[str, Any], target_steps: int, with_confidence: bool = False) -> Dict[str, Any]:
    steps: List[Dict[str, Any]] = list(step_info.values())
    target_steps = min(target_steps, len(steps))
    if target_steps <= 0:
        return {
            "num_predictions": 0,
            "time_regular": 0.0,
            "time_speculate": 0.0,
            "time_saved_percentage": 0.0,
            "tokens_regular": 0.0,
            "tokens_speculate": 0.0,
            "tokens_wasted_percentage": 0.0,
            "match_number": 0,
            "accuracy": 0.0,
            "speculative_window_accuracy": 0.0,
        }
    time_checker_regular = 0
    token_checker_regular = 0
    token_checker_speculate = 0
    time_checker_speculate = 0
    prev_match = False
    match_counter = 0

    if with_confidence:
        guessed_and_gt = steps[0].get("guessed_moves + ground_truth_move", steps[0].get("guessed_moves", []))
        if isinstance(guessed_and_gt, list) and len(guessed_and_gt) > 0:
            num_predictions = len(guessed_and_gt) - 1  # exclude ground truth
        else:
            num_predictions = len(steps[0].get("guessed_moves", []))
    else:
        num_predictions = len(steps[0].get("guessed_moves", []))

    speculative_window_match_counter = 0
    num_speculative_window = 0
    result_num_predictions = num_predictions

    for step in range(target_steps - 1):
        time_checker_regular += steps[step]["time_taken_current_agent"]
        token_checker_regular += steps[step]["total_tokens_current_agent"]
        token_checker_speculate += steps[step]["total_tokens_current_agent"]

        if with_confidence:
            guessed_and_gt = steps[step].get("guessed_moves + ground_truth_move", steps[step].get("guessed_moves", []))
            confidence_scores = steps[step].get("confidence_scores", [])
            if len(guessed_and_gt) > 1 and len(confidence_scores) >= 1:
                guessed_moves = guessed_and_gt[:-1]
                n = min(len(guessed_moves), len(confidence_scores))
                scores = (confidence_scores[:-1] if len(confidence_scores) > n else confidence_scores)[:n]
                filtered_guessed_moves = [guessed_moves[i] for i in range(n) if scores[i] > CONFIDENCE_THRESHOLD]
            else:
                filtered_guessed_moves = steps[step].get("guessed_moves", [])
            moves_to_check = filtered_guessed_moves
            num_predictions_step = len(filtered_guessed_moves)
            result_num_predictions = num_predictions_step
        else:
            moves_to_check = steps[step]["guessed_moves"]
            num_predictions_step = num_predictions

        if steps[step]["current_move"] in moves_to_check and prev_match is False:
            prev_match = True
            match_counter += 1
            speculative_window_match_counter += 1
            num_speculative_window += 1

            spec_time = steps[step]["guess_prediction_time"] + steps[step + 1]["time_taken_current_agent"]
            spec_tokens = steps[step]["guess_total_tokens"] + steps[step + 1]["total_tokens_current_agent"]

            if spec_time < steps[step]["time_taken_current_agent"]:
                time_checker_speculate += steps[step]["time_taken_current_agent"]
                token_checker_speculate += num_predictions_step * spec_tokens
            else:
                time_checker_speculate += spec_time
                token_checker_speculate += (num_predictions_step - 1) * steps[step]["total_tokens_current_agent"] + spec_tokens

        elif prev_match:
            prev_match = False
            if steps[step]["current_move"] in moves_to_check:
                match_counter += 1
        else:
            num_speculative_window += 1
            time_checker_speculate += steps[step]["time_taken_current_agent"]
            token_checker_speculate += num_predictions_step * steps[step]["total_tokens_current_agent"]

    time_checker_regular += steps[target_steps - 1]["time_taken_current_agent"]
    token_checker_regular += steps[target_steps - 1]["total_tokens_current_agent"]
    token_checker_speculate += steps[target_steps - 1]["total_tokens_current_agent"]
    if not prev_match:
        time_checker_speculate += steps[target_steps - 1]["time_taken_current_agent"]
        token_checker_speculate += steps[target_steps - 1]["total_tokens_current_agent"]

    time_saved_percentage = ((time_checker_regular - time_checker_speculate) / time_checker_regular * 100) if time_checker_regular > 0 else 0
    tokens_wasted_percentage = ((token_checker_regular - token_checker_speculate) / token_checker_regular * 100) if token_checker_regular > 0 else 0
    accuracy = match_counter / target_steps
    speculative_window_accuracy = speculative_window_match_counter / num_speculative_window if num_speculative_window else 0

    return {
        "num_predictions": result_num_predictions,
        "time_regular": time_checker_regular,
        "time_speculate": time_checker_speculate,
        "time_saved_percentage": time_saved_percentage,
        "tokens_regular": token_checker_regular,
        "tokens_speculate": token_checker_speculate,
        "tokens_wasted_percentage": tokens_wasted_percentage,
        "match_number": match_counter,
        "accuracy": accuracy,
        "speculative_window_accuracy": speculative_window_accuracy,
    }


def run_single_analysis(file_path: str, target_steps: int, with_confidence: bool = False) -> tuple:
    step_info: Dict[str, Any] = load_step_info(file_path)
    result = analyze_with_truncation(step_info, target_steps, with_confidence=with_confidence)
    return [result], len(step_info)


def print_analysis_results(results: list, num_steps: int) -> None:
    for result in results:
        print(f"\nAnalysis with {result['num_predictions']} predictions:")
        print(f"Time taken regular: {result['time_regular']:.4f}")
        print(f"Time taken speculate: {result['time_speculate']:.4f}")
        print(f"Match number: {result['match_number']}")
        print(f"Accuracy: {result['accuracy']}")
        print(f"Speculative window accuracy: {result['speculative_window_accuracy']}")


def log_results_to_csv(
    all_results: list,
    base_path: str,
    output_file: str = "analysis_results.csv",
) -> None:
    fieldnames = [
        "timestamp", "target_steps", "guess_file", "num_predictions",
        "time_regular", "time_speculate", "time_saved_percentage",
        "tokens_regular", "tokens_speculate", "tokens_wasted_percentage",
        "match_number", "accuracy", "speculative_window_accuracy",
    ]
    path = os.path.join(base_path, output_file)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in all_results:
            writer.writerow(entry)
    print(f"Results logged to {path}")


def run_batch_analysis(
    file_paths: list,
    target_steps_list: list,
    base_path: str,
    with_confidence: bool = False,
) -> None:
    all_results = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for file_path in file_paths:
        guess_number = file_path.split("_")[-1].split(".")[0]
        print(f"Analyzing file: {file_path} with guess number: {guess_number}")
        for target_steps in target_steps_list:
            results, num_steps = run_single_analysis(file_path, target_steps, with_confidence=with_confidence)
            print(f"\n=== Analysis for {file_path} with {target_steps} steps ===")
            print_analysis_results(results, num_steps)
            for result in results:
                all_results.append({
                    "timestamp": timestamp,
                    "target_steps": target_steps,
                    "guess_file": f"guess_{guess_number}",
                    **result,
                })
    output_file = "analysis_results_with_confidence.csv" if with_confidence else "analysis_results.csv"
    log_results_to_csv(all_results, base_path, output_file=output_file)


def run_analysis_for_path(
    base_path: str,
    config: dict,
    target_steps_list: Optional[list[int]] = None,
    with_confidence: bool = False,
) -> None:
    if target_steps_list is None:
        target_steps_list = config.get("analysis", {}).get("target_steps", [20, 30, 40, 50])

    if with_confidence:
        step_info_logs = glob.glob(f"{base_path}/steps_info*_confidence_prediction.json")
    else:
        step_info_logs = glob.glob(f"{base_path}/steps_info*_guess_*.json")

    if not step_info_logs:
        print(f"No step info logs found in {base_path}. Skipping analysis.")
        return
    print(f"Found {len(step_info_logs)} step info logs in {base_path}")
    run_batch_analysis(step_info_logs, target_steps_list, base_path, with_confidence=with_confidence)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Analyze guess accuracy and time savings. Use --confidence for confidence-aware analysis.",
    )
    p.add_argument("--config", default="config.yml", help="Path to config YAML")
    p.add_argument("--base-dir", default=None, help="Base directory with per-run trajectory subdirs (default: from config paths.trajectories)")
    p.add_argument("--confidence", action="store_true", help="Use confidence-aware analysis (steps_info*_confidence_prediction.json, filter by confidence > 50)")
    args = p.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    if args.base_dir is not None:
        config.setdefault("paths", {})["trajectories"] = args.base_dir
    base_dir = config.get("paths", {}).get("trajectories")
    if not base_dir:
        print("Error: base-dir not provided and config paths.trajectories not set.")
        sys.exit(1)

    title = "Guess Analysis (confidence-aware)" if args.confidence else "Guess Analysis"
    print(f"=== {title} ===")
    print(f"Analyzing trajectories in: {base_dir}\n")
    if not os.path.exists(base_dir):
        print(f"Error: Directory not found: {base_dir}")
        sys.exit(1)

    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        if os.path.isdir(folder_path):
            print(f"\nProcessing folder: {folder_path}")
            run_analysis_for_path(folder_path, config=config, with_confidence=args.confidence)
    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()
