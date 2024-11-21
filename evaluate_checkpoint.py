import json
import os
import evaluate
from peft.config import PeftConfig
from peft.peft_model import PeftModel
from evaluation.dutch_normalizer import DutchTextNormalizer
from evaluation.evaluation import compute_wer
import torch
from train.load_datasets import load_datasets
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from datasets import Audio, load_dataset
from jiwer import wer
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

whisper_norm = DutchTextNormalizer()


def load_and_merge_model(peft_model_path, device):
    # Load the base model
    peft_config = PeftConfig.from_pretrained(peft_model_path)
    base_model = WhisperForConditionalGeneration.from_pretrained(peft_config.base_model_name_or_path)

    # Load the PEFT model
    peft_model = PeftModel.from_pretrained(base_model, peft_model_path)

    # Merge the PEFT model with the base model
    merged_model = peft_model.merge_and_unload()

    # Move model to the specified GPU
    merged_model.to(device)

    return merged_model



def evaluate_model_on_gpu(peft_model_id, processor, dataset, device):
    print(f"Evaluating {peft_model_id} on {device}")
    merged_model = load_and_merge_model(peft_model_id, device)
    return compute_wer(merged_model, processor, dataset, device)


def main():
    # Model and dataset paths
    base_model_name = "openai/whisper-large-v3"
    # Load the processor
    processor = WhisperProcessor.from_pretrained(base_model_name)

    # Load the test dataset
    interview_dataset = load_dataset(
        "csv",
        data_files={
            "train": "datasets/interview-v3-filtered_segments_train.csv",
            "test": "datasets/interview-v3-filtered_segments_test.csv",
        },
    )
    interview_dataset = interview_dataset.remove_columns(
        [
            "transcription",
            "wer",
        ]
    )
    interview_dataset = interview_dataset.cast_column(
        "audio", Audio(sampling_rate=16000)
    )
    test_dataset = interview_dataset['test'] # type: ignore

    models = [
        "models/large-v3-big-dataset/checkpoint-306",
        "models/large-v3-big-dataset/checkpoint-612",
        "models/large-v3-big-dataset/checkpoint-918",
        "models/large-v3-big-dataset",
    ]

    # Determine available GPUs
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    model_metrics = {}

    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = []
        for i, peft_model_id in enumerate(models):
            # Assign a GPU based on index
            device = devices[i % len(devices)]
            # Schedule evaluation on that GPU
            futures.append(
                executor.submit(evaluate_model_on_gpu, peft_model_id, processor, test_dataset, device)
            )

        # Collect results
        for i, future in enumerate(futures):
            model_metrics[models[i]] = future.result()
            print(models[i])
            print(model_metrics)

    # Save the metrics to a file
    with open(os.path.join('models', 'metrics.json'), 'w') as fp:
        json.dump(model_metrics, fp)
    print(model_metrics)


if __name__ == "__main__":
    main()
