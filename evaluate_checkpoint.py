import json
import os
import evaluate
from evaluation.dutch_normalizer import DutchTextNormalizer
import torch
from train.load_datasets import load_datasets
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel, PeftConfig
from datasets import load_dataset
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


def compute_wer(model, processor, dataset, device):
    model.eval()
    model.to(device)

    all_references = []
    all_hypotheses = []

    for batch in tqdm(dataset):

        input_features = processor(
            batch["audio"]["array"], sampling_rate=16000, return_tensors="pt"
        ).input_features
        input_features = input_features.to(device)

        with torch.no_grad():
            predicted_ids = model.generate(
                input_features, language="nl", task="transcribe"
            )

        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

        all_references.extend([batch["sentence"]])
        all_hypotheses.extend(transcription)

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    wer = wer_metric.compute(references=all_references, predictions=all_hypotheses)
    wer = round(100 * wer, 2)  # type: ignore
    cer = cer_metric.compute(references=all_references, predictions=all_hypotheses)
    cer = round(100 * cer, 2)  # type: ignore

    norm_predictions = [whisper_norm(pred) for pred in all_hypotheses]
    norm_references = [whisper_norm(ref) for ref in all_references]
    norm_wer = wer_metric.compute(
        references=norm_references, predictions=norm_predictions
    )
    norm_wer = round(100 * norm_wer, 2)  # type: ignore
    norm_cer = cer_metric.compute(
        references=norm_references, predictions=norm_predictions
    )
    norm_cer = round(100 * norm_cer, 2)  # type: ignore

    print("WER : ", wer)
    print("CER : ", cer)
    print("\nNORMALIZED WER : ", norm_wer)
    print("NORMALIZED CER : ", norm_cer)

    return {
        "wer": wer,
        "norm_wer": norm_wer,
        "cer": cer,
        "norm_cer": norm_cer,
    }


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
    test_dataset = load_datasets()
    test_dataset = test_dataset["test"]

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
