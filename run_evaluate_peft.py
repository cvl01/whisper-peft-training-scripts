import gc
import os
from convert.merge_peft_model import merge_peft_model
from datasets import Dataset
from evaluation.evaluation import evaluate_model, evaluate_peft_model
from peft.config import PeftConfig
from peft.peft_model import PeftModel
from train.prepare_dataset import map_prepare_dataset
from transformers import (
    BitsAndBytesConfig,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizerFast,
)

from train.load_datasets import load_datasets

# VARIABLES
model_name_or_path = "openai/whisper-large-v3"
language_abbr = "nl"
task = "transcribe"

# WHISPER
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)
tokenizer = WhisperTokenizerFast.from_pretrained(
    model_name_or_path, language=language_abbr, task=task
)
processor = WhisperProcessor.from_pretrained(
    model_name_or_path, language=language_abbr, task=task
)

# DATASETS
dataset = load_datasets()
# test_dataset: Dataset = map_prepare_dataset(dataset['test'], feature_extractor=feature_extractor, tokenizer=tokenizer) # type: ignore


models = [
    "models/large-v3-big-dataset/checkpoint-306",
    "models/large-v3-big-dataset/checkpoint-612",
    "models/large-v3-big-dataset/checkpoint-918",
]
model_metrics = {}
for model_id in models:
    model_id = os.path.abspath(model_id)
    peft_config = PeftConfig.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(
        peft_config.base_model_name_or_path,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="auto",
    )
    peft_model = PeftModel.from_pretrained(model, model_id)

    merged_model = merge_peft_model(peft_model, model_id + '-merged/')
    tokenizer.save_pretrained(merged_model)
    processor.save_pretrained(merged_model)

    del model, peft_model
    gc.collect()

    print(merged_model)
    print(model_id)

    tuned_metrics = evaluate_model(merged_model, dataset["test"])

    print(tuned_metrics)
    model_metrics[model_id] = tuned_metrics

print(model_metrics)

# open_aimetrics = evaluate_model("openai/whisper-large-v3", dataset['test'])
# print(open_aimetrics)
