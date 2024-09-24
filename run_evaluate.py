
from evaluation.evaluation import evaluate_model
from transformers import (
    WhisperFeatureExtractor,
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
# test_dataset: Dataset = map_prepare_dataset(dataset['test'], feature_extractor=feature_extractor, tokenizer=tokenizer)


models = [
    'models/large-v3-big-dataset/checkpoint-306',
    'models/large-v3-big-dataset/checkpoint-612',
    'models/large-v3-big-dataset/checkpoint-918',
]

tuned_metrics = evaluate_model("models/large-v3-big-dataset", dataset['test'])
print(tuned_metrics)



open_aimetrics = evaluate_model("openai/whisper-large-v3", dataset['test'])

print(tuned_metrics)
print("-----")
print(open_aimetrics)
