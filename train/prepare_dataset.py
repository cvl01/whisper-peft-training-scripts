
from datasets import Dataset, DatasetDict


def map_prepare_dataset(dataset: Dataset | DatasetDict, feature_extractor, tokenizer):
    def prepare_dataset(batch):
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        del batch["audio"]

        # encode target text to label ids
        batch["labels"] = tokenizer(batch["sentence"]).input_ids
        return batch

    return dataset.map(prepare_dataset, num_proc=4)
