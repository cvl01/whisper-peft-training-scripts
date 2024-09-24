from datasets import (
    Audio,
    Dataset,
    DatasetDict,
    interleave_datasets,
    load_dataset,
)


def load_datasets() -> DatasetDict:

    tts_dataset = load_dataset("csv", data_files="datasets/tts-dataset-4000-v3.csv")
    tts_dataset = tts_dataset.cast_column("audio", Audio(sampling_rate=16000))

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

    cv_dataset = DatasetDict()

    cv_dataset["train"] = load_dataset(
        "mozilla-foundation/common_voice_17_0", "nl", split="train+validation"
    )
    cv_dataset["test"] = load_dataset(
        "mozilla-foundation/common_voice_17_0", "nl", split="test"
    )

    cv_dataset = cv_dataset.remove_columns(
        [
            "accent",
            "age",
            "client_id",
            "down_votes",
            "gender",
            "locale",
            "path",
            "segment",
            "up_votes",
            "variant",
        ]
    )
    cv_dataset = cv_dataset.cast_column("audio", Audio(sampling_rate=16000))

    big_dataset = DatasetDict()
    big_dataset["train"] = interleave_datasets(
        [interview_dataset["train"], tts_dataset["train"], cv_dataset["train"]],
        probabilities=[0.1, 0.35, 0.55],
        seed=42,
    ).shuffle(seed=42)
    big_dataset["test"] = interleave_datasets(
        [interview_dataset["test"], cv_dataset["test"]],
        probabilities=[0.15, 0.85],
        seed=42,
    ).shuffle(seed=42)

    return big_dataset
