import gc
import evaluate
from jiwer import process
from librosa import feature
import torch
import numpy as np
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from train.data_collator import DataCollatorSpeechSeq2SeqWithPadding
from transformers import BitsAndBytesConfig, WhisperFeatureExtractor, WhisperProcessor, WhisperTokenizerFast, pipeline
from peft import PeftModel, PeftConfig
from transformers import WhisperForConditionalGeneration, Seq2SeqTrainer
from datasets import Dataset, DatasetDict
from evaluation.dutch_normalizer import DutchTextNormalizer

whisper_norm = DutchTextNormalizer()
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

def evaluate_peft_model(model_id:str, test_dataset: Dataset, processor, tokenizer):

    peft_config = PeftConfig.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(
        peft_config.base_model_name_or_path, quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map="auto"
    )
    model = PeftModel.from_pretrained(model, model_id)
    
    predictions = []
    references = []
    norm_predictions = []
    norm_references = []

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    eval_dataloader = DataLoader(test_dataset, batch_size=8, collate_fn=data_collator) # type: ignore

    model.eval()
    for step, batch in enumerate(tqdm(eval_dataloader)):
        with torch.amp.autocast("cuda"):
            with torch.no_grad():
                generated_tokens = (
                    model.generate(
                        input_features=batch["input_features"].to("cuda"),
                        decoder_input_ids=batch["labels"][:, :4].to("cuda"),
                        max_new_tokens=255,
                    )
                    .cpu()
                    .numpy()
                )
                labels = batch["labels"].cpu().numpy()
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                
                batch_norm_predictions = [whisper_norm(text) for text in decoded_preds]
                batch_norm_references = [whisper_norm(ref) for ref in decoded_labels]

                 # Append batch results to the overall results
                predictions.extend(decoded_preds)
                norm_predictions.extend(batch_norm_predictions)
                references.extend(decoded_labels)
                norm_references.extend(batch_norm_references)

                
        del generated_tokens, labels, batch
        gc.collect()

    wer = wer_metric.compute(references=references, predictions=predictions)
    wer = round(100 * wer, 2) # type: ignore
    cer = cer_metric.compute(references=references, predictions=predictions)
    cer = round(100 * cer, 2) # type: ignore
    norm_wer = wer_metric.compute(
        references=norm_references, predictions=norm_predictions
    )
    norm_wer = round(100 * norm_wer, 2) # type: ignore
    norm_cer = cer_metric.compute(
        references=norm_references, predictions=norm_predictions
    )
    norm_cer = round(100 * norm_cer, 2) # type: ignore

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

def evaluate_model(model_id: str, test_dataset: Dataset):
    print(model_id)
    whisper_asr = pipeline(
        "automatic-speech-recognition", model=model_id, device_map="auto", batch_size=8
    )
    
    predictions = []
    references = []
    norm_predictions = []
    norm_references = []

    batch_size = 8
    for start_idx in tqdm(range(0, len(test_dataset), batch_size)):
        end_idx = start_idx + batch_size
        batch = test_dataset[start_idx:end_idx]

        #  Extract audio and references in batch
        batch_audio = batch["audio"]

        # Normalize references

        batch_outputs = whisper_asr(
            batch_audio,
            generate_kwargs={"language": "nl", "task": "transcribe"},
            max_new_tokens=255,
        )
        batch_texts = [output["text"] for output in batch_outputs] # type: ignore
        batch_norm_predictions = [whisper_norm(text) for text in batch_texts]
        batch_norm_references = [whisper_norm(ref) for ref in batch["sentence"]]

        # Append batch results to the overall results
        predictions.extend(batch_texts)
        norm_predictions.extend(batch_norm_predictions)
        references.extend(batch["sentence"])
        norm_references.extend(batch_norm_references)

    wer = wer_metric.compute(references=references, predictions=predictions)
    wer = round(100 * wer, 2) # type: ignore
    cer = cer_metric.compute(references=references, predictions=predictions)
    cer = round(100 * cer, 2) # type: ignore
    norm_wer = wer_metric.compute(
        references=norm_references, predictions=norm_predictions
    )
    norm_wer = round(100 * norm_wer, 2) # type: ignore
    norm_cer = cer_metric.compute(
        references=norm_references, predictions=norm_predictions
    )
    norm_cer = round(100 * norm_cer, 2) # type: ignore

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
