import os
import json

from convert.merge_peft_model import merge_peft_model
from convert.convert_to_ctranslate2 import convert_to_ctranslate2
from evaluation.evaluation import evaluate_model
from peft.mapping import get_peft_model
from peft import (get_peft_model, LoraConfig, prepare_model_for_kbit_training)
from transformers import (
    BitsAndBytesConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizerFast,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from train.data_collator import DataCollatorSpeechSeq2SeqWithPadding
from train.load_datasets import load_datasets
from train.prepare_dataset import map_prepare_dataset

# VARIABLES
model_name_or_path = "openai/whisper-large-v3"
language_abbr = "nl"
task = "transcribe"
peft_model_id = 'models/large-v3-big-dataset-v2'

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
dataset = map_prepare_dataset(dataset, feature_extractor=feature_extractor, tokenizer=tokenizer)

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# LOAD MODEL
model = WhisperForConditionalGeneration.from_pretrained(
    model_name_or_path, quantization_config=BitsAndBytesConfig(load_in_8bit=True)
)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

model.freeze_encoder()
model.model.encoder.gradient_checkpointing = False

model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)

model = get_peft_model(model, config)


training_args = Seq2SeqTrainingArguments(
    output_dir=peft_model_id,  # change to a repo name of your choice
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-3,
    warmup_ratio=0.1,
    num_train_epochs=6,
    logging_strategy="epoch",
    save_strategy="epoch",
    save_only_model=True,
    eval_strategy="no",
    fp16=True,
    per_device_eval_batch_size=16,
    generation_max_length=128,
    remove_unused_columns=False,  # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
    label_names=["labels"],  # same reason as above
)


class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control


trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset['train'],
    data_collator=data_collator,
    tokenizer=processor.feature_extractor,
    # callbacks=[SavePeftModelCallback],
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!


trainer.train()

model.save_pretrained(training_args.output_dir)
tokenizer.save_pretrained(training_args.output_dir)
processor.save_pretrained(training_args.output_dir)

print(f"Model saved in: {training_args.output_dir}")


merged_model = merge_peft_model(model, training_args.output_dir)

convert_to_ctranslate2(training_args.output_dir, training_args.output_dir + '-ct2')

dataset = load_datasets()
tuned_metrics = evaluate_model(training_args.output_dir, dataset['test'])
print(tuned_metrics)

with open(os.path.join(training_args.output_dir, 'metrics.json'), 'w') as fp:
    json.dump(tuned_metrics, fp)