from peft.mixed_model import PeftMixedModel
from peft.peft_model import PeftModel


def merge_peft_model(peft_model: PeftModel | PeftMixedModel, output_path: str):
    merged_model = peft_model.merge_and_unload()

    merged_model.save_pretrained(output_path)

    return output_path
    