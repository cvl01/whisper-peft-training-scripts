import os
from ctranslate2.converters import TransformersConverter


def convert_to_ctranslate2(model_path, output_path):
    model_path = os.path.abspath(model_path)

    converter = TransformersConverter(
        model_name_or_path=model_path,
        copy_files=["tokenizer.json", "preprocessor_config.json"],
    )

    converter.convert(output_dir=output_path, quantization="float16", force=True)
