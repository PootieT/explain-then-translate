# from .local_huggingface_model import LocalHuggingfaceModel
from inference.local_huggingface_model import LocalHuggingfaceModel

_model = LocalHuggingfaceModel(
    "Salesforce/codegen2-1B",
    model_kwargs=dict(trust_remote_code=True),
    set_pad_token_id_plus_2=True
)

name = "codegen21b"

completions = _model.completions
