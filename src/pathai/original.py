"""Import the supplied project modules directly, without copying their GPT."""

import importlib.util
import os
import sys
from pathlib import Path

PROJECT = (
    Path(
        os.environ.get(
            "AESCLEPIUS_PROJECT", Path(__file__).resolve().parents[3] / "aesclepius-jr"
        )
    )
    .expanduser()
    .resolve()
)


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, PROJECT / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


gpt_module = _load_module("model", "model.py")
training_module = _load_module("aesclepius_train", "train.py")
ToyAesclea = gpt_module.ToyAesclea
ToyConfig = gpt_module.ToyConfig
collate = training_module.collate
make_batch = training_module.make_batch
TOKEN_TO_ID = training_module.TOKEN_TO_ID
