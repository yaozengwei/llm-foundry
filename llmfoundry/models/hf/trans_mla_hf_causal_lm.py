# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

# Modified from llmfoundry/models/hf/hf_causal_lm.py

"""Implements a Hugging Causal LM wrapped inside a :class:`.ComposerModel`."""

import logging

from transformers import PreTrainedModel
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

from llmfoundry.models.hf.hf_causal_lm import ComposerHFCausalLM
from llmfoundry.models.hf.trans_mla.qwen2.modeling_qwen2 import Qwen2ForCausalLM as TransMLAQwen2ForCausalLM

__all__ = ['TransMLAHFCausalLM']

log = logging.getLogger(__name__)


class TransMLAHFCausalLM(ComposerHFCausalLM):
    # Function transform_model is defined in class BaseHuggingFaceModel.
    # Override this function to transform the transformers models to the MLA-version.
    def transform_model(self, model: PreTrainedModel) -> PreTrainedModel:
        if isinstance(model, Qwen2ForCausalLM):
            new_model = TransMLAQwen2ForCausalLM(model.config)
        else:
            raise TypeError(f"Model class of {type(model)} is not supported.")
        return new_model
