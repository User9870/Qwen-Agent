# Copyright 2023 The Qwen team, Alibaba Group. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
from typing import Union

from .azure import TextChatAtAzure
from .base import LLM_REGISTRY, BaseChatModel, ModelServiceError
from .manager import LLMManager
from .oai import TextChatAtOAI
from .openvino import OpenVINO
from .transformers_llm import Transformers
from .qwen_dashscope import QwenChatAtDS
from .qwenaudio_dashscope import QwenAudioChatAtDS
from .qwenomni_oai import QwenOmniChatAtOAI
from .qwenvl_dashscope import QwenVLChatAtDS
from .qwenvl_oai import QwenVLChatAtOAI



__all__ = [
    'BaseChatModel',
    'LLMManager',
    'QwenChatAtDS',
    'TextChatAtOAI',
    'TextChatAtAzure',
    'QwenVLChatAtDS',
    'QwenVLChatAtOAI',
    'QwenAudioChatAtDS',
    'QwenOmniChatAtOAI',
    'OpenVINO',
    'Transformers',
    'ModelServiceError',
]
