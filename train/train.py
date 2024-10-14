# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence
import pandas as pd

import torch

import transformers

from utils.constants import IGNORE_INDEX, DEFAULT_GRAPH_TOKEN, DEFAULT_GRAPH_START_TOKEN, DEFAULT_GRAPH_END_TOKEN, DEFAULT_GRAPH_PAD_ID
from torch.utils.data import Dataset
from llaga_trainer import LLaGATrainer

from model import *

import random
from tqdm import trange
from utils import conversation as conversation_lib
from utils.utils import tokenizer_graph_token
import scipy.sparse as sp
import numpy as np



