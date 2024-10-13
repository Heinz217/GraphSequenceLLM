#    Copyright 2023 Haotian Liu
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



from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from utils.constants import IGNORE_INDEX

from model.gls_arch import GslMetaModel, GslMetaForCausalLM


class GSLConfig(LlamaConfig):
    model_type = "gls"


class GSLLlamaModel(GslMetaModel, LlamaModel):
    config_class = GSLConfig

    def __init__(self, config: LlamaConfig):
        super(GSLLlamaModel, self).__init__(config)


class GlsLlamaForCausalLM(LlamaForCausalLM, GslMetaForCausalLM):
    config_class = GSLConfig

    def __init__(self, config):
        super(GlsLlamaForCausalLM, self).__init__(config)
        self.model = GSLLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            graph: Optional[torch.FloatTensor] = None,
            graph_emb: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids, attention_mask, position_ids, inputs_embeds, labels = self._prepare_inputs(input_ids, attention_mask, past_key_values, labels, graph, graph_emb)


        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)  # hidden_states: (batch_size, seq_len, hidden_size) -> logits: (batch_size, seq_len, vocab_size)
        # 这里的lm_head是一个全连接层，将hidden_size映射到vocab_size(有过定义)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()  # logits[..., :-1, :]  # shape: (batch_size, seq_len, vocab_size)
            shift_labels = labels[..., 1:].contiguous()  # 确保tensor在内存中是连续的

            loss_fct = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)

            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "graph": kwargs.get("graph", None),
                "graph_emb": kwargs.get("graph_emb", None),
            }
        )

AutoConfig.register("gls", GSLConfig)
AutoModelForCausalLM.register(GSLConfig, GlsLlamaForCausalLM)


