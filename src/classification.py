import torch
from torch.nn import CrossEntropyLoss, Dropout, Linear
from transformers import Qwen2_5_VLPreTrainedModel, Qwen2_5_VLModel, SmolVLMPreTrainedModel, SmolVLMModel, SmolVLMConfig, LlamaConfig, SmolVLMForConditionalGeneration
from transformers.modeling_outputs import SequenceClassifierOutput

class QwenVisionModelWithClassification(Qwen2_5_VLPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.vision_language_model = Qwen2_5_VLModel(config)

        self.dropout = Dropout(0.1)
        self.classifier = Linear(config.hidden_size, config.num_labels)
        
        self.post_init()
    
    def forward(
            self,
            input_ids = None,
            attention_mask = None,
            pixel_values = None,
            labels = None,
            **kwargs,
    ):
        outputs = self.vision_language_model(
            input_ids,attention_mask,pixel_values, **kwargs
        )

        
        # TODO: take a look at this
        last_hidden_state = outputs.last_hidden_state

        sequence_output = last_hidden_state[:,-1,:]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return SequenceClassifierOutput(
            loss = loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class SmolVisionModel(SmolVLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.vision_language_model = SmolVLMModel(config)

        self.hidden_size = config.text_config.hidden_size

        self.dropout = Dropout(0.1)
        self.classifier = Linear(self.hidden_size, config.num_labels)
        
        self.post_init()
    
    def forward(
            self,
            input_ids = None,
            attention_mask = None,
            pixel_values = None,
            labels = None,
            **kwargs,
    ):
        outputs = self.vision_language_model(
            input_ids=input_ids,attention_mask=attention_mask,pixel_values=pixel_values, **kwargs
        )

        
        # TODO: take a look at this
        last_hidden_state = outputs.last_hidden_state
        pooled_output = last_hidden_state[:, -1, :]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return SequenceClassifierOutput(
            loss = loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

