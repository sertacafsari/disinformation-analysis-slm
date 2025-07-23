from torch.nn import CrossEntropyLoss, Dropout, Linear
from transformers import SmolVLMPreTrainedModel, SmolVLMModel
from transformers.modeling_outputs import SequenceClassifierOutput


class SmolVisionModelForClassification(SmolVLMPreTrainedModel):
    """
        A class of SmolVLM2 to perform classification tasks.
        It adds a linear classification layer with dropout to the base model.
        Inherits configuration from SmolVLMPreTrainedModel.
    """
    def __init__(self, config):

        # Initialize Pretrained model
        super().__init__(config)

        self.num_labels = config.num_labels
        self.config = config

        # Initialize the VLM with text and image encoders
        self.vision_language_model = SmolVLMModel(config)

        self.hidden_size = config.text_config.hidden_size

        # Dropout
        self.dropout = Dropout(0.1)

        # The linear classification layer
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

        # Get the last hidden state
        last_hidden_state = outputs.last_hidden_state

        # Select the final token’s hidden-state embedding
        pooled_output = last_hidden_state[:, -1, :]

        # Apply dropout to the pooled output
        pooled_output = self.dropout(pooled_output)

        # Get the logits 
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            # Calculate the loss
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        # Return model with classification head
        return SequenceClassifierOutput(
            loss = loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )