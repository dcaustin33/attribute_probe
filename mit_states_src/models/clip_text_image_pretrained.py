from PIL import Image
import requests
import torch

from transformers import CLIPProcessor, CLIPModel
import torch.nn as nn

class CLIP_text_image(nn.Module):

    def __init__(self, args = None):
        super().__init__()
        self.args = args
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.linear1 = nn.Linear(312, 312)
        self.classifier1 = nn.Sequential(nn.Linear(312, 312), nn.ReLU(), nn.BatchNorm1d(312), nn.Linear(312, 312))

    def forward(self, prompts, images):
        text = prompts
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        for i in inputs:
            inputs[i] = inputs[i].cuda()
        inputs['pixel_values'] = images.cuda()
        outputs = self.clip(**inputs)

        binary_pred1 = self.linear1(outputs.logits_per_image)
        classifier_pred1 = self.classifier1(outputs.logits_per_image)


        return (binary_pred1, classifier_pred1), outputs.logits_per_image


class CLIP_text_image_concat(nn.Module):

    def __init__(self, adjectives: int, nouns: int, concat: int, args = None):
        super().__init__()
        self.args = args
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.linear_adjectives = nn.ModuleList([nn.Linear(768 + 512, 1) for i in range(adjectives)])
        self.linear_nouns = nn.ModuleList([nn.Linear(768 + 512, 1) for i in range(nouns)])
        #self.linear_concat = nn.ModuleList([nn.Linear(768 + 512, 1) for i in range(concat)])

        self.classifier_adjectives =nn.ModuleList([nn.Sequential(nn.Linear(768 + 512, 312), nn.ReLU(), nn.BatchNorm1d(312), nn.Linear(312, 1)) for i in range(adjectives)])
        self.classifier_nouns = nn.ModuleList([nn.Sequential(nn.Linear(768 + 512, 312), nn.ReLU(), nn.BatchNorm1d(312), nn.Linear(312, 1)) for i in range(nouns)])
        #self.classifier_concat = nn.ModuleList([nn.Sequential(nn.Linear(768 + 512, 312), nn.ReLU(), nn.BatchNorm1d(312), nn.Linear(312, 1)) for i in range(concat)])
        
        self.adjective_text_embeddings = None
        self.noun_text_embeddings = None


    def create_text_embeddings(self, adjectives, nouns):
        with torch.no_grad():
            text = adjectives 
            inputs = self.processor(text=text, return_tensors="pt", padding=True)
            for i in inputs:
                inputs[i] = inputs[i].cuda()
            images = torch.randn(2, 3, 224, 224).cuda()
            inputs['pixel_values'] = images
            self.adjective_text_embeddings = self.clip(**inputs).text_embeds

            text = nouns
            inputs = self.processor(text=text, return_tensors="pt", padding=True)
            for i in inputs:
                inputs[i] = inputs[i].cuda()
            images = torch.randn(2, 3, 224, 224).cuda()
            inputs['pixel_values'] = images
            self.noun_text_embeddings = self.clip(**inputs).text_embeds
            return


    def forward(self, images):
        text = ['']
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        for i in inputs:
            inputs[i] = inputs[i].cuda()
        inputs['pixel_values'] = images.cuda()
        outputs = self.clip(**inputs)

        image_embed = outputs.image_embeds
        image_out = outputs.vision_model_output['pooler_output']

        classifications1 = []

        #first do the adjectives
        for i in range(self.adjective_text_embeddings.size(0)):
            new_text_embed = self.adjective_text_embeddings[i].unsqueeze(0).repeat(image_embed.size(0), 1)
            new_text_out = self.adjective_text_embeddings[i].unsqueeze(0).repeat(image_out.size(0), 1)
            final_embed = torch.cat((image_embed, new_text_embed), dim=1)
            final_out = torch.cat((image_out, new_text_out), dim=1)

            adjective_linear = self.linear_adjectives[i](final_out)

            adjective_classifier = self.classifier_adjectives[i](final_out)


            inter_class = [adjective_linear, adjective_classifier]
            if i == 0:
                classifications1 = inter_class

            else:
                classifications1[0] = torch.cat((classifications1[0], inter_class[0]), dim=1)
                classifications1[1] = torch.cat((classifications1[1], inter_class[1]), dim=1)

        classifications2 = []
        #now do the nouns
        for i in range(self.noun_text_embeddings.size(0)):
            new_text_embed = self.noun_text_embeddings[i].unsqueeze(0).repeat(image_embed.size(0), 1)
            new_text_out = self.noun_text_embeddings[i].unsqueeze(0).repeat(image_out.size(0), 1)
            final_embed = torch.cat((image_embed, new_text_embed), dim=1)
            final_out = torch.cat((image_out, new_text_out), dim=1)

            noun_linear = self.linear_nouns[i](final_out)

            noun_classifier = self.classifier_nouns[i](final_out)


            inter_class = [noun_linear, noun_classifier]
            if i == 0:
                classifications2 = inter_class

            else:
                classifications2[0] = torch.cat((classifications2[0], inter_class[0]), dim=1)
                classifications2[1] = torch.cat((classifications2[1], inter_class[1]), dim=1)

            
        classifications = [classifications1[0], classifications2[0], classifications1[1], classifications2[1]]

        return classifications, outputs.logits_per_image


class CLIP_text_image_concat_embeddings(nn.Module):

    def __init__(self, adjectives: int, nouns: int, concat: int, args = None):
        super().__init__()
        self.args = args
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        self.clip.text_transformer = newCLIPTextTransformer(self.clip.text_transformer)

        self.linear_adjectives = nn.ModuleList([nn.Linear(768 + 512, 1) for i in range(adjectives)])
        self.linear_nouns = nn.ModuleList([nn.Linear(768 + 512, 1) for i in range(nouns)])
        self.linear_concat = nn.ModuleList([nn.Linear(768 + 512, 1) for i in range(concat)])

        self.classifier_adjectives =nn.ModuleList([nn.Sequential(nn.Linear(768 + 512, 312), nn.ReLU(), nn.BatchNorm1d(312), nn.Linear(312, 1)) for i in range(adjectives)])
        self.classifier_nouns = nn.ModuleList([nn.Sequential(nn.Linear(768 + 512, 312), nn.ReLU(), nn.BatchNorm1d(312), nn.Linear(312, 1)) for i in range(nouns)])
        self.classifier_concat = nn.ModuleList([nn.Sequential(nn.Linear(768 + 512, 312), nn.ReLU(), nn.BatchNorm1d(312), nn.Linear(312, 1)) for i in range(concat)])

    def forward(self, prompts, images):
        text = prompts
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        for i in inputs:
            inputs[i] = inputs[i].cuda()
        inputs['pixel_values'] = images.cuda()
        outputs = self.clip(**inputs)

        image_embed = outputs.image_embeds
        image_out = outputs.vision_model_output['pooler_output']
        text_embed = outputs.text_embeds
        text_out = outputs.text_model_output['pooler_output']

        classifications = []

        for i in range(text_embed.size(0)):
            new_text_embed = text_embed[i].unsqueeze(0).repeat(image_embed.size(0), 1)
            new_text_out = text_out[i].unsqueeze(0).repeat(image_out.size(0), 1)
            final_embed = torch.cat((image_embed, new_text_embed), dim=1)
            final_out = torch.cat((image_out, new_text_out), dim=1)

            adjective_linear = self.linear_adjectives[i](final_out)
            noun_linear = self.linear_nouns[i](final_out)
            concat_linear = self.linear_concat[i](final_out)

            adjective_classifier = self.classifier_adjectives[i](final_out)
            noun_classifier = self.classifier_nouns[i](final_out)
            concat_classifier = self.classifier_concat[i](final_out)


            inter_class = [adjective_linear, noun_linear, concat_linear, adjective_classifier, noun_classifier, concat_classifier]
            if i == 0:
                classifications = inter_class

            else:
                classifications[0] = torch.cat((classifications[0], inter_class[0]), dim=1)
                classifications[1] = torch.cat((classifications[1], inter_class[1]), dim=1)
                classifications[2] = torch.cat((classifications[2], inter_class[2]), dim=1)
                classifications[3] = torch.cat((classifications[3], inter_class[3]), dim=1)
                classifications[4] = torch.cat((classifications[4], inter_class[4]), dim=1)
                classifications[5] = torch.cat((classifications[5], inter_class[5]), dim=1)

        return classifications, outputs.logits_per_image



#in case we do not want to use a prompt but instead use an embedding of prompts
from transformers.models.clip.modeling_clip import *

def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class newCLIPTextTransformer(CLIPTextTransformer):
    def __init__(self, old_text_transformer, first_dim: int = 10, second_dim: int = 10):
        super().__init__(old_text_transformer.config, )
        self.embeddings = torch.nn.Embedding(100, 512)
        self.old_text_transformer = old_text_transformer
        self.get_all = torch.LongTensor([i for i in range(100)]).cuda()
        self.first_dim = first_dim
        self.second_dim = second_dim

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:
        """
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None:
            raise ValueError("You have to specify either input_ids")
        input_ids = torch.ones(self.first_dim, self.second_dim).cuda()
        attention_mask = torch.ones(self.first_dim, self.second_dim).cuda()
        input_shape = (self.first_dim, self.second_dim)
        input_ids = input_ids.view(-1, input_shape[-1])

        

        bsz, seq_len = input_shape
        hidden_states = self.embeddings(self.get_all).view(self.first_dim, self.second_dim, -1)
        
        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = self.old_text_transformer._build_causal_attention_mask(bsz, seq_len, hidden_states.dtype).to(
            hidden_states.device
        )
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.old_text_transformer.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.old_text_transformer.final_layer_norm(last_hidden_state)

        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0]), input_ids.to(torch.int).argmax(dim=-1)
        ]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class CLIP_text_image_with_attribute(nn.Module):

    def __init__(self, adjectives: int, nouns: int, concat: int, args = None):
        super().__init__()
        self.args = args
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        self.linear_adjectives = nn.Linear(768 + 512, adjectives)
        self.linear_nouns = nn.Linear(768 + 512, nouns)

        self.classifier_adjectives =  nn.Sequential(nn.Linear(768 + 512, 312), nn.ReLU(), nn.BatchNorm1d(312), nn.Linear(312, adjectives))
        self.classifier_nouns = nn.Sequential(nn.Linear(768 + 512, 312), nn.ReLU(), nn.BatchNorm1d(312), nn.Linear(312, nouns))

    def forward(self, prompts, images):
        text = prompts
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        for i in inputs:
            inputs[i] = inputs[i].cuda()
        inputs['pixel_values'] = images.cuda()
        outputs = self.clip(**inputs)

        image_embed = outputs.image_embeds
        image_out = outputs.vision_model_output['pooler_output']
        text_embed = outputs.text_embeds
        text_out = outputs.text_model_output['pooler_output']

        classifications = []

        #now we concatenate the image and text embeddings
        embed = torch.cat((image_embed, text_embed), dim=1)
        out = torch.cat((image_out, text_out), dim=1)

        classifications.append(self.linear_adjectives(out))
        classifications.append(self.linear_nouns(out))

        classifications.append(self.classifier_adjectives(out))
        classifications.append(self.classifier_nouns(out))


        return classifications, outputs.logits_per_image