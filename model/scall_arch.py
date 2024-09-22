from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .scGPT_main.scgpt.build import get_cell_embed
from .multimodel_project.build import build_sc_alignment

from constants import IGNORE_INDEX, GENE_TOKEN_INDEX



class scallMetaModel:

    def __init__(self, config):
        super(scallMetaModel, self).__init__(config)

        if hasattr(config, "gene_embed"):
            self.gene_embed = get_cell_embed(config)
            self.gene_projector = build_sc_alignment(config)

    def gene_embed(self):
        gene_embed = getattr(self, 'gene_embed', None)
        return gene_embed

    def initialize_vision_modules(self, model_args):

        pretrain_mm_mlp_resample = model_args.pretrain_mm_mlp_resample

        if pretrain_mm_mlp_resample is not None:
            ca_projector_weights = torch.load(pretrain_mm_mlp_resample, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.gene_projector.load_state_dict(get_w(ca_projector_weights, 'mm_resample'))



class scallMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_cell_embed(self):
        return self.get_model().cell_embed()

    def cell_embedding(self,gene_sequence,gene_expression):
        gene_features = self.get_model().gene_embed().get_emdeds(gene_sequence,gene_expression)
        gene_token = self.get_model().gene_projector(gene_features)
        return gene_token

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        gene_sequence,gene_expression
    ):
        gene_embed = self.cell_embedding(gene_sequence,gene_expression)
        if gene_embed is None or gene_sequence is None or gene_expression in None:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels


        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        for cur_input_ids, gene_token,label in zip(input_ids,gene_embed,labels):
            token_indices =torch.where(cur_input_ids == GENE_TOKEN_INDEX)[0].tolist()[0]

            gene_label=torch.full((gene_token.shape[0],), IGNORE_INDEX, device=label.device, dtype=label.dtype)
            cur_label=torch.cat(label[:token_indices],gene_label,label[token_indices+1:])
            cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:token_indices])
            cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[token_indices+1:])
            cur_new_input_embeds = torch.cat(cur_input_embeds_1,gene_token,cur_input_embeds_2)
            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_label)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

