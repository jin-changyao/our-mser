import torch
import torch.nn as nn
import torch.nn.functional as F

def get_params(model):
    total_params = sum(p.numel() for p in model.parameters())  # Total parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # Trainable parameters

    print(f"Total number of parameters: {total_params}")
    print(f"Trainable number of parameters: {trainable_params}")

class speechLLM(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        llm: nn.Module,
        encoder_projector: nn.Module,
        config,
    ):
        super().__init__()
        # modality encoder 
        self.encoder =  encoder
        # print('speech encoder:')
        # get_params(self.encoder)
        # projector
        self.encoder_projector = encoder_projector
        # print('speech projector:')
        # get_params(self.encoder_projector)

        # llm
        self.llm = llm
        # print('LLM:')
        # get_params(self.llm)
        self.config = config

    def forward(self,
                input_ids,
                attention_mask,
                audio_features,
                audio_masks,
                labels=None,
                mode='train',             
                ):
        # audio encoder
        results = self.encoder(audio_features, audio_masks, output_attentions=True)
        encoder_outs = results.last_hidden_state

        # find attention masks for encoder outputs
        A_features = []
        encoder_masks_idx = []
        for batch in range(encoder_outs.shape[0]):
            layer = 0
            while layer < 24:
                try:
                    padding_idx = sum(results.attentions[layer][batch][0][0] != 0)
                    encoder_masks_idx.append(padding_idx)
                    break
                except:
                    layer += 1
            truncated_feature = torch.mean(encoder_outs[batch][:padding_idx],0) #Shape is [768]
            A_features.append(truncated_feature)
        A_features = torch.stack(A_features,0).to(encoder_outs.device) #Shape is [batch,768]
        
        encoder_masks = torch.zeros([encoder_outs.shape[0], encoder_outs.shape[1]])
        for i, padding_idx in enumerate(encoder_masks_idx):
            encoder_masks[i][:padding_idx] = 1
        encoder_masks = encoder_masks.to(encoder_outs.device)       

        # projector
        if self.config.projector == "q-former":
            encoder_outs = self.encoder_projector(encoder_outs, encoder_masks)
            ## can be removed
            encoder_masks = torch.ones(encoder_outs.size()[:-1], dtype=torch.long).to(encoder_outs.device)
        elif self.config.projector == "linear":
            encoder_outs = self.encoder_projector(encoder_outs)
            encoder_masks = torch.ones(encoder_outs.size()[:-1], dtype=torch.long).to(encoder_outs.device)

        # embed tokens
        input_ids[input_ids == -1] = 0 # need to check
        if hasattr(self.llm.model, "embed_tokens"):
            inputs_embeds = self.llm.model.embed_tokens(input_ids)
        elif hasattr(self.llm.model.model, "embed_tokens"):
            inputs_embeds = self.llm.model.model.embed_tokens(input_ids)     
        else:
            inputs_embeds = self.llm.model.model.model.embed_tokens(input_ids)

        # concat inputs
        # print(encoder_outs.shape, inputs_embeds.shape)
        inputs_embeds = torch.cat([encoder_outs, inputs_embeds], 1)
        # print(inputs_embeds.shape)

        # concat attention masks
        # print(encoder_masks.shape, attention_mask.shape)
        attention_mask = torch.cat([encoder_masks, attention_mask], 1)
        # print(attention_mask.shape)

        #inference
        if mode=='inference':
            return inputs_embeds, attention_mask

        # concat labels
        encoder_labels = torch.ones(encoder_masks.size(), dtype=torch.long).to(encoder_masks.device)
        encoder_labels[:,:] = -100
        labels = torch.cat([encoder_labels, labels], 1)

        # LLM
        #print(inputs_embeds.shape, attention_mask.shape, labels.shape)
        model_outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, return_dict=True)
        return model_outputs
    
    @torch.no_grad()
    def generate(self,
                input_ids,
                attention_mask,
                audio_features,
                audio_masks,            
                ):

        inputs_embeds, attention_mask = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_features=audio_features,
            audio_masks=audio_masks,
            mode='inference'
        )

        model_outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            num_beams=self.config.num_beams,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            # early_stopping=True,
            max_length=self.config.max_length,
            # length_penalty=0.1,
            repetition_penalty=1.0,
            num_return_sequences=1
        )

        return model_outputs