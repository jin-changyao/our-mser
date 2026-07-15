import torch
import torch.nn as nn
import torch.nn.functional as F

def get_params(model):
    total_params = sum(p.numel() for p in model.parameters())  # Total parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # Trainable parameters

    print(f"Total number of parameters: {total_params}")
    print(f"Trainable number of parameters: {trainable_params}")


def get_lm_input_embeddings(model):
    if hasattr(model, "get_input_embeddings"):
        embeddings = model.get_input_embeddings()
        if embeddings is not None:
            return embeddings
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens
    if hasattr(model, "model") and hasattr(model.model, "model") and hasattr(model.model.model, "embed_tokens"):
        return model.model.model.embed_tokens
    if (
        hasattr(model, "model")
        and hasattr(model.model, "model")
        and hasattr(model.model.model, "model")
        and hasattr(model.model.model.model, "embed_tokens")
    ):
        return model.model.model.model.embed_tokens
    raise AttributeError("Cannot find LLM input embedding layer.")


class AVPrefixProjector(nn.Module):
    def __init__(self, input_dim, hidden_size, num_tokens, dropout=0.05):
        super().__init__()
        self.num_tokens = num_tokens
        self.hidden_size = hidden_size
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_tokens * hidden_size),
        )

    def forward(self, features):
        tokens = self.net(features)
        return tokens.view(features.size(0), self.num_tokens, self.hidden_size)


class TextGuidedPrefixAdapter(nn.Module):
    def __init__(self, hidden_size, num_tokens, dropout=0.05):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_tokens = num_tokens
        self.base_norm = nn.LayerNorm(hidden_size)
        self.film = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size * 2),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, prefix_tokens, text_rep):
        base = self.base_norm(prefix_tokens)
        gamma, beta = self.film(text_rep).chunk(2, dim=-1)
        guided = base * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        pooled_base = base.mean(dim=1)
        gate = torch.sigmoid(self.gate(torch.cat([text_rep, pooled_base], dim=-1)))
        prefix = gate.unsqueeze(1) * guided + (1.0 - gate.unsqueeze(1)) * base
        return prefix, gate.squeeze(-1)


class AVPrefixLLM(nn.Module):
    def __init__(
        self,
        llm: nn.Module,
        config,
    ):
        super().__init__()
        self.llm = llm
        self.config = config
        self.audio_projector = AVPrefixProjector(
            config.mm_audio_dim,
            config.mm_hidden_size,
            config.mm_audio_tokens,
            config.mm_projector_dropout,
        )
        self.video_projector = AVPrefixProjector(
            config.mm_video_dim,
            config.mm_hidden_size,
            config.mm_video_tokens,
            config.mm_projector_dropout,
        )
        self.text_guided_mm = bool(getattr(config, "text_guided_mm", False))
        self.text_guided_audio = bool(getattr(config, "text_guided_audio", True))
        self.text_guided_video = bool(getattr(config, "text_guided_video", True))
        self.log_mm_gates = bool(getattr(config, "log_mm_gates", False))
        self.last_mm_gate_stats = {}
        if self.text_guided_mm:
            self.audio_adapter = TextGuidedPrefixAdapter(
                config.mm_hidden_size,
                config.mm_audio_tokens,
                config.mm_projector_dropout,
            )
            self.video_adapter = TextGuidedPrefixAdapter(
                config.mm_hidden_size,
                config.mm_video_tokens,
                config.mm_projector_dropout,
            )

    def _pool_target_text(self, target_text_input_ids, target_text_attention_mask, dtype):
        if target_text_input_ids is None or target_text_attention_mask is None:
            raise ValueError("TEXT_GUIDED_MM=True requires target_text_input_ids and target_text_attention_mask.")
        device = next(self.llm.parameters()).device
        target_text_input_ids = target_text_input_ids.to(device)
        target_text_attention_mask = target_text_attention_mask.to(device)
        # The guide text is a conditioning signal. Keeping it out of the
        # backward graph avoids reducing the same LLM/LoRA parameters twice
        # when DeepSpeed gradient checkpointing is enabled.
        with torch.no_grad():
            text_outputs = self.llm(
                input_ids=target_text_input_ids,
                attention_mask=target_text_attention_mask,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            hidden = text_outputs.hidden_states[-1].to(dtype)
        mask = target_text_attention_mask.to(hidden.device).unsqueeze(-1).to(hidden.dtype)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return ((hidden * mask).sum(dim=1) / denom).detach()

    def _collect_gate_stats(self, gates):
        stats = {}
        for name, gate in gates.items():
            gate = gate.detach().float()
            stats[f"{name}_mean"] = gate.mean().item()
            stats[f"{name}_std"] = gate.std(unbiased=False).item()
            stats[f"{name}_min"] = gate.min().item()
            stats[f"{name}_max"] = gate.max().item()
        self.last_mm_gate_stats = stats
        return stats

    def build_inputs_embeds(
        self,
        input_ids,
        attention_mask,
        mm_audio_features,
        mm_video_features,
        labels=None,
        target_text_input_ids=None,
        target_text_attention_mask=None,
    ):
        input_ids = input_ids.clone()
        input_ids[input_ids == -1] = 0
        embed_tokens = get_lm_input_embeddings(self.llm)
        text_embeds = embed_tokens(input_ids)
        dtype = text_embeds.dtype
        audio_tokens = self.audio_projector(mm_audio_features.to(text_embeds.device).to(dtype))
        video_tokens = self.video_projector(mm_video_features.to(text_embeds.device).to(dtype))
        if self.text_guided_mm:
            text_rep = self._pool_target_text(
                target_text_input_ids=target_text_input_ids,
                target_text_attention_mask=target_text_attention_mask,
                dtype=dtype,
            )
            prefix_parts = []
            gates = {}
            if self.text_guided_audio:
                audio_tokens, gates["gate_a"] = self.audio_adapter(audio_tokens, text_rep)
                prefix_parts.append(audio_tokens)
            if self.text_guided_video:
                video_tokens, gates["gate_v"] = self.video_adapter(video_tokens, text_rep)
                prefix_parts.append(video_tokens)
            if not prefix_parts:
                raise ValueError("TEXT_GUIDED_MM=True requires TEXT_GUIDED_AUDIO=True or TEXT_GUIDED_VIDEO=True.")
            prefix_embeds = torch.cat(prefix_parts, dim=1)
            if self.log_mm_gates:
                self._collect_gate_stats(gates)
        else:
            prefix_embeds = torch.cat([audio_tokens, video_tokens], dim=1)
        inputs_embeds = torch.cat([prefix_embeds, text_embeds], dim=1)

        prefix_mask = torch.ones(
            input_ids.size(0),
            prefix_embeds.size(1),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        if labels is not None:
            prefix_labels = torch.full(
                (labels.size(0), prefix_embeds.size(1)),
                -100,
                dtype=labels.dtype,
                device=labels.device,
            )
            labels = torch.cat([prefix_labels, labels], dim=1)
        return inputs_embeds, attention_mask, labels

    def forward(
        self,
        input_ids,
        attention_mask,
        mm_audio_features,
        mm_video_features,
        labels=None,
        target_text_input_ids=None,
        target_text_attention_mask=None,
        **kwargs,
    ):
        inputs_embeds, attention_mask, labels = self.build_inputs_embeds(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mm_audio_features=mm_audio_features,
            mm_video_features=mm_video_features,
            labels=labels,
            target_text_input_ids=target_text_input_ids,
            target_text_attention_mask=target_text_attention_mask,
        )
        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        attention_mask,
        mm_audio_features,
        mm_video_features,
        target_text_input_ids=None,
        target_text_attention_mask=None,
        **kwargs,
    ):
        inputs_embeds, attention_mask, _ = self.build_inputs_embeds(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mm_audio_features=mm_audio_features,
            mm_video_features=mm_video_features,
            labels=None,
            target_text_input_ids=target_text_input_ids,
            target_text_attention_mask=target_text_attention_mask,
        )
        return self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )

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
