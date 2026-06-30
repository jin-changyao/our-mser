import torch
import torch.nn as nn


class EncoderProjectorConcat(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.k = int(config.downsample_rate)
        if config.encoder_size=='small':
            self.encoder_dim = 768
        else:
            self.encoder_dim = 1024
        if config.feature == 'smile':
            self.encoder_dim = 130

        self.llm_dim = 4096
        self.linear1 = nn.Linear(self.encoder_dim * self.k, 2048)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2048, self.llm_dim)

    def forward(self, x):
        batch_size, seq_len, dim = x.size()
        num_frames_to_discard = seq_len % self.k
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        seq_len = x.size(1)
        
        x = x.contiguous()
        x = x.view(batch_size, seq_len // self.k, dim * self.k)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class EncoderProjectorQFormer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.k = int(config.downsample_rate)
        if config.encoder_size=='small':
            self.encoder_dim = 768
        else:
            self.encoder_dim = 1024
        self.llm_dim = 4096
        from transformers import Blip2QFormerConfig, Blip2QFormerModel
        configuration = Blip2QFormerConfig()
        configuration.encoder_hidden_size = self.encoder_dim
        configuration.num_hidden_layers = 5

        self.query_len = 64
        self.query = nn.Parameter(torch.zeros(1, self.query_len, configuration.hidden_size))
        self.query.data.normal_(mean=0.0, std=1.0)
        self.qformer = Blip2QFormerModel(configuration)

        self.linear1 = nn.Linear(configuration.hidden_size * self.k, 2048)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2048, self.llm_dim)
        self.dropout = nn.Dropout(0.3)
        # self.norm = nn.LayerNorm(self.llm_dim, eps=1e-5)

    def forward(self, x, atts):
        query = self.query.expand(x.shape[0], -1, -1)
        
        query_output = self.qformer(
            query_embeds=query,
            encoder_hidden_states=x,
            encoder_attention_mask=atts,
            return_dict=True,
        )

        x = query_output.last_hidden_state
        batch_size, seq_len, dim = x.size()
        num_frames_to_discard = seq_len % self.k
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        seq_len = x.size(1)
        
        x = x.contiguous()
        x = x.view(batch_size, seq_len // self.k, dim * self.k)

        query_proj = self.dropout(x)
        query_proj = self.linear1(query_proj)
        query_proj = self.relu(query_proj)
        query_proj = self.linear2(query_proj)

        #print(query_proj.shape)
        return query_proj