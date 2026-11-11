import torch
from tests.config import *
from tests.adapters import *

class MyLLM(torch.nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self._config = config
        d_model = config.d_model
        d_ff = config.d_ff
        num_layers = config.num_layers
        vocab_size = config.vocab_size

        # Token embedding matrix. Shape is (vocab_size, d_model)
        self.token_embeddings = torch.nn.ParameterDict({
            'weight': torch.nn.Parameter(torch.empty(vocab_size, d_model)) })
        
        self.layers = torch.nn.ModuleList()
        # layer specific parameters
        for layer_index in range(num_layers):
            layer = torch.nn.ModuleDict({
                'attn': torch.nn.ParameterDict({
                    'q_proj': torch.nn.ParameterDict({
                        'weight': torch.nn.Parameter(torch.empty(d_model, d_model)) }),
                    'k_proj': torch.nn.ParameterDict({
                        'weight': torch.nn.Parameter(torch.empty(d_model, d_model)) }),
                    'v_proj': torch.nn.ParameterDict({
                        'weight': torch.nn.Parameter(torch.empty(d_model, d_model)) }),
                    'output_proj': torch.nn.ParameterDict({
                        'weight': torch.nn.Parameter(torch.empty(d_model, d_model)) }),
                }),
                'ln1': torch.nn.ParameterDict({
                    'weight': torch.nn.Parameter(torch.ones(d_model))  }),
                'ffn': torch.nn.ParameterDict({
                    'w1': torch.nn.ParameterDict({
                        'weight': torch.nn.Parameter(torch.empty(d_ff, d_model)) }),
                    'w2': torch.nn.ParameterDict({
                        'weight': torch.nn.Parameter(torch.empty(d_model, d_ff)) }),
                    'w3': torch.nn.ParameterDict({
                        'weight': torch.nn.Parameter(torch.empty(d_ff, d_model)) })
                }),
                'ln2': torch.nn.ParameterDict({
                    'weight': torch.nn.Parameter(torch.ones(d_model)) })
            })
            self.layers.append(layer)
        
        self.ln_final =  torch.nn.ParameterDict({
                'weight':torch.nn.Parameter(torch.ones(d_model)) })
        self.lm_head =  self.token_embeddings
        # self.lm_head

    def init_params(self):
        num_layers = self._config.num_layers
        d_model = self._config.d_model
        d_ff = self._config.d_ff

        torch.nn.init.normal_(self.token_embeddings['weight'], std=0.02)
        # layer specific parameters
        for layer in self.layers:
            # Shape is (num_heads * (d_model / num_heads), d_model).
            qk_weight_std = (1.0 / d_model) ** 0.5

            torch.nn.init.normal_(layer['attn']['q_proj']['weight'],
                                  mean=0.0, std=qk_weight_std)
            torch.nn.init.normal_(layer['attn']['k_proj']['weight'],
                                  mean=0.0, std=qk_weight_std)
            torch.nn.init.normal_(layer['attn']['v_proj']['weight'],
                                  mean=0.0, std=qk_weight_std)
            torch.nn.init.normal_(layer['attn']['output_proj']['weight'],
                                  mean=0.0, std=(1.0 / (d_model * 2.0 * num_layers)) ** 0.5)

            
            torch.nn.init.normal_(layer['ffn']['w1']['weight'],
                                  mean=0.0, std=(2.0 / d_model) ** 0.5)
            torch.nn.init.normal_(layer['ffn']['w2']['weight'],
                                  mean=0.0, std=(1.0 / (d_ff * 2.0 * num_layers)) ** 0.5)
            torch.nn.init.normal_(layer['ffn']['w3']['weight'],
                                  mean=0.0, std=(2.0 / d_model) ** 0.5)

        torch.nn.init.normal_(self.ln_final['weight'],
                              mean=0.0, std=(1.0 / d_model) ** 0.5)
        

    def forward(self, in_indices):
        named_params = dict(self.named_parameters())
        named_params['lm_head.weight'] = named_params['token_embeddings.weight']
        # print(dict(self.named_parameters()).keys())
        return run_transformer_lm(
                self._config.vocab_size, 
                self._config.max_seq_len, 
                self._config.d_model, 
                self._config.num_layers, 
                self._config.num_heads, 
                self._config.d_ff,
                self._config.rope_theta, 
                named_params, 
                in_indices)
