"""slightly bigger transformer that predicts learned word pieces instead of bytes"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from model import TransformerBlock


@dataclass(frozen=True)
class SubwordConfig:
    # these are experiment settings not magic numbers that guarantee good replies
    # more pieces shorten text but make the input and output tables bigger
    vocab_size: int = 4096
    # this counts word pieces now so it is different from the old byte window
    context_length: int = 256
    # each piece gets this many learned features and each head gets 48 of them
    embedding_dim: int = 192
    # keep four heads and four blocks so we can try more width without too much cost
    num_heads: int = 4
    num_layers: int = 4
    dropout: float = 0.1

    def __post_init__(self):
        # the dataclass calls this after filling in the settings so bad sizes fail early
        if min(self.vocab_size, self.context_length, self.embedding_dim, self.num_heads, self.num_layers) < 1:
            raise ValueError("Model sizes must be positive.")
        if self.embedding_dim % self.num_heads or not 0 <= self.dropout < 1:
            raise ValueError("Heads must divide the width and dropout must be 0..1 exclusive.")


class SubwordLanguageModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config if config is not None else SubwordConfig()
        c = self.config
        # same overall structure but a bigger vocabulary and wider representations
        self.token_embedding = nn.Embedding(c.vocab_size, c.embedding_dim)
        self.position_embedding = nn.Embedding(c.context_length, c.embedding_dim)
        self.embedding_dropout = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(c) for _ in range(c.num_layers)])
        self.final_norm = nn.LayerNorm(c.embedding_dim)
        self.output_head = nn.Linear(c.embedding_dim, c.vocab_size, bias=False)
        self.register_buffer("causal_mask", torch.triu(torch.ones(c.context_length, c.context_length, dtype=torch.bool), diagonal=1), persistent=False)
        # small random starting weights help keep the first activations under control
        # norm scales keep their usual starting values and no trained weights get loaded
        self.apply(self._initialize)
        for block in self.blocks:
            nn.init.normal_(block.attention.in_proj_weight, mean=0.0, std=0.02)
            nn.init.zeros_(block.attention.in_proj_bias)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids, targets=None):
        if input_ids.ndim != 2 or input_ids.shape[0] < 1 or not 1 <= input_ids.shape[1] <= self.config.context_length:
            raise ValueError("Need a nonempty batch within the token context window.")
        if targets is not None and targets.shape != input_ids.shape:
            raise ValueError("Targets must match the input shape.")
        length = input_ids.shape[1]
        positions = torch.arange(length, device=input_ids.device)
        # combine what each piece is with where it appears in the current window
        x = self.embedding_dropout(self.token_embedding(input_ids) + self.position_embedding(positions))
        for block in self.blocks:
            x = block(x, self.causal_mask[:length, :length])
        logits = self.output_head(self.final_norm(x))
        # logits are raw scores for every possible next piece at every position
        loss = None
        if targets is not None:
            # ignored targets let us skip visitor text and padding during reply training
            loss = F.cross_entropy(logits.reshape(-1, self.config.vocab_size), targets.reshape(-1), ignore_index=-100)
        return logits, loss
