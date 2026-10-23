"""tiny byte model for radahn

text turns into bytes then embeddings then next byte scores
no downloads or training happen in this file
another script has to actually make the model first
"""

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F


# settings go here so i can mess with them easy
@dataclass(frozen=True)
class ModelConfig:
    context_length: int = 512  # how many bytes it can look at
    embedding_dim: int = 128  # how many numbers represent each spot
    num_heads: int = 4        # splits the 128 numbers into 4 groups of 32
    num_layers: int = 4       # how many blocks the text goes through
    dropout: float = 0.1      # drops some values while training so it memorizes less

    def __post_init__(self) -> None:
        if min(self.context_length, self.embedding_dim, self.num_heads, self.num_layers) < 1:
            raise ValueError("Context, width, heads, and layers must be positive.")
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError("embedding_dim must divide evenly across num_heads.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be between 0 (inclusive) and 1 (exclusive).")


# using bytes so i dont have to train a whole tokenizer
# every byte already has a number even for names it hasnt seen
class ByteTokenizer:
    vocab_size = 256

    @staticmethod
    def encode(text: str) -> list[int]:
        return list(text.encode("utf-8"))

    @staticmethod
    def decode(token_ids: Iterable[int]) -> str:
        # random bytes might not make valid text yet
        # shows a replacement character instead of crashing but doesnt fix the guess
        return bytes(token_ids).decode("utf-8", errors="replace")


# this works on each spot after attention mixes the context
class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            # make it wider for a bit so theres more room to work
            nn.Linear(config.embedding_dim, 4 * config.embedding_dim),
            # need a nonlinear step or the linear layers could act like just one
            # gelu does that part
            nn.GELU(),
            # shrink it back so i can add it to the input again
            nn.Linear(4 * config.embedding_dim, config.embedding_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# each block shares context then works on each spot
class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        # normalize first to help learning stay stable
        self.attention_norm = nn.LayerNorm(config.embedding_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=config.embedding_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,  # shape goes batch then position then features
        )
        self.attention_dropout = nn.Dropout(config.dropout)
        self.feed_forward_norm = nn.LayerNorm(config.embedding_dim)
        self.feed_forward = FeedForward(config)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        normalized = self.attention_norm(x)
        # queries keys and values all come from the same text
        # mask blocks future bytes so it cant peek at the answers
        attended, _ = self.attention(
            normalized, normalized, normalized,
            attn_mask=causal_mask,
            need_weights=False,
        )
        # add the input back so info and gradients have a direct way through
        x = x + self.attention_dropout(attended)
        x = x + self.feed_forward(self.feed_forward_norm(x))
        return x


# put everything together and guess the next byte
class RadahnLanguageModel(nn.Module):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config if config is not None else ModelConfig()
        config = self.config

        # one lookup for the byte and another for where it is
        # word order matters so the model needs both
        self.byte_embedding = nn.Embedding(ByteTokenizer.vocab_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(config.context_length, config.embedding_dim)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )
        self.final_norm = nn.LayerNorm(config.embedding_dim)
        # one score for each possible next byte called logits
        self.output_head = nn.Linear(config.embedding_dim, ByteTokenizer.vocab_size, bias=False)

        # true in this mask means dont look here
        # each spot sees itself and earlier bytes but nothing after it
        mask = torch.triu(
            torch.ones(config.context_length, config.context_length, dtype=torch.bool),
            diagonal=1,
        )
        # buffer moves to the gpu with the model but doesnt get learned
        # can make this mask again later so no need to save it with the weights
        self.register_buffer("causal_mask", mask, persistent=False)

        # pytorch sets up the layers without loading any pretrained weights
        # some stuff starts at fixed values like biases and norm scales
        # still counts as starting from scratch

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """gives back next byte scores and loss if targets are passed in

        input shape is batch size then sequence length
        targets have the same shape but shifted one byte ahead already
        hell as input and ello as target teaches the next byte at each spot
        scores have batch size then sequence length then 256 byte choices
        loss is one number or none when theres no targets
        inputs and targets need to be on the same device as the model
        """
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch_size, sequence_length].")
        batch_size, sequence_length = input_ids.shape
        if batch_size < 1 or not 1 <= sequence_length <= self.config.context_length:
            raise ValueError(
                f"Use a nonempty batch and 1..{self.config.context_length} positions."
            )
        if targets is not None and targets.shape != input_ids.shape:
            raise ValueError("targets must match input_ids and be shifted one byte ahead.")

        positions = torch.arange(sequence_length, device=input_ids.device)
        x = self.byte_embedding(input_ids) + self.position_embedding(positions)
        x = self.embedding_dropout(x)
        mask = self.causal_mask[:sequence_length, :sequence_length]
        for block in self.blocks:
            x = block(x, mask)
        logits = self.output_head(self.final_norm(x))

        loss = None
        if targets is not None:
            # loss checks if the real next byte got a good score
            # cross entropy already handles log softmax so dont do it twice
            # ignored labels skip visitor text and padding during answer focused training
            loss = F.cross_entropy(
                logits.reshape(-1, ByteTokenizer.vocab_size),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss


# no training or generation loop here yet
# defining the model isnt the same as making it or training it
