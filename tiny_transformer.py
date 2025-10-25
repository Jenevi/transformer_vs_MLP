#!/usr/bin/env python3
"""Minimal character-level transformer language model.

This script trains a tiny transformer on a small corpus to demonstrate how
language modeling works. Defaults are intentionally tiny so it can run on a
CPU in under a minute, but you can tweak hyperparameters via the CLI.
"""
from __future__ import annotations

import argparse
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_TEXT = textwrap.dedent(
    """
    Tiny transformer models can learn patterns in characters.
    This default corpus leaves plenty of room to experiment with
    hyperparameters, dataset sizes, and sampling temperature.
    """
).strip()


def read_corpus(path: Path | None) -> str:
    if path is None:
        return DEFAULT_TEXT
    data = path.read_text(encoding="utf-8")
    if not data.strip():
        raise ValueError(f"Corpus file {path} is empty")
    return data


class TinyTokenizer:
    def __init__(self, text: str):
        vocab = sorted(set(text))
        if len(vocab) < 2:
            raise ValueError("Corpus must contain at least two distinct characters")
        self.stoi = {ch: idx for idx, ch in enumerate(vocab)}
        self.itos = {idx: ch for ch, idx in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor([self.stoi[ch] for ch in text], dtype=torch.long)

    def decode(self, indices: Iterable[int]) -> str:
        return "".join(self.itos[int(idx)] for idx in indices)


@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int = 64
    n_embd: int = 64
    n_head: int = 4
    n_layer: int = 2
    dropout: float = 0.1


class FeedForward(nn.Module):
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        hidden = 4 * n_embd
        self.net = nn.Sequential(
            nn.Linear(n_embd, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float, block_size: int):
        super().__init__()
        assert n_embd % n_head == 0, "Embedding dimension must be divisible by number of heads"
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.out = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * self.scale
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.out(y))
        return y


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadSelfAttention(config.n_embd, config.n_head, config.dropout, config.block_size)
        self.ff = FeedForward(config.n_embd, config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(f"Cannot forward sequence of length {T}, block_size is {self.config.block_size}")

        positions = torch.arange(0, T, device=idx.device)
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(positions)[None, :, :]
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            logits_flat = logits.view(B * T, -1)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_idx], dim=1)
        return idx


@dataclass
class TrainConfig:
    batch_size: int = 16
    block_size: int = 64
    max_iters: int = 800
    eval_interval: int = 100
    lr: float = 3e-4
    grad_clip: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def get_batch(data: torch.Tensor, config: TrainConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    if data.size(0) <= config.block_size + 1:
        repeats = math.ceil((config.block_size + 2) / data.size(0))
        data = data.repeat(repeats)

    max_start = data.size(0) - config.block_size - 1
    ix = torch.randint(0, max_start + 1, (config.batch_size,))
    x = torch.stack([data[i : i + config.block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + config.block_size] for i in ix])
    device = torch.device(config.device)
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model: TransformerLM, train_data: torch.Tensor, val_data: torch.Tensor, config: TrainConfig, eval_batches: int = 20) -> Tuple[float, float]:
    model.eval()
    losses = {"train": 0.0, "val": 0.0}
    for split, data in ("train", train_data), ("val", val_data):
        total = 0.0
        for _ in range(eval_batches):
            x, y = get_batch(data, config)
            _, loss = model(x, y)
            total += float(loss)
        losses[split] = total / eval_batches
    model.train()
    return losses["train"], losses["val"]


def train(corpus: str, args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)

    tokenizer = TinyTokenizer(corpus)
    encoded = tokenizer.encode(corpus)
    min_tokens = args.block_size + 2
    if encoded.size(0) < min_tokens:
        repeats = math.ceil(min_tokens / encoded.size(0))
        encoded = encoded.repeat(repeats)
    split_idx = int(0.9 * encoded.size(0))
    train_data = encoded[:split_idx]
    val_data = encoded[split_idx:]
    if val_data.numel() < args.block_size + 1:
        # fall back to repeating the corpus for the validation set if it's tiny
        val_data = encoded.clone()

    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        block_size=args.block_size,
        max_iters=args.max_iters,
        eval_interval=args.eval_interval,
        lr=args.lr,
        grad_clip=args.grad_clip,
        device=args.device,
    )

    model_cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
    )

    model = TransformerLM(model_cfg).to(train_cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr)

    for step in range(train_cfg.max_iters + 1):
        if step % train_cfg.eval_interval == 0:
            train_loss, val_loss = estimate_loss(model, train_data, val_data, train_cfg)
            print(f"step {step:4d} | train loss {train_loss:.3f} | val loss {val_loss:.3f}")

        xb, yb = get_batch(train_data, train_cfg)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()

    prompt = torch.tensor([[tokenizer.stoi[next(iter(tokenizer.stoi))]]], device=train_cfg.device)
    generated = model.generate(prompt, max_new_tokens=args.sample_tokens, temperature=args.temperature)
    text = tokenizer.decode(generated[0].tolist())
    print("\n=== Sample ===")
    print(text)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a tiny Transformer language model")
    parser.add_argument("--text-path", type=Path, default=None, help="Optional path to a custom training corpus")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--max-iters", type=int, default=800)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--n-embd", type=int, default=64)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--sample-tokens", type=int, default=200, help="Number of new tokens to sample after training")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", choices=["cpu", "cuda"])
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    corpus = read_corpus(args.text_path)
    print(f"Corpus length: {len(corpus)} characters | Vocab size: {len(set(corpus))}")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"

    train(corpus, args)


if __name__ == "__main__":
    main()
