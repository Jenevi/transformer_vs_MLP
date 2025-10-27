"""Tests for the tiny transformer components."""

import torch
import pytest

import tiny_transformer as tt


def test_read_corpus_from_file(tmp_path):
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text("abc", encoding="utf-8")
    assert tt.read_corpus(corpus_path) == "abc"


def test_read_corpus_rejects_empty_file(tmp_path):
    corpus_path = tmp_path / "empty.txt"
    corpus_path.write_text(" \n\t", encoding="utf-8")
    with pytest.raises(ValueError):
        tt.read_corpus(corpus_path)


def test_tokenizer_roundtrip():
    text = "hello world"
    tokenizer = tt.TinyTokenizer(text)
    encoded = tokenizer.encode(text)
    assert encoded.dtype == torch.long
    decoded = tokenizer.decode(encoded.tolist())
    assert decoded == text
    assert tokenizer.vocab_size == len(set(text))


def test_tokenizer_requires_distinct_characters():
    with pytest.raises(ValueError):
        tt.TinyTokenizer("aaaaa")


def test_transformer_forward_shapes():
    config = tt.ModelConfig(
        vocab_size=6,
        block_size=8,
        n_embd=16,
        n_head=4,
        n_layer=1,
        dropout=0.0,
    )
    model = tt.TransformerLM(config)
    idx = torch.randint(0, config.vocab_size, (2, config.block_size))
    logits, loss = model(idx, idx)
    assert logits.shape == (2, config.block_size, config.vocab_size)
    assert loss is not None


def test_generate_appends_expected_length():
    config = tt.ModelConfig(
        vocab_size=5,
        block_size=6,
        n_embd=12,
        n_head=3,
        n_layer=1,
        dropout=0.0,
    )
    model = tt.TransformerLM(config)
    prompt = torch.randint(0, config.vocab_size, (1, 4))
    output = model.generate(prompt, max_new_tokens=3)
    assert output.shape == (1, prompt.shape[1] + 3)


def test_mlp_forward_shapes():
    config = tt.MLPConfig(
        vocab_size=6,
        block_size=8,
        n_embd=8,
        hidden_size=32,
        n_layers=2,
        dropout=0.0,
    )
    model = tt.MLPLanguageModel(config)
    idx = torch.randint(0, config.vocab_size, (2, config.block_size))
    logits, loss = model(idx, idx)
    assert logits.shape == (2, config.block_size, config.vocab_size)
    assert loss is not None


def test_mlp_generate_appends_expected_length():
    config = tt.MLPConfig(
        vocab_size=5,
        block_size=6,
        n_embd=8,
        hidden_size=16,
        n_layers=1,
        dropout=0.0,
    )
    model = tt.MLPLanguageModel(config)
    prompt = torch.randint(0, config.vocab_size, (1, 4))
    output = model.generate(prompt, max_new_tokens=3)
    assert output.shape == (1, prompt.shape[1] + 3)
