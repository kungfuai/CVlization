"""Prepare the Shakespeare dataset for character-level language modeling."""

import argparse
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import requests


def download_dataset(input_path: str) -> str:
    if not os.path.exists(input_path):
        data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        with open(input_path, "w") as f:
            f.write(requests.get(data_url).text)
    with open(input_path, "r") as f:
        return f.read()


def build_vocab(data: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    chars = sorted(list(set(data)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos


def encode(text: str, stoi: Dict[str, int]) -> List[int]:
    return [stoi[c] for c in text]


def load_spacy(model: str):
    try:
        import spacy
    except ImportError as exc:  # pragma: no cover - informative error for users
        raise ImportError(
            "spaCy is required for POS augmentation. Install it with `pip install spacy` "
            "and download a model, e.g. `python -m spacy download en_core_web_sm`."
        ) from exc

    try:
        return spacy.load(model)
    except OSError as exc:  # pragma: no cover
        raise OSError(
            f"spaCy model '{model}' is not installed. Run `python -m spacy download {model}`"
        ) from exc


def coarse_pos(tag: str) -> str:
    mapping = {
        "ADJ": "ADJ",
        "ADV": "ADV",
        "INTJ": "INTJ",
        "NOUN": "NOUN",
        "PROPN": "NOUN",
        "PRON": "PRON",
        "VERB": "VERB",
        "AUX": "VERB",
        "ADP": "ADP",
        "CCONJ": "CONJ",
        "SCONJ": "CONJ",
        "DET": "DET",
        "NUM": "NUM",
        "PART": "PART",
        "PUNCT": "PUNCT",
        "SYM": "SYM",
    }
    return mapping.get(tag, "OTHER")


def augment_with_pos(text: str, stoi: Dict[str, int], pos_offset: int, nlp, pos_vocab: List[str]) -> List[int]:
    pos_to_id = {pos: idx for idx, pos in enumerate(pos_vocab)}
    doc = nlp(text)
    augmented_ids: List[int] = []
    token_iter = iter(doc)
    current_token = next(token_iter, None)
    while current_token is not None and coarse_pos(current_token.pos_) not in pos_to_id:
        current_token = next(token_iter, None)
    i = 0
    length = len(text)
    while i < length:
        if current_token is not None and i == current_token.idx:
            pos_id = pos_to_id[coarse_pos(current_token.pos_)]
            augmented_ids.append(pos_offset + pos_id)
            current_token = next(token_iter, None)
            while current_token is not None and coarse_pos(current_token.pos_) not in pos_to_id:
                current_token = next(token_iter, None)
        augmented_ids.append(stoi[text[i]])
        i += 1
    return augmented_ids


def main():
    parser = argparse.ArgumentParser(description="Prepare Tiny Shakespeare dataset")
    parser.add_argument(
        "--with-pos",
        action="store_true",
        help="Augment character stream with POS program tokens using spaCy.",
    )
    parser.add_argument(
        "--spacy-model",
        default="en_core_web_sm",
        help="spaCy model to use when --with-pos is enabled.",
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(__file__)
    input_file_path = os.path.join(base_dir, "input.txt")
    data = download_dataset(input_file_path)
    print(f"length of dataset in characters: {len(data):,}")

    stoi, itos = build_vocab(data)
    vocab_size = len(stoi)
    print("all the unique characters:", "".join(itos[i] for i in range(vocab_size)))
    print(f"vocab size: {vocab_size:,}")

    n = len(data)
    split_ix = int(n * 0.9)
    train_data = data[:split_ix]
    val_data = data[split_ix:]

    train_ids = np.array(encode(train_data, stoi), dtype=np.uint16)
    val_ids = np.array(encode(val_data, stoi), dtype=np.uint16)
    train_ids.tofile(os.path.join(base_dir, "train.bin"))
    val_ids.tofile(os.path.join(base_dir, "val.bin"))
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    meta = {
        "vocab_size": vocab_size,
        "itos": itos,
        "stoi": stoi,
    }
    with open(os.path.join(base_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    if not args.with_pos:
        return

    pos_vocab = ["NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "NUM", "CONJ", "ADP", "PART", "PUNCT", "INTJ", "SYM", "OTHER"]
    pos_offset = vocab_size
    nil_id = pos_offset + len(pos_vocab)

    print(f"Augmenting with POS tokens using spaCy model '{args.spacy_model}'")
    nlp = load_spacy(args.spacy_model)
    nlp.max_length = max(len(train_data), len(val_data), getattr(nlp, "max_length", 0)) + 100

    train_aug = augment_with_pos(train_data, stoi, pos_offset, nlp, pos_vocab)
    val_aug = augment_with_pos(val_data, stoi, pos_offset, nlp, pos_vocab)

    train_aug = np.array(train_aug, dtype=np.uint32)
    val_aug = np.array(val_aug, dtype=np.uint32)

    train_aug.tofile(os.path.join(base_dir, "train_with_pos.bin"))
    val_aug.tofile(os.path.join(base_dir, "val_with_pos.bin"))
    print(
        "wrote augmented binaries: train_with_pos.bin ({} tokens), val_with_pos.bin ({} tokens)".format(
            len(train_aug), len(val_aug)
        )
    )

    meta_pos = {
        "program_offset": pos_offset,
        "program_nil_id": nil_id,
        "program_pos_vocab": pos_vocab,
        "spacy_model": args.spacy_model,
    }
    with open(os.path.join(base_dir, "meta_pos.pkl"), "wb") as f:
        pickle.dump(meta_pos, f)


if __name__ == "__main__":
    main()
