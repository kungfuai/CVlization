## POS-Augmented Shakespeare Utilities

Use the updated preprocess script to generate the optional POS-tagged token stream:

```
uv run python data/shakespeare_char/prepare.py --with-pos
```

Peek at the interleaved POS and character tokens without altering spacing:

```
uv run python data/shakespeare_char/peek_pos_augmented.py --start 0 --length 120
```

Run the `prepare.py` command without `--with-pos` to regenerate the classic char-only binaries.
