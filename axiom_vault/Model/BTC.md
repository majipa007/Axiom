
# Token and Positional Embeddings

## Tokenization

The tokenizer converts text into token IDs.

```text
Text → Tokens → Token IDs
````

Token IDs are only identifiers. Their numerical values do not contain meaning.

## Token Embeddings

`nn.Embedding` is a trainable lookup table.

```text
Token embedding table: [V, C]
```

Where:

* `V` = vocabulary size
* `C` = `d_model`
* `C` = number of values used to represent each token

Each token ID retrieves one vector containing `C` values.

```text
Token IDs:        [B, T]
Token embeddings: [B, T, C]
```

## Positional Embeddings

The position embedding table stores one learnable vector for every possible position.

```text
Position embedding table: [T_max, C]
```

Where:

* `T_max` = maximum context length
* `C` = `d_model`

Position indices are generated as:

```text
[0, 1, 2, ..., T-1]
```

Looking them up produces:

```text
Position embeddings: [T, C]
```

The same position vectors are used across every sequence in the batch.

## Combining the Embeddings

Token and position embeddings are added:

```text
final_embedding = token_embedding + position_embedding
```

Shape calculation:

```text
[B, T, C] + [T, C] → [B, T, C]
```

Conceptually:

```text
token identity + token position = input representation
```

Both embedding tables begin with random values and are learned during training.

## Complete Shape Flow

```text
Text
  ↓
Token IDs             [B, T]
  ↓
Token Embeddings      [B, T, C]
          +
Position Embeddings      [T, C]
  ↓
Combined Embeddings   [B, T, C]
```

## PyTorch Components

```python
self.token_embedding = nn.Embedding(vocab_size, d_model)
self.position_embedding = nn.Embedding(context_length, d_model)
```

`C` and `d_model` refer to the same value: the width of each token representation inside the Transformer.
