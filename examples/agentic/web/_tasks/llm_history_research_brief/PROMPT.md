Research the history of three foundational large language models by visiting these Wikipedia articles, in this order:

1. https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)
2. https://en.wikipedia.org/wiki/BERT_(language_model)
3. https://en.wikipedia.org/wiki/GPT-3

For each article, find: the year the model was introduced, the organization that introduced it, and (where applicable) the parameter count.

Then produce a `ResearchBrief` structured output containing:

- `title`: a one-line title for the brief (e.g. "The Foundations of Modern LLMs (2017-2020)").
- `introduction`: a 1-2 sentence opening paragraph framing the topic.
- `table_rows`: one row per model in the order you visited them (Transformer / BERT / GPT-3), each with `model`, `year`, `org`, and `parameters` (use "N/A" for missing values).
- `common_themes`: a 2-3 sentence synthesis paragraph (at least 50 characters) drawing conclusions about what the three models share — architectures, scaling trends, impact. This is a REQUIRED synthesis based on the facts you just observed across all three sources; it is not "filling gaps with training knowledge," it IS the conclusion-drawing the task explicitly demands.

The output schema enforces every field; you cannot omit `common_themes`.
