#!/bin/bash
# Centralized prompt templates for CheckboxQA benchmark
#
# To switch prompts, change the ACTIVE_PROMPT variable below.
# All adapters source this file to ensure consistent prompting across models.

# General document understanding prompt
# Best overall performance: phi_4 (47.5%), qwen3_vl_2b (45%), moondream3 (45%)
PROMPT_GENERAL="Look carefully at this form/document image and answer concisely (Yes/No or brief text). "

# CheckboxQA paper prompt (Python list format)
# From: https://arxiv.org/abs/2304.12666
# Instructs model to return answers as Python list
PROMPT_PAPER='Answer the question. Do not write a full sentence. Provide a value as a Python list. If there is a single answer, the output should be a one-element list like ["ANSWER"]. If there are multiple valid answers, the list will have several elements, e.g., ["ANSWER 1", "ANSWER 2"]. Question: '

# Checkbox-specific prompt (explicit checkbox instructions)
# Tested worse than general prompt - made some models less accurate
PROMPT_CHECKBOX="This is a form with checkboxes. For the question below, find which checkbox is marked (✓) or checked and answer accordingly. "

# Active prompt - CHANGE THIS LINE to switch prompts globally
ACTIVE_PROMPT="$PROMPT_PAPER"

# Export for use in adapters
export ACTIVE_PROMPT
