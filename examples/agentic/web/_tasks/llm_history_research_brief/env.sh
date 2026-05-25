# shellcheck shell=bash
# Per-task env overrides sourced by evaluate.sh before running this task.
# This task uses browser-use's StructuredOutputAction path (instead of the
# default free-text DoneAction). agent.py registers a `ResearchBrief`
# Pydantic model with a REQUIRED `common_themes: str` field; Pydantic
# validation forces the model to fill it -- bypassing DoneAction.text's
# "only report observed data" guidance that otherwise blocks the
# synthesis paragraph. See agent.py's _OUTPUT_MODELS.
export BROWSER_USE_OUTPUT_MODEL=research_brief

# 40 hops: 3 article navigations + scrolling + the structured-output
# emission step. Default 20 is too tight.
export BROWSER_USE_MAX_STEPS=40
