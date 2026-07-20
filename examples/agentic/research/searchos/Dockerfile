# SearchOS — multi-agent open-domain information-seeking (deep research).
# Pure API-based LLM orchestration: no torch / CUDA, runs on CPU.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# git: pip installs SearchOS from a pinned commit. ca-certificates: TLS to the
# LLM / search endpoints.
RUN apt-get update && apt-get install -y --no-install-recommends \
      git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Pin SearchOS to a specific commit for reproducibility. The [tavily] extra adds
# the Tavily web-search client so both offline (local-corpus) and web (Serper /
# Tavily) modes work; Serper needs no extra (it uses core `requests`).
ARG SEARCHOS_REF=361373bfc7196697a9bcd4731dfc34880790f4aa
RUN pip install --no-cache-dir \
      "searchos[tavily] @ git+https://github.com/antins-labs/SearchOS.git@${SEARCHOS_REF}"

# Two surgical grounding patches (see patch_search_persona.py for the full
# rationale): (1) reword the search-agent persona so gpt-4.1-class models
# don't read "end with no tool call" as license to answer from prior
# knowledge without ever searching; (2) add an SF_ENABLE_SUMMARY_REPLAY env
# gate so hallucinated sub-agent summaries can't fill coverage cells as
# unanchored agent:// evidence. Exact-match patches: the build fails if the
# upstream text drifts, so they can't rot silently.
COPY patch_search_persona.py /tmp/patch_search_persona.py
RUN python /tmp/patch_search_persona.py && rm /tmp/patch_search_persona.py

WORKDIR /workspace
COPY run_search.py local_corpus_provider.py corpus.json /workspace/

# Default entrypoint runs one offline research session. Override args via `run.sh`.
ENTRYPOINT ["python", "/workspace/run_search.py"]
