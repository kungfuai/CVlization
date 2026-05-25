Open https://en.wikipedia.org/wiki/Python_(programming_language). On the right side of the article there is an infobox titled 'Python' that lists structured facts about the language. From that infobox, extract these three fields:

- "first_appeared": the year Python first appeared, as an integer.
- "paradigm": the programming paradigms shown in the infobox; return as a single comma-separated string.
- "typing_discipline": the typing discipline shown in the infobox; return as a single comma-separated string.

Reply with ONLY a single-line JSON object containing exactly those three keys. No commentary, no markdown fences, no triple-backticks, no explanation. Exactly one line, valid JSON.

Example output format (your values will differ): {"first_appeared": 1991, "paradigm": "multi-paradigm: object-oriented, ...", "typing_discipline": "dynamic, strong, ..."}
