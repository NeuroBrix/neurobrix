"""The OpenAI-compatible HTTP door of the serving daemon (docs/serving.md, "OpenAI-compatible HTTP").

Engine policy over the daemon's one inference worker: the HTTP door parses a request, maps it to the loaded family's
own flow, and writes the answer in the OpenAI shapes. stdlib only.
"""
