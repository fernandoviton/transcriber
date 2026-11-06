# Agent rules

## Private sessions: do not read

Never open, read, search, list the contents of, copy, or summarize anything inside
`sessions/` folders whose name matches `*day*`.
They contain private recordings. This covers every tool: file viewers, grep/search,
shell commands (Get-Content, Select-String, cat, type), scripts, and sub-agents.

If a task seems to need one of these folders, stop and ask the user. Use the test sessions
(`*-test*`) for debugging instead.

Never commit recordings, transcripts, session exports, local profiles, or `.env` files.
Use synthetic examples when updating documentation or testing the pipeline.
