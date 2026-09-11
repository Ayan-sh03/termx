# termx

A terminal-native AI coding assistant written in Rust. termx talks to any OpenAI-compatible LLM endpoint and can read, write and edit files, search the filesystem, and run shell commands through an approval flow.

## Features

- OpenAI-compatible LLM client (bring your own key via `OPENAI_API_KEY` / base URL)
- Built-in tools: read / write / edit / insert file, list dir, search in file, run shell, ask oracle
- Approval step before potentially destructive actions
- Session management with streaming responses
- Async runtime (tokio) with a mockable LLM client for tests

## Requirements

- Rust (edition 2024)
- An OpenAI-compatible API key

## Run

```bash
export OPENAI_API_KEY=sk-...
cargo run
```

## Test

```bash
cargo test
```

## License

MIT