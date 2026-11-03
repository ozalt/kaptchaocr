# MCP protocol

The server can expose its tools over the Model Context Protocol so AI agents
can call it directly.

- Protocol version: `2025-03-26`
- Endpoint: `POST /mcp`
- Methods: `initialize`, `tools/list`, `tools/call`
- Server name: `kaptchaocr`

Start with `--mcp` to add MCP alongside REST, or `--only-mcp` to serve only
MCP:

```sh
kaptchaocr --address 0.0.0.0:8000 --ocr --det --slide --mcp
```

Only tools backed by enabled features are advertised. With `--ocr --det
--slide`, the available tools are:

| Tool | Purpose |
| --- | --- |
| `ocr` | OCR recognition (`image`, `color_filter`, `charset_range`) |
| `det` | Bounding boxes (`image`) |
| `slide_match` | Template slider matching (`target_image`, `background_image`, `simple_target`) |
| `slide_comparison` | Gap comparison (`target_image`, `background_image`) |

## Initialize

```json
{
  "jsonrpc": "2.0",
  "id": 0,
  "method": "initialize",
  "params": {
    "protocolVersion": "2025-03-26",
    "capabilities": {},
    "clientInfo": {"name": "client-name", "version": "1.0.0"}
  }
}
```

## List tools

```json
{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
```

## Call a tool

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "ocr",
    "arguments": {
      "image": "base64_encoded_image",
      "color_filter": "green"
    }
  }
}
```

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "content": [
      {"type": "text", "text": "{\"probability\":null,\"text\":\"8A62N1\"}"}
    ],
    "isError": false
  }
}
```

Detection call:

```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "tools/call",
  "params": {
    "name": "det",
    "arguments": {"image": "base64_encoded_image"}
  }
}
```

```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "result": {
    "content": [
      {"type": "text", "text": "{\"bboxes\":[[80,3,98,21],[56,6,76,25]]}"}
    ],
    "isError": false
  }
}
```

The same API key / rate-limit settings apply as for REST when the server is
started with `--api-key` / `--rate-limit`.
