# Vers Python SDK (`vers-sdk`)

Official Python client for the [Vers](https://vers.sh) `/api/v2` REST API, providing both synchronous (`VersClient`) and asynchronous (`AsyncVersClient`) high-level clients with automatic `Idempotency-Key` handling, retry reuse, NDJSON operation streaming with cursor resumption and polling fallback, and binary file transfer.

## Installation

```sh
pip install vers-sdk
```

## Quick Start (Sync)

```python
from vers_sdk import VersClient

# Reads VERS_API_KEY and optional VERS_BASE_URL (default: https://api.vers.sh)
with VersClient() as client:
    # Create a computer and wait for the durable operation to succeed
    computer = client.create(vcpu_count=2, mem_size_mib=1024)
    cid = computer["computer_id"]

    try:
        # Execute a command (direct argv list — shell strings are rejected)
        result = client.exec(cid, ["uname", "-a"])
        print(result["stdout"])

        # Stream stdout/stderr events over NDJSON
        for event in client.exec_stream(cid, ["sh", "-lc", "echo hello; echo err >&2"]):
            print(event)

        # Write and read files (binary or text)
        client.write_text(cid, "/tmp/hello.txt", "hello from vers\n", create_dirs=True)
        print(client.read_text(cid, "/tmp/hello.txt"))

        client.write_file(cid, "/tmp/data.bin", b"\x00\x01\x02\x03")
        raw = client.read_file(cid, "/tmp/data.bin")

        # Branch a running computer synchronously (201 Created)
        branched = client.branch(cid, count=1)
        branch_id = branched["computers"][0]["computer_id"]
        client.delete(branch_id)

        # Create a checkpoint (wait_for="durable" by default, or "local" for fast same-node restore)
        ckpt_op = client.checkpoint(cid, name="before-sleep", wait_for="durable")
        checkpoint_id = ckpt_op["result"]["id"]

        # Restore a new computer from a checkpoint
        restored = client.restore(checkpoint_id)
        client.delete(restored["computer_id"])

        # Sleep and wake
        client.sleep(cid)
        woken = client.wake(cid)
        print("Woken IP:", woken["ip"])
    finally:
        client.delete(cid)
```

## Quick Start (Async)

```python
import asyncio
from vers_sdk import AsyncVersClient

async def main() -> None:
    async with AsyncVersClient() as client:
        computer = await client.create(vcpu_count=2, mem_size_mib=1024)
        cid = computer["computer_id"]
        try:
            res = await client.exec(cid, ["echo", "hello async"])
            print(res["stdout"])

            async for event in client.exec_stream(cid, ["sh", "-lc", "echo streaming"]):
                print(event)

            computers = await client.list(limit=20)
            print("Total in page:", len(computers["computers"]))
        finally:
            await client.delete(cid)

asyncio.run(main())
```

## Configuration

```python
from vers_sdk import VersClient, AsyncVersClient

client = VersClient(
    api_key="your-api-key",          # or set VERS_API_KEY
    base_url="https://api.vers.sh",  # or set VERS_BASE_URL
    max_retries=2,                   # retries 408, 429, 5xx, and transport errors
    timeout=30.0,                    # per-request HTTP timeout (seconds)
    operation_timeout=300.0,         # default wait_for_operation timeout (seconds)
    http_client=custom_httpx_client, # optional custom httpx.Client / httpx.AsyncClient
)
```

## Durable Operations & Idempotency

Mutating `/api/v2` lifecycle calls (`create`, `restore`, `checkpoint`, `delete`) automatically generate a UUIDv4 `Idempotency-Key` header once per call and reuse the exact same key across all retries. You may also supply your own `idempotency_key="..."`.

Pass `wait=False` to return the initial `202 Accepted` `Operation` immediately and follow it yourself with `wait_for_operation`:

```python
op = client.create(vcpu_count=1, mem_size_mib=512, wait=False)
completed = client.wait_for_operation(op["operation_id"], timeout=120.0)
computer = client.get(completed["resource"]["id"])
```

`wait_for_operation` streams `GET /api/v2/operations/{id}/events?follow=true` (`application/x-ndjson`), resumes from the last witnessed cursor across disconnects and bounded follow windows, handles `gap` events, and automatically falls back to polling `GET /api/v2/operations/{id}` when streaming is unavailable.

## Error Handling

```python
from vers_sdk import (
    APIError,
    NotFoundError,
    RateLimitError,
    VersOperationError,
)

try:
    computer = client.create(vcpu_count=2, mem_size_mib=1024)
except VersOperationError as exc:
    print(
        f"Operation {exc.operation_id} {exc.status} at stage={exc.stage}: "
        f"[{exc.code}] {exc.message} (cleanup={exc.cleanup_operation_id})"
    )
except NotFoundError as exc:
    print(f"Not found: {exc.status} {exc.error_message}")
except RateLimitError as exc:
    print(f"Rate limited: {exc.status}")
except APIError as exc:
    print(f"API error: {exc.status} {exc.error_message}")
```

## Low-Level Client

The generated async `VersSdkClient` remains exported from `vers_sdk` for full low-level endpoint coverage and backwards compatibility.

## Development & Testing

```sh
# Run offline unit tests
uv run --with pytest --with pytest-asyncio pytest

# Run live E2E smoke test (requires VERS_API_KEY)
VERS_API_KEY=... uv run python scripts/smoke.py
```

## Requirements

- Python 3.10+
- `httpx>=0.24`

## License

MIT
