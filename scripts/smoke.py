#!/usr/bin/env python3
"""Live end-to-end smoke test for VersClient against the Vers /api/v2 API.

Gated on the VERS_API_KEY environment variable. Always cleans up every computer
created during the run in a try/finally block.

Usage:
    VERS_API_KEY=... uv run python scripts/smoke.py
"""

from __future__ import annotations

import base64
import os
import sys

from vers_sdk import VersClient


def main() -> int:
    api_key = os.environ.get("VERS_API_KEY")
    if not api_key:
        print("VERS_API_KEY is not set; skipping live smoke test.")
        return 0

    base_url = os.environ.get("VERS_BASE_URL", "https://api.vers.sh")
    created_ids: list[str] = []

    with VersClient(api_key=api_key, base_url=base_url) as client:
        try:
            print(f"1. Creating computer on {base_url}...")
            comp = client.create(vcpu_count=1, mem_size_mib=512)
            cid = comp["computer_id"]
            created_ids.append(cid)
            print(f"   Created {cid} (state={comp.get('state')}, ip={comp.get('ip')})")

            print("2. Fetching computer via get() and list()...")
            fetched = client.get(cid)
            assert fetched["computer_id"] == cid
            listing = client.list(limit=20)
            assert any(c["computer_id"] == cid for c in listing.get("computers", []))

            print("3. Executing command via exec()...")
            exec_res = client.exec(cid, ["echo", "hello-vers-smoke"])
            assert exec_res["exit_code"] == 0, exec_res
            assert "hello-vers-smoke" in exec_res["stdout"], exec_res

            print("4. Streaming command output via exec_stream()...")
            chunks: list[bytes] = []
            exit_code: int | None = None
            for ev in client.exec_stream(cid, ["sh", "-lc", "echo streamed-ok"]):
                if ev.get("type") == "chunk" and ev.get("stream") == "stdout":
                    chunks.append(base64.b64decode(ev["data_b64"]))
                elif ev.get("type") == "exit":
                    exit_code = ev.get("exit_code")
            assert exit_code == 0, f"unexpected exit_code={exit_code}"
            assert b"streamed-ok" in b"".join(chunks)

            print("5. Writing and reading binary + text files...")
            binary_data = b"\x00\x01\x02vers-binary\xff\xfe"
            client.write_file(cid, "/tmp/smoke/test.bin", binary_data, create_dirs=True)
            assert client.read_file(cid, "/tmp/smoke/test.bin") == binary_data

            client.write_text(cid, "/tmp/smoke/test.txt", "hello text\n")
            assert client.read_text(cid, "/tmp/smoke/test.txt") == "hello text\n"

            print("6. Sleeping and waking computer...")
            client.sleep(cid)
            woken = client.wake(cid)
            assert woken.get("ip"), woken
            print(f"   Woke {cid} (ip={woken['ip']})")

            print("7. Branching computer...")
            branched = client.branch(cid, count=1)
            branch_cid = branched["computers"][0]["computer_id"]
            created_ids.append(branch_cid)
            print(f"   Branched -> {branch_cid}")

            print("8. Creating durable checkpoint and restoring...")
            ckpt_op = client.checkpoint(cid, name="python-sdk-smoke", wait_for="durable")
            checkpoint_id = ckpt_op["result"]["id"]
            print(f"   Checkpoint {checkpoint_id} durable")

            restored = client.restore(checkpoint_id)
            restored_cid = restored["computer_id"]
            created_ids.append(restored_cid)
            assert client.read_text(restored_cid, "/tmp/smoke/test.txt") == "hello text\n"
            print(f"   Restored {restored_cid} and verified file persistence")

            print("ALL SMOKE CHECKS PASSED.")
            return 0
        finally:
            for cid in reversed(created_ids):
                try:
                    print(f"Cleanup: deleting {cid}...")
                    client.delete(cid, wait=True)
                except Exception as exc:
                    print(f"Warning: failed to delete {cid}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
