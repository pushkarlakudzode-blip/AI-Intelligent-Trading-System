import asyncio
import base64
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import urllib.request
import websockets

async def capture_dashboard_viewports():
    temp_dir = tempfile.mkdtemp()
    proc = subprocess.Popen([
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        "--headless=new",
        "--remote-debugging-port=9222",
        f"--user-data-dir={temp_dir}",
        "--disable-gpu",
        "--no-first-run",
    ])
    
    try:
        # Wait for CDP to be ready
        for _ in range(20):
            try:
                with urllib.request.urlopen("http://localhost:9222/json/version", timeout=1) as resp:
                    if resp.status == 200:
                        break
            except Exception:
                time.sleep(0.3)
        
        with urllib.request.urlopen("http://localhost:9222/json/list", timeout=3) as resp:
            targets = json.loads(resp.read().decode())
        
        page_targets = [t for t in targets if t.get("type") == "page"]
        if not page_targets:
            raise RuntimeError("No page target found in Chrome")
        ws_url = page_targets[0]["webSocketDebuggerUrl"]
        print(f"Connecting to target page: {page_targets[0].get('title')} ({ws_url})", flush=True)
        
        async with websockets.connect(ws_url, max_size=50 * 1024 * 1024) as ws:
            pending_responses = {}
            loop = asyncio.get_running_loop()

            async def listen_loop():
                try:
                    async for raw in ws:
                        msg = json.loads(raw)
                        mid = msg.get("id")
                        if mid and mid in pending_responses:
                            fut = pending_responses[mid]
                            if not fut.done():
                                fut.set_result(msg.get("result", {}))
                except asyncio.CancelledError:
                    pass

            listener_task = asyncio.create_task(listen_loop())
            msg_id = 0

            async def send_cmd(method, params=None):
                nonlocal msg_id
                msg_id += 1
                cid = msg_id
                fut = loop.create_future()
                pending_responses[cid] = fut
                await ws.send(json.dumps({"id": cid, "method": method, "params": params or {}}))
                res = await asyncio.wait_for(fut, timeout=15)
                pending_responses.pop(cid, None)
                return res

            print("Enabling Page domain...", flush=True)
            await send_cmd("Page.enable")

            # Set viewport 1920x1080 initially
            await send_cmd("Emulation.setDeviceMetricsOverride", {
                "width": 1920,
                "height": 1080,
                "deviceScaleFactor": 1,
                "mobile": False
            })

            print("Navigating to http://localhost:8501...", flush=True)
            await send_cmd("Page.navigate", {"url": "http://localhost:8501"})

            print("Waiting 10 seconds for Streamlit WebSocket hydration & chart rendering...", flush=True)
            await asyncio.sleep(10.0)

            # Viewport 1: 1920 x 1080
            print("Capturing 1920x1080...", flush=True)
            res_1080 = await send_cmd("Page.captureScreenshot", {"format": "png"})
            png_bytes = base64.b64decode(res_1080["data"])
            Path("dashboard_1920x1080.png").write_bytes(png_bytes)
            print(f"Saved dashboard_1920x1080.png ({len(png_bytes)} bytes)", flush=True)

            # Viewport 2: 1366 x 768
            print("Capturing 1366x768...", flush=True)
            await send_cmd("Emulation.setDeviceMetricsOverride", {
                "width": 1366,
                "height": 768,
                "deviceScaleFactor": 1,
                "mobile": False
            })
            await asyncio.sleep(2.0)
            res_768 = await send_cmd("Page.captureScreenshot", {"format": "png"})
            png_bytes_768 = base64.b64decode(res_768["data"])
            Path("dashboard_1366x768.png").write_bytes(png_bytes_768)
            print(f"Saved dashboard_1366x768.png ({len(png_bytes_768)} bytes)", flush=True)

            listener_task.cancel()

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    asyncio.run(capture_dashboard_viewports())
