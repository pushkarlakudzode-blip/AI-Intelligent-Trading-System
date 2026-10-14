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

PAGES_TO_TEST = [
    ("2_market_analysis", "2. Market Analysis"),
    ("3_predictive_models", "3. Predictive Models"),
    ("4_news_nlp", "4. News & NLP"),
    ("5_signal_fusion", "5. Signal Fusion"),
    ("8_real_time_data", "8. Real-Time Data"),
]

async def capture_all_pages():
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
        ws_url = page_targets[0]["webSocketDebuggerUrl"]
        
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

            async def eval_js(expr):
                res = await send_cmd("Runtime.evaluate", {"expression": expr, "returnByValue": True})
                return res.get("result", {}).get("value")

            await send_cmd("Page.enable")
            await send_cmd("Runtime.enable")
            await send_cmd("Emulation.setDeviceMetricsOverride", {
                "width": 1920,
                "height": 1080,
                "deviceScaleFactor": 1,
                "mobile": False
            })

            print("Navigating to http://localhost:8501...", flush=True)
            await send_cmd("Page.navigate", {"url": "http://localhost:8501"})
            await asyncio.sleep(8.0)

            for file_slug, page_label in PAGES_TO_TEST:
                print(f"Switching to page: {page_label}...", flush=True)
                js_click = f"""
                (() => {{
                    const labels = Array.from(document.querySelectorAll('label'));
                    const target = labels.find(l => l.innerText.includes('{page_label}'));
                    if (target) {{
                        target.click();
                        return true;
                    }}
                    return false;
                }})()
                """
                clicked = await eval_js(js_click)
                print(f"Clicked {page_label}: {clicked}", flush=True)
                await asyncio.sleep(4.0)

                res = await send_cmd("Page.captureScreenshot", {"format": "png"})
                png_bytes = base64.b64decode(res["data"])
                fname = f"dashboard_{file_slug}.png"
                Path(fname).write_bytes(png_bytes)
                print(f"Saved {fname} ({len(png_bytes)} bytes)", flush=True)

            listener_task.cancel()

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    asyncio.run(capture_all_pages())
