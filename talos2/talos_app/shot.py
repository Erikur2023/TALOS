"""Capture d'ecran de l'IHM (Edge headless + protocole DevTools, temps reel) — pour la doc et les tests visuels.
  python -m talos_app.shot out.png [--wait 8] [--js "code a executer"] [--after 10]"""
import argparse, asyncio, base64, json, os, subprocess, time, urllib.request

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"


async def run(a):
    import websockets
    prof = os.path.join(os.environ.get("TEMP", "."), "talos_shot_profile")
    p = subprocess.Popen([EDGE, "--headless=new", "--remote-debugging-port=9333", "--use-angle=swiftshader",
                          "--enable-unsafe-swiftshader", "--window-size=%d,%d" % (a.w, a.h), "--user-data-dir=" + prof,
                          "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            try:
                tabs = json.load(urllib.request.urlopen("http://127.0.0.1:9333/json"))
                ws_url = [t for t in tabs if t["type"] == "page"][0]["webSocketDebuggerUrl"]
                break
            except Exception:
                time.sleep(0.2)
        async with websockets.connect(ws_url, max_size=2 ** 28) as ws:
            n = [0]

            async def call(method, **params):
                n[0] += 1
                await ws.send(json.dumps({"id": n[0], "method": method, "params": params}))
                while True:
                    m = json.loads(await ws.recv())
                    if m.get("id") == n[0]:
                        return m.get("result", {})
            await call("Emulation.setDeviceMetricsOverride", width=a.w, height=a.h, deviceScaleFactor=1, mobile=False)
            await call("Page.navigate", url=a.url)
            await asyncio.sleep(a.wait)
            if a.js:
                r = await call("Runtime.evaluate", expression=a.js, awaitPromise=True)
                print("js:", json.dumps(r.get("result", {}).get("value"))[:300])
            await asyncio.sleep(a.after)
            r = await call("Page.captureScreenshot", format="png")
            open(a.out, "wb").write(base64.b64decode(r["data"]))
            print("capture :", a.out)
    finally:
        p.terminate()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--url", default="http://127.0.0.1:8770/")
    ap.add_argument("--wait", type=float, default=8)
    ap.add_argument("--after", type=float, default=2)
    ap.add_argument("--js", default="")
    ap.add_argument("--w", type=int, default=1680)
    ap.add_argument("--h", type=int, default=1000)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
