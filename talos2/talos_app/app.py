"""TALOS C2 — application de bureau.
  python -m talos_app.app            fenetre native (WebView2 / pywebview)
  python -m talos_app.app --browser  dans le navigateur par defaut
Avant le premier lancement : python -m talos_app.datasets (telechargement ~1,1 Go + construction des couches)."""
import argparse, os, sys, threading, time, urllib.request


def wait_ready(url, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(0.3)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--browser", action="store_true")
    a = ap.parse_args()
    from .datasets import BUILD
    if not os.path.exists(os.path.join(BUILD, "front.json")):
        sys.exit("Donnees absentes : lancer d'abord  python -m talos_app.datasets")
    from .server import serve
    threading.Thread(target=serve, kwargs={"port": a.port}, daemon=True).start()
    url = "http://127.0.0.1:%d/" % a.port
    if not wait_ready(url + "api/region", 120):
        sys.exit("le serveur ne repond pas")
    if a.browser:
        import webbrowser
        webbrowser.open(url)
        while True:
            time.sleep(3600)
    import webview
    webview.create_window("TALOS C2 — supervision robot autonome", url, width=1680, height=1000, min_size=(1280, 760),
                          background_color="#070b10")
    webview.start()


if __name__ == "__main__":
    main()
