"""Serve the offline SENTINEL demo with seekable local media."""

import argparse
import functools
import re
import shutil
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class ReplayHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        self.remaining = None
        requested_range = self.headers.get("Range")
        path = Path(self.translate_path(self.path))
        if not requested_range or not path.is_file():
            return super().send_head()

        size = path.stat().st_size
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested_range.strip())
        start, end = 0, size - 1
        if match and any(match.groups()):
            first, last = match.groups()
            if first:
                start = int(first)
                end = min(int(last), size - 1) if last else size - 1
            else:
                start = max(0, size - int(last))
        else:
            start = size

        if start >= size or start > end:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None

        stream = path.open("rb")
        stream.seek(start)
        self.remaining = end - start + 1
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(self.remaining))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        return stream

    def copyfile(self, source, outputfile):
        if self.remaining is None:
            shutil.copyfileobj(source, outputfile)
            return
        while self.remaining:
            chunk = source.read(min(self.remaining, 64 * 1024))
            if not chunk:
                break
            outputfile.write(chunk)
            self.remaining -= len(chunk)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=4173)
    parser.add_argument("--open", action="store_true", help="Open the replay in your browser.")
    args = parser.parse_args()
    handler = functools.partial(ReplayHandler, directory=str(Path(__file__).parent))
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        print(f"SENTINEL demo at http://127.0.0.1:{args.port}", flush=True)
        if args.open:
            threading.Timer(0.4, webbrowser.open, args=(f"http://127.0.0.1:{args.port}",)).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
