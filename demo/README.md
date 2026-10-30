# Run the SENTINEL demo

Use this guide to run the cached demo on your laptop and compare **Fusion classifier** with **Qwen detector**.
You need Python 3 and a current browser. The demo includes its media, fonts, scripts, and saved predictions.
After download, you can present offline without a GPU, model weights, or Python packages.

## Download the demo

Clone SENTINEL, then enter its directory:

```sh
git clone https://github.com/xena1t/SENTINEL.git
cd SENTINEL
```

If you do not use Git, select **Code**, then **Download ZIP** on [SENTINEL GitHub](https://github.com/xena1t/SENTINEL).
Extract the ZIP, then open a terminal in the extracted repository directory.
You can also copy the complete `demo` folder to another laptop.

## Start on macOS or Linux

From the repository directory, run:

```sh
python3 demo/serve.py --open
```

If you copied only the `demo` folder, open a terminal inside that folder and run:

```sh
python3 serve.py --open
```

On macOS, you can also double-click `Start SENTINEL.command` inside `demo`.
If macOS blocks the launcher, use the terminal command above.

## Start on Windows

From the repository directory, run this command in PowerShell or Command Prompt:

```powershell
py demo/serve.py --open
```

If you copied only the `demo` folder, open a terminal inside that folder and run:

```powershell
py serve.py --open
```

If your Python installation uses `python` instead of `py`, use `python` in the command.

## Compare the models

The browser opens at [http://127.0.0.1:4173](http://127.0.0.1:4173).
Look for **CACHED REPLAY**, three dataset cards, and both model cards.
Keep the terminal open while you use the demo.

1. Select **YES** or **NO** in a dataset card. Each selection opens a saved observation with all recorded inputs enabled.
2. For a video, select **Play recording**. Select **Hold window** to stop on a completed window.
3. Toggle the camera, RF, or audio switch to compare both models on that same observation.
4. Read the drone scores, decisions, and changes relative to all recorded inputs. Toggle an input back on to restore it.
5. Use the arrows or timeline slider to inspect another video window.
6. On TRIDENT, expand **Inspect 7 released sensor combinations** to compare the saved scores for that observation.
7. Expand **Follow the evidence through the model** to explain the illustrated pipeline.

For a rehearsal sequence, follow [Present the three dataset pairs](docs/present.md).
For keyboard shortcuts and control behavior, see the [control reference](docs/controls.md).
For score definitions and source limitations, see the [evidence reference](docs/evidence.md).

## Stop or restart the demo

To stop the server, press **Ctrl+C** in its terminal.
To restart it, run the startup command again.

If port 4173 is occupied, choose another port:

```sh
python3 demo/serve.py --port 4174 --open
```

On Windows, replace `python3` with `py`. From inside `demo`, omit `demo/` from the script path.

## Resolve startup problems

- If Python is missing, install Python 3 from [python.org](https://www.python.org/downloads/), then reopen the terminal.
- If the browser does not open, enter the URL printed in the terminal.
- If you opened `index.html` directly, close that tab and start `serve.py`. The demo needs an HTTP server to load its scripts and data.
- If media is missing, download or copy the complete `demo` folder, including `assets` and `data`.
- If a teammate cannot open your localhost URL, have them start their own copy. The server accepts connections only from its own laptop.

For developer checks and asset regeneration, use the [maintenance guide](docs/maintenance.md).
You do not need those commands to present.
