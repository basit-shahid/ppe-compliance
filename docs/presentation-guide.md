# Presentation and demonstration guide

Use this as the rehearsal checklist and speaker script. The software is a local
research prototype; independent final accuracy evaluation remains pending.

## 1. Prepare the presentation machine

From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[ui]"
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/verify_presentation.py
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

Use the existing environment if it is already installed. Setup requires internet;
local inference with available weights does not. Put the selected trained model
at `models/best.pt`. It is intentionally not committed. See
[training reproduction](phase-3-colab.md) for a fresh clone without weights.

Open **http://127.0.0.1:8501**. If that port is occupied, use
`--server.port 8502` and open the corresponding URL. Keep the laptop plugged in
and close unrelated heavy applications before measuring speed.

The rehearsal command writes a new unique `outputs/presentation-*` folder with
dependency/test logs, a verification JSON, image exports and an eight-frame
synthetic video. It checks actual local weights without a physical camera.
Use `--skip-tests` only when rerunning the smoke check after an unchanged suite;
`--output` must name a directory that does not already exist.

For offline rehearsal, retain the environment, weights, included sample, report
PDF and a completed exported MP4. Do not plan to train or download models live.

## 2. A six-minute demonstration

| Time | Action | Explain |
| --- | --- | --- |
| 0:00–0:40 | Show Inspection and the three source modes | Local inference assists worker PPE review |
| 0:40–1:40 | Check included sample and Analyze image | Actual person/helmet/vest evidence; the sample is development validation data |
| 1:40–2:20 | Open Source comparison and Worker evidence | Scores, ownership states and the reasons for review |
| 2:20–2:50 | Change Balanced to Detail; analyze again | Changed settings invalidate previous results; input size affects detail/speed |
| 2:50–3:30 | Download `compliance.json` | Source/model hashes, rule configuration and explanatory worker records |
| 3:30–4:20 | Show a completed scored video or short full analysis | Source FPS differs from processing FPS; 300-frame UI default is bounded |
| 4:20–5:10 | Open Project guide | Detection, association, status rules, tracking and exports are distinct stages |
| 5:10–6:00 | Open Validation | Saved validation metrics, weakest helmet recall and independent-test limitation |

The included close-worker sample currently requires review because the person
touches the image boundary. Both equipment matches can exist while the overall
status stays amber. This is a useful explanation of conservative rules; do not
change thresholds just to force a green demonstration.

If time permits, open Live camera, keep **In this browser**, choose the tested
camera index/backend, and press **Start live camera**. The feed appears below the
controls; stop it with **Stop live preview**. Show source motion separately from
timestamped AI evidence. Optional Desktop window mode retains Q/Esc, Tab and
fullscreen controls. Keep a recorded export available if camera hardware fails.

## 3. Video choices to explain accurately

- **Smooth live review** defaults to **In this browser**. Upload a clip and press
  **Start live review**. Source motion is independent of sampled AI; browser
  dashboard updates are also sampled. Use **Stop live preview** in the page.
- **Analyze every frame** performs inference before scored playback. Set frame
  limit to 0 for an entire clip. A completed bounded preview is not full coverage.
- **Live camera** uses newest-frame capture. Frames can be skipped intentionally
  to avoid an expanding queue. Recording saves selected AI frames when enabled.
- Efficient export is silent H.264. A browser-compatible movie does not imply
  real-time inference, preserved audio, or verified identity tracking.

For a short terminal camera check, run on the actual demonstration computer:

```powershell
.\.venv\Scripts\ppe.exe webcam --camera 0 --low-latency --track --image-size 416 --device cpu --duration 20
```

Only select CUDA `0` on a machine with configured NVIDIA PyTorch. The default
CPU mode is portable. Driver-accepted capture settings still need measurement.

## 4. Claims supported by the evidence

| You can say | Avoid claiming |
| --- | --- |
| We fine-tuned YOLO11 nano for three labels | We invented the YOLO architecture |
| Development validation mAP50 was 90.08% | The system is 90% accurate on any worksite |
| The code associates plausible head/torso evidence | A detected helmet is definitely worn correctly |
| Unknown/missing evidence defaults to review | Every missing box proves a violation |
| Software checks and local inference pass | Tests establish real-world safety or final accuracy |
| The live source can remain responsive during sampled AI | Smooth display means every frame has been analyzed |
| Independent test data is still required | Validation scenes are an independent final test set |

## 5. Troubleshooting during rehearsal

| Symptom | Action |
| --- | --- |
| Missing model message | Restore the selected `models/best.pt`; do not substitute general COCO weights for PPE assessment |
| Import or missing Streamlit/FFmpeg error | Use the project `.venv` and install `-e ".[ui]"`; run `pip check` |
| Corrupt image | Try a normal JPEG/PNG/WebP; validation should show an error without a traceback or package install |
| Image exceeds size limit | Resize below 24 megapixels; 200 MB compressed size alone does not bound decoded memory |
| CUDA/MPS rejected | Select `cpu`; use GPU only with a verified backend |
| Slow first result | Model load and warmup are startup work; use an already exported backup for time-limited demos |
| Video takes too long | Use a short clip, Fast/Balanced, a smaller processing resolution or a bounded frame limit |
| MP4 preview unavailable | Check the run's encoder log and disk space; original/partial artifacts may still be present |
| Output was removed | Analyze again. The UI reports moved/missing files without crashing |
| Camera does not open | Close other camera users, verify index/permissions and try auto, MSMF or DSHOW on Windows |
| No live preview appears | Select In this browser, then Start; wait for source/model startup and read any error/log in the panel |
| Camera window not visible remotely | Desktop window is optional; select In this browser for the embedded preview. The camera must still be attached to the app host |
| Worker is amber despite equipment boxes | Read the evidence reasons: weak/small/border/overlap/ambiguity/confirmation warnings may apply |

Camera launch errors are recorded in `outputs/camera-launch.log`; smooth-review
launch errors in `outputs/recorded-review-launch.log`. Reports remain local until
you explicitly copy/download them. Uploaded review inputs and playback caches
can use disk space; archive or remove unwanted local artifacts between sessions.

Embedded-preview errors and logs are shown directly under the live panel; their
worker files are under `outputs/.browser-previews/`. The presentation laptop was
checked with **camera index 1 / MSMF / 1280x720**. Camera 0 opened without readable
frames on this setup, so select the verified combination when rehearsing here.

## 6. What to bring to the examination

- Current [project report](PPE_Project_Comprehensive_Report.pdf).
- [Pipeline guide](pipeline.md) and [viva/code walkthrough](code-walkthrough.md).
- [Slide content and speaker notes](submission-slides.md).
- `models/best.pt`, an installed environment and the included attributed sample.
- A scored MP4 and JSON reports that have already been checked on the laptop.
- The latest generated `verification.json` and automated test log.
- An honest explanation of independent-data and sustained-performance limits.

## 7. Optional browser QA

`check_workspace_browser.py` checks an already-running app using a dedicated
Chromium DevTools session. Install `.[qa]` for its optional `websockets` dependency.
Use a disposable browser profile, not a personal authenticated profile. Launch
your installed Chromium browser with `--headless=new --disable-gpu`, a dedicated
`--user-data-dir`, and a local `--remote-debugging-port=9234`. Keep that process
running while testing.

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py --server.port 8502 --server.headless true
# In another terminal, after launching the dedicated browser:
.\.venv\Scripts\python.exe scripts/check_workspace_browser.py --url http://127.0.0.1:8502 --debug-port 9234
```

It checks desktop rendering, actual sample inference, evidence/navigation and a
390px viewport. Screenshots/report go under `outputs/`. A headless environment
does not verify physical camera access; run that check locally if using the camera
in your presentation.

Add `--video <short-local-clip.mp4>` to additionally upload, analyze, export and
play a short clip in the actual browser. You can use the eight-frame
`rehearsal-input.mp4` created by the presentation verifier. This is an integration
check, not a natural-motion accuracy benchmark.
