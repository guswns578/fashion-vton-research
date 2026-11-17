"""Queue the CLO3D ControlNet graph on ComfyUI 8188 and wait for it.
  python run_clo3d_test.py            -> one run (both controls)
  python run_clo3d_test.py --ablate   -> same seed x {none, depth-only, normal-only, both}
  python run_clo3d_test.py --maps gc1r [--ablate]  -> use prepared real maps gc1r_{normal,depth}.png from COMFY/input
Results: work/clo3d_test_result.json, images copied to outputs/clo3d_test_result_<tag>.png.

GPU memory: `nvidia-smi -lms 100` written to a file (work/gpu_samples_<tag>.txt) while the prompt runs.
The value recorded is "the maximum whole-GPU used memory observed in this run at 100 ms sampling";
it includes ComfyUI's model cache and any other process, and can miss shorter spikes.
Timeout: only this script's own prompt is touched (deleted if pending, /interrupt only if it is the one running).
Exit code 1 on any failure/timeout."""

import json, time, subprocess, urllib.request, sys, shutil
from pathlib import Path

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / "work"))
import build_clo3d_workflow as B

COMFY = B.COMFY
TIMEOUT_S = 900


def get(u):
    return json.load(urllib.request.urlopen("http://127.0.0.1:8188" + u, timeout=10))


def post(u, d):
    r = urllib.request.Request(
        "http://127.0.0.1:8188" + u,
        data=json.dumps(d).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.load(urllib.request.urlopen(r, timeout=30))


def cancel_own(pid):
    q = get("/queue")
    if any(item[1] == pid for item in q["queue_pending"]):
        post("/queue", {"delete": [pid]})
        return "deleted-pending"
    if any(item[1] == pid for item in q["queue_running"]):
        post("/interrupt", {})
        return "interrupted-running"
    return "not-in-queue"


def run(api, tag):
    q = get("/queue")
    if q["queue_running"] or q["queue_pending"]:
        sys.exit(f"queue not empty, refusing to start: {q}")
    sample_file = root / "work" / f"gpu_samples_{tag}.txt"
    smi = None
    pid = None
    try:
        with sample_file.open("w") as fh:
            smi = subprocess.Popen(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                    "-lms",
                    "100",
                ],
                stdout=fh,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        t0 = time.time()
        pid = post("/prompt", {"prompt": api, "client_id": "clo3d-test"})["prompt_id"]
        while True:
            h = get("/history/" + pid)
            if pid in h and h[pid].get("status", {}).get("completed") is not None:
                break
            if time.time() - t0 > TIMEOUT_S:
                print("TIMEOUT", pid, cancel_own(pid))
                sys.exit(1)
            time.sleep(0.5)
    finally:
        if smi:
            smi.terminate()
            smi.wait()
    samples = [int(x) for x in sample_file.read_text().split() if x.strip().isdigit()]
    el = time.time() - t0
    st = h[pid]["status"]
    outs = [o for v in h[pid].get("outputs", {}).values() for o in v.get("images", [])]
    res = {
        "tag": tag,
        "prompt_id": pid,
        "elapsed_s_polling": round(el, 2),
        "status": st.get("status_str"),
        "gpu_used_max_observed_mib_100ms": max(samples) if samples else None,
        "gpu_used_min_observed_mib_100ms": min(samples) if samples else None,
        "gpu_samples": len(samples),
        "outputs": outs,
    }
    if st.get("status_str") != "success":
        for m in st.get("messages", []):
            if m[0] == "execution_error":
                res["error"] = m[1]
                print(json.dumps(m[1], indent=1)[:4000])
        print(json.dumps(res, indent=2))
        sys.exit(1)
    for o in outs:
        shutil.copyfile(
            COMFY / "output" / o["subfolder"] / o["filename"],
            root / "outputs" / f"clo3d_test_result_{tag}.png",
        )
    print(json.dumps(res))
    return res


if __name__ == "__main__":
    # --maps PREFIX : use outputs/prepared/<PREFIX>_{normal,depth}.png (must already be in COMFY/input,
    #                 i.e. prepare_clo3d_maps.py run without --no-install) instead of the synthetic maps
    maps = sys.argv[sys.argv.index("--maps") + 1] if "--maps" in sys.argv else None
    imgs = {"normal_img": f"{maps}_normal.png", "depth_img": f"{maps}_depth.png"} if maps else {}
    for k in imgs.values():
        if maps and not (COMFY / "input" / k).exists():
            sys.exit(f'missing {COMFY / "input" / k}')
    tagp = f"{maps}_" if maps else ""
    if "--ablate" in sys.argv:
        variants = [
            ("none", 0.0, 0.0),
            ("depth", 0.0, B.DEPTH_STR),
            ("normal", B.NORMAL_STR, 0.0),
            ("both", B.NORMAL_STR, B.DEPTH_STR),
        ]
    else:
        variants = [("both", B.NORMAL_STR, B.DEPTH_STR)]
    results = [
        run(
            B.api_prompt(normal_str=n, depth_str=d, prefix=f"CLO3D/{tagp}ablate_{t}", **imgs),
            tagp + t,
        )
        for t, n, d in variants
    ]
    (root / "work/clo3d_test_result.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
