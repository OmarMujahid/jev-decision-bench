"""Builds every task file in tasks/ from public datasets (Hugging Face datasets-server) and code-labelled generators.
Usage: python3 build_tasks.py [name-filter]
Pages are cached in .cache/, so a second run is instant. A cold build takes a while because the datasets-server rate-limits anonymous callers."""
import glob, os, subprocess, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
flt = sys.argv[1] if len(sys.argv) > 1 else ""
scripts = [p for p in sorted(glob.glob(f"{ROOT}/builders/*.py")) if not p.endswith(("_common.py", "_prefetch.py")) and flt in os.path.basename(p)]
failed = []
for path in scripts:
    print(f"== {os.path.basename(path)}", flush=True)
    if subprocess.run([sys.executable, path], cwd=f"{ROOT}/builders").returncode:
        failed.append(os.path.basename(path))
ok = subprocess.run([sys.executable, f"{ROOT}/validate.py", *sorted(glob.glob(f"{ROOT}/tasks/*.json"))]).returncode == 0
print("\nfailed builders:", failed or "none", "| validation:", "OK" if ok else "FAILED")
sys.exit(1 if failed or not ok else 0)
