"""
check_env.py  —  Verify that all pipeline dependencies are importable and report versions.
"""
import platform
import sys

print("Python   :", sys.version.split()[0])
print("Exec path:", sys.executable)
print("Platform :", platform.platform())
print()

checks = [
    ("torch",           lambda m: f"{m.__version__}  CUDA={m.cuda.is_available()}"),
    ("torchvision",     lambda m: m.__version__),
    ("ultralytics",     lambda m: m.__version__),
    ("transformers",    lambda m: m.__version__),
    ("cv2",             lambda m: m.__version__),
    ("PIL",             lambda m: m.__version__),
    ("numpy",           lambda m: m.__version__),
    ("pandas",          lambda m: m.__version__),
    ("tqdm",            lambda m: m.__version__),
    ("sklearn",         lambda m: m.__version__),
    ("rapidfuzz",       lambda m: m.__version__),
    ("lmdb",            lambda m: m.__version__),
    ("datasets",        lambda m: m.__version__),
    ("accelerate",      lambda m: m.__version__),
]

all_ok = True
for name, version_fn in checks:
    try:
        import importlib
        mod = importlib.import_module(name)
        print(f"  OK  {name:<18} {version_fn(mod)}")
    except Exception as exc:
        print(f"  FAIL {name:<17} {exc}")
        all_ok = False

print()
if all_ok:
    print("All checks passed.")
else:
    print("Some checks failed — run:  pip install -r requirements.txt")
    sys.exit(1)
