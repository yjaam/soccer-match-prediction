import tls_requests.models.libraries as L
import inspect, textwrap, re
src = inspect.getsource(L)
print("libraries.py:", L.__file__)
# Print the first place where cache dir / library name is set (best-effort)
for pat in ["cache", "CACHE", "library", "LIB", "dylib", "darwin"]:
    if pat in src:
        pass
print("---- excerpt ----")
lines = src.splitlines()
for i, line in enumerate(lines, 1):
    if "dylib" in line or "darwin" in line or "cache" in line.lower() or "download" in line.lower():
        print(f"{i}: {line}")
