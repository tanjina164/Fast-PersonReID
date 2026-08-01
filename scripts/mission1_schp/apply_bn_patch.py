
import shutil
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCHP_MODULES_DIR = os.path.join(REPO_ROOT, "external", "SCHP", "modules")

patch_src = os.path.join(os.path.dirname(__file__), "bn_patch.py")
bn_dst = os.path.join(SCHP_MODULES_DIR, "bn.py")

shutil.copy(patch_src, bn_dst)

init_content = "from .bn import ABN, InPlaceABN, InPlaceABNSync\n"
with open(os.path.join(SCHP_MODULES_DIR, "__init__.py"), "w") as f:
    f.write(init_content)

print(f"Patched: {bn_dst}")
print(f"Patched: {os.path.join(SCHP_MODULES_DIR, '__init__.py')}")
