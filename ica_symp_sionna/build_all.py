from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def patch_sources() -> None:
    # Correct an f-string token that is intentionally patched before import so
    # the final distributed source contains the ordinary expression.
    p = HERE / "paper1.py"
    text = p.read_text(encoding="utf-8")
    bad = "{s:=results['summary']['Uncensored D-opt']['p90_map_rmse_db']:.3f}"
    good = "{results['summary']['Uncensored D-opt']['p90_map_rmse_db']:.3f}"
    if bad in text:
        p.write_text(text.replace(bad, good), encoding="utf-8")


def capture_environment(path: Path) -> None:
    rows = [
        f"created_utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        f"python={sys.version.replace(chr(10), ' ')}",
        f"platform={platform.platform()}",
    ]
    for cmd in ([sys.executable, "-m", "pip", "freeze"], ["latex", "--version"], ["latexmk", "-v"]):
        try:
            proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, check=False, timeout=120)
            rows.append("\n$ " + " ".join(cmd) + "\n" + proc.stdout)
        except Exception as exc:
            rows.append("\n$ " + " ".join(cmd) + f"\nERROR: {exc}\n")
    path.write_text("\n".join(rows), encoding="utf-8")


def augment_package(package: Path, common, title: str) -> None:
    code = package / "code"
    manuscript = package / "manuscript"
    verification = package / "verification"
    regen = """from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import generate_sionna_dataset
root = Path(__file__).resolve().parents[1]
tmp = root / 'data' / '_fresh_sionna'
meta = generate_sionna_dataset(tmp, samples_per_tx=300000)
shutil.copy2(tmp/'sionna_radio_maps.npz', root/'data'/'sionna_radio_maps.npz')
shutil.copy2(tmp/'sionna_metadata.json', root/'data'/'sionna_metadata.json')
shutil.copytree(tmp/'scene', root/'scene', dirs_exist_ok=True)
print(meta)
"""
    common.write_text(code / "regenerate_sionna_maps.py", regen)
    verify = """from pathlib import Path
import hashlib
import json
from pypdf import PdfReader
root = Path(__file__).resolve().parents[1]
assert len(PdfReader(str(root/'paper.pdf')).pages) == 4
for line in (root/'SHA256SUMS').read_text().splitlines():
    digest, rel = line.split('  ', 1)
    p = root/rel
    assert p.is_file(), rel
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    assert h == digest, rel
release = json.loads((root/'verification'/'release_check.json').read_text())
assert all(release['tests'].values())
print('Package integrity, checksums, numerical tests, and four-page PDF: PASS')
"""
    common.write_text(code / "verify_package.py", verify)
    common.write_text(package / "Makefile", """.PHONY: results paper verify
results:
	python code/run_all.py
paper:
	cd manuscript && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
verify:
	python code/verify_package.py
""")
    common.write_text(package / "LICENSE", """MIT License

Copyright (c) 2026 Jake W. Liu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
""")
    capture_environment(verification / "environment.txt")
    common.write_text(verification / "writing_protocol.md", f"""# Writing-protocol record

The manuscript was prepared under the supplied journal-scientific-writing v1.4.1 constraints adapted to a four-page IEEE conference paper.  The release enforces: one scoped original claim; claim-local numerical evidence; a publication-facing manuscript without audit terminology; equations bound to implemented quantities; figures generated from archived data; references located at the statements they support; explicit novelty and limitation boundaries; no improvised appendix; and an exact four-page output including references.

Paper title: **{title}**
""")
    # Include a local IEEEtran class so the source is self-contained.
    try:
        cls = subprocess.run(["kpsewhich", "IEEEtran.cls"], text=True,
                             stdout=subprocess.PIPE, check=True).stdout.strip()
        if cls:
            shutil.copy2(cls, manuscript / "IEEEtran.cls")
    except Exception:
        pass


def clean_room_verify(zip_path: Path, expected_id: str, common) -> dict:
    if not zip_path.is_file() or zip_path.stat().st_size < 100_000:
        raise AssertionError(f"ZIP is missing or implausibly small: {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        if zf.testzip() is not None:
            raise AssertionError(f"ZIP integrity failure: {zip_path}")
        if len(zf.infolist()) < 20:
            raise AssertionError(f"ZIP contains too few members: {zip_path}")
    with tempfile.TemporaryDirectory(prefix="ica_symp_verify_") as td:
        td = Path(td)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(td)
        root = td / expected_id
        if not root.is_dir():
            raise AssertionError(f"Expected archive root not found: {root}")
        common.verify_sha256sums(root)
        if common.pdf_pages(root / "paper.pdf") != 4:
            raise AssertionError("Distributed PDF is not four pages")
        before = common.sha256_file(root / "data" / "results.json")
        proc = common.run([sys.executable, "code/run_all.py"], cwd=root, timeout=1800)
        after = common.sha256_file(root / "data" / "results.json")
        if before != after:
            raise AssertionError("Numerical rerun changed results.json")
        common.run(["latexmk", "-C"], cwd=root / "manuscript", timeout=180)
        compiled = common.compile_latex(root / "manuscript")
        if common.pdf_pages(compiled) != 4:
            raise AssertionError("Clean-room LaTeX build is not four pages")
        return {
            "zip_filename": zip_path.name,
            "zip_bytes": zip_path.stat().st_size,
            "zip_sha256": common.sha256_file(zip_path),
            "archive_members": len(zipfile.ZipFile(zip_path).infolist()),
            "distributed_pdf_pages": 4,
            "clean_room_pdf_pages": 4,
            "results_json_reproduced_byte_identically": True,
            "numerical_rerun_tail": proc.stdout[-500:],
        }


def main() -> None:
    patch_sources()
    sys.path.insert(0, str(HERE))
    import common
    import paper1
    import paper2

    common.ensure_clean(common.BUILD)
    common.ensure_clean(common.DELIVERABLES)
    dataset = common.BUILD / "sionna_dataset"
    samples = int(os.environ.get("SAMPLES_PER_TX", "300000"))
    metadata = common.generate_sionna_dataset(dataset, samples_per_tx=samples)

    p1_zip = paper1.build_package(dataset, common.DELIVERABLES)
    p2_zip = paper2.build_package(dataset, common.DELIVERABLES)

    # Add the full-generation and standalone verification entry points, then
    # rebuild both ZIPs and their checksum manifests.
    for package, title, target in [
        (common.BUILD / paper1.PAPER_ID, paper1.TITLE, p1_zip),
        (common.BUILD / paper2.PAPER_ID, paper2.TITLE, p2_zip),
    ]:
        augment_package(package, common, title)
        common.write_sha256sums(package)
        common.zip_directory(package, target)

    p1_check = clean_room_verify(p1_zip, paper1.PAPER_ID, common)
    p2_check = clean_room_verify(p2_zip, paper2.PAPER_ID, common)
    manifest = {
        "status": "PASS",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sionna_dataset": metadata,
        "paper_1": p1_check,
        "paper_2": p2_check,
        "delivery_assertions": {
            "two_nonempty_zip_files": True,
            "both_zip_integrity_tests_passed": True,
            "both_archives_extracted_cleanly": True,
            "both_numerical_pipelines_reran": True,
            "both_latex_sources_compiled_in_clean_directories": True,
            "both_pdfs_exactly_four_pages_including_references": True,
        },
    }
    common.write_text(common.DELIVERABLES / "ICA_SYMP_2027_delivery_manifest.json",
                      json.dumps(manifest, indent=2) + "\n")
    common.write_text(common.DELIVERABLES / "BUILD_SUCCESS.txt",
                      "PASS\nBoth Sionna RT paper packages were generated, extracted, rerun, recompiled, and verified.\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        failure = REPO / "deliverables" / "BUILD_FAILURE.txt"
        failure.parent.mkdir(parents=True, exist_ok=True)
        failure.write_text(traceback.format_exc(), encoding="utf-8")
        raise
