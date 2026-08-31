from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import textwrap
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "_build"
DELIVERABLES = ROOT.parent / "deliverables"


@dataclass(frozen=True)
class Box:
    name: str
    x0: float
    x1: float
    y0: float
    y1: float
    z0: float
    z1: float
    material: str

    def footprint(self) -> tuple[float, float, float, float]:
        return self.x0, self.x1, self.y0, self.y1


def ensure_clean(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run(cmd: Sequence[str], cwd: Path | None = None, timeout: int = 1800,
        env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    proc = subprocess.run(
        list(cmd), cwd=str(cwd) if cwd else None, env=merged,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=timeout, check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout[-12000:]}"
        )
    return proc


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_sha256sums(root: Path) -> None:
    rows = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS":
            rows.append(f"{sha256_file(p)}  {p.relative_to(root).as_posix()}")
    write_text(root / "SHA256SUMS", "\n".join(rows) + "\n")


def verify_sha256sums(root: Path) -> None:
    sums = root / "SHA256SUMS"
    if not sums.is_file() or sums.stat().st_size == 0:
        raise AssertionError("SHA256SUMS is missing or empty")
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, rel = line.split("  ", 1)
        p = root / rel
        if not p.is_file():
            raise AssertionError(f"Missing checksummed file: {rel}")
        if sha256_file(p) != digest:
            raise AssertionError(f"Checksum mismatch: {rel}")


def write_box_ply(path: Path, box: Box) -> None:
    x0, x1, y0, y1, z0, z1 = box.x0, box.x1, box.y0, box.y1, box.z0, box.z1
    vertices = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    ]
    faces = [
        (0, 2, 1), (0, 3, 2),
        (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4),
        (3, 7, 6), (3, 6, 2),
        (0, 4, 7), (0, 7, 3),
        (1, 2, 6), (1, 6, 5),
    ]
    lines = [
        "ply", "format ascii 1.0", f"element vertex {len(vertices)}",
        "property float x", "property float y", "property float z",
        f"element face {len(faces)}", "property list uchar int vertex_indices",
        "end_header",
    ]
    lines.extend(f"{x:.8f} {y:.8f} {z:.8f}" for x, y, z in vertices)
    lines.extend("3 " + " ".join(map(str, f)) for f in faces)
    write_text(path, "\n".join(lines) + "\n")


def scene_boxes() -> list[Box]:
    # The geometry is deliberately compact so that the complete Sionna RT study
    # remains reproducible on a CPU-only GitHub runner.
    h = 3.0
    t = 0.20
    return [
        Box("floor", 0.0, 20.0, 0.0, 15.0, -0.20, 0.0, "floor-mat"),
        Box("south-wall", 0.0, 20.0, 0.0, t, 0.0, h, "wall-mat"),
        Box("north-wall", 0.0, 20.0, 15.0-t, 15.0, 0.0, h, "wall-mat"),
        Box("west-wall", 0.0, t, 0.0, 15.0, 0.0, h, "wall-mat"),
        Box("east-wall", 20.0-t, 20.0, 0.0, 15.0, 0.0, h, "wall-mat"),
        Box("vwall-a", 8.70, 8.90, 0.20, 5.15, 0.0, h, "wall-mat"),
        Box("vwall-b", 8.70, 8.90, 6.85, 10.50, 0.0, h, "wall-mat"),
        Box("hwall-a", 8.70, 12.90, 10.30, 10.50, 0.0, h, "wall-mat"),
        Box("hwall-b", 14.70, 19.80, 10.30, 10.50, 0.0, h, "wall-mat"),
        Box("vwall-c", 15.30, 15.50, 10.30, 14.80, 0.0, h, "wall-mat"),
        Box("wood-partition", 3.80, 4.00, 7.00, 12.80, 0.0, 2.6, "wood-mat"),
        Box("metal-cabinet", 10.70, 13.50, 5.00, 7.80, 0.0, 2.2, "metal-mat"),
    ]


def build_scene_assets(scene_dir: Path) -> tuple[Path, list[Box]]:
    ensure_clean(scene_dir)
    meshes = scene_dir / "meshes"
    meshes.mkdir(parents=True, exist_ok=True)
    boxes = scene_boxes()
    for box in boxes:
        write_box_ply(meshes / f"{box.name}.ply", box)

    material_xml = """
    <bsdf type="itu-radio-material" id="wall-mat">
        <string name="type" value="concrete"/>
        <float name="thickness" value="0.20"/>
    </bsdf>
    <bsdf type="itu-radio-material" id="floor-mat">
        <string name="type" value="concrete"/>
        <float name="thickness" value="0.20"/>
    </bsdf>
    <bsdf type="itu-radio-material" id="wood-mat">
        <string name="type" value="wood"/>
        <float name="thickness" value="0.10"/>
    </bsdf>
    <bsdf type="itu-radio-material" id="metal-mat">
        <string name="type" value="metal"/>
        <float name="thickness" value="0.10"/>
    </bsdf>
    """
    shapes = []
    for box in boxes:
        shapes.append(f"""
    <shape type="ply" id="mesh-{box.name}">
        <string name="filename" value="meshes/{box.name}.ply"/>
        <boolean name="face_normals" value="true"/>
        <ref id="{box.material}" name="bsdf"/>
    </shape>""")
    xml = "<scene version=\"2.1.0\">\n" + material_xml + "\n".join(shapes) + "\n</scene>\n"
    xml_path = scene_dir / "indoor_twin.xml"
    write_text(xml_path, xml)
    write_text(scene_dir / "geometry.json", json.dumps([box.__dict__ for box in boxes], indent=2))
    return xml_path, boxes


def configure_mitsuba() -> str:
    import mitsuba as mi
    current = mi.variant()
    if current:
        return current
    candidates = [
        "llvm_ad_mono_polarized",
        "llvm_ad_rgb",
        "scalar_mono_polarized",
        "scalar_rgb",
    ]
    available = set(mi.variants())
    for variant in candidates:
        if variant in available:
            mi.set_variant(variant)
            return variant
    raise RuntimeError(f"No compatible Mitsuba variant. Available: {sorted(available)}")


def _material_items(scene):
    mats = scene.radio_materials
    if hasattr(mats, "items"):
        return list(mats.items())
    names = list(mats)
    return [(n, mats[n]) for n in names]


def find_wall_material(scene):
    items = _material_items(scene)
    for name, material in items:
        if "wall" in str(name).lower():
            return str(name), material
    for name, material in items:
        if "concrete" in str(name).lower():
            return str(name), material
    raise RuntimeError(f"Wall radio material not found. Materials: {[n for n, _ in items]}")


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def normalize_rss_array(raw: np.ndarray, num_tx: int) -> np.ndarray:
    arr = np.asarray(raw)
    arr = np.squeeze(arr)
    if arr.ndim == 2 and num_tx == 1:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise RuntimeError(f"Unexpected radio-map RSS shape: {arr.shape}")
    if arr.shape[0] != num_tx:
        axes = [i for i, n in enumerate(arr.shape) if n == num_tx]
        if not axes:
            raise RuntimeError(f"Cannot locate transmitter axis in {arr.shape}")
        arr = np.moveaxis(arr, axes[0], 0)
    if np.nanmin(arr) < -1.0:
        # Defensive branch for API variants returning dBm.
        return arr.astype(float)
    return 10.0 * np.log10(np.maximum(arr.astype(float), 1e-18) / 1e-3)


def make_scene(xml_path: Path, tx_positions: Sequence[tuple[float, float, float]],
               frequency_hz: float = 3.5e9, power_dbm: float = 20.0):
    configure_mitsuba()
    import mitsuba as mi
    from sionna.rt import load_scene, PlanarArray, Transmitter

    scene = load_scene(str(xml_path))
    scene.frequency = float(frequency_hz)
    scene.tx_array = PlanarArray(
        num_rows=1, num_cols=1,
        vertical_spacing=0.5, horizontal_spacing=0.5,
        pattern="iso", polarization="V",
    )
    for idx, position in enumerate(tx_positions):
        tx = Transmitter(
            name=f"tx-{idx+1}",
            position=list(position),
            orientation=[0.0, 0.0, 0.0],
            power_dbm=float(power_dbm),
        )
        scene.add(tx)
    return scene


def solve_radio_map(scene, samples_per_tx: int, seed: int) -> np.ndarray:
    from sionna.rt import RadioMapSolver
    solver = RadioMapSolver()
    rm = solver(
        scene,
        center=(10.0, 7.5, 1.20),
        orientation=(0.0, 0.0, 0.0),
        size=(19.0, 14.0),
        cell_size=(0.5, 0.5),
        samples_per_tx=int(samples_per_tx),
        max_depth=5,
        los=True,
        specular_reflection=True,
        diffuse_reflection=False,
        refraction=True,
        diffraction=False,
        edge_diffraction=False,
        seed=int(seed),
        rr_depth=-1,
    )
    return normalize_rss_array(_to_numpy(rm.rss), len(scene.transmitters))


def generate_sionna_dataset(output: Path, samples_per_tx: int) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    scene_dir = output / "scene"
    xml_path, boxes = build_scene_assets(scene_dir)
    eps0, sigma0 = 5.0, 0.050
    deps, dsigma = 0.50, 0.015

    # Paper 1: common-random-number central differences for one transmitter.
    scene1 = make_scene(xml_path, [(2.50, 2.50, 2.60)])
    wall_name, wall = find_wall_material(scene1)
    maps1 = {}
    for tag, eps, sig in [
        ("nominal", eps0, sigma0),
        ("eps_plus", eps0 + deps, sigma0),
        ("eps_minus", eps0 - deps, sigma0),
        ("sigma_plus", eps0, sigma0 + dsigma),
        ("sigma_minus", eps0, sigma0 - dsigma),
    ]:
        wall.relative_permittivity = float(eps)
        wall.conductivity = float(sig)
        maps1[tag] = solve_radio_map(scene1, samples_per_tx=samples_per_tx, seed=271828)[0]

    # Paper 2: independent map with two access points. The maximum is the
    # best-server RSS used by the graph planner.
    scene2 = make_scene(xml_path, [(2.50, 2.50, 2.60), (17.50, 12.50, 2.60)])
    _, wall2 = find_wall_material(scene2)
    wall2.relative_permittivity = eps0
    wall2.conductivity = sigma0
    maps2 = solve_radio_map(scene2, samples_per_tx=samples_per_tx, seed=314159)

    nominal = maps1["nominal"]
    ny, nx = nominal.shape
    x = np.linspace(10.0 - 19.0/2 + 0.25, 10.0 + 19.0/2 - 0.25, nx)
    y = np.linspace(7.5 - 14.0/2 + 0.25, 7.5 + 14.0/2 - 0.25, ny)

    npz_path = output / "sionna_radio_maps.npz"
    np.savez_compressed(
        npz_path,
        x=x, y=y,
        p1_nominal=maps1["nominal"],
        p1_eps_plus=maps1["eps_plus"],
        p1_eps_minus=maps1["eps_minus"],
        p1_sigma_plus=maps1["sigma_plus"],
        p1_sigma_minus=maps1["sigma_minus"],
        p2_tx1=maps2[0], p2_tx2=maps2[1],
        p2_best_server=np.maximum(maps2[0], maps2[1]),
    )
    metadata = {
        "backend": "Sionna RT",
        "sionna_rt_version": "2.0.1",
        "mitsuba_variant": configure_mitsuba(),
        "frequency_hz": 3.5e9,
        "transmit_power_dbm": 20.0,
        "samples_per_tx": int(samples_per_tx),
        "map_center_m": [10.0, 7.5, 1.2],
        "map_size_m": [19.0, 14.0],
        "cell_size_m": [0.5, 0.5],
        "max_depth": 5,
        "interactions": {"los": True, "specular_reflection": True,
                         "refraction": True, "diffraction": False,
                         "diffuse_reflection": False},
        "wall_material_name": wall_name,
        "nominal_wall_relative_permittivity": eps0,
        "nominal_wall_conductivity_s_per_m": sigma0,
        "finite_difference_steps": {"relative_permittivity": deps,
                                    "conductivity_s_per_m": dsigma},
        "geometry": [box.__dict__ for box in boxes],
        "npz_sha256": sha256_file(npz_path),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_text(output / "sionna_metadata.json", json.dumps(metadata, indent=2) + "\n")
    return metadata


def point_blocked(x: float, y: float, boxes: Iterable[Box], clearance: float = 0.20) -> bool:
    for box in boxes:
        if box.name == "floor":
            continue
        if (box.x0-clearance <= x <= box.x1+clearance and
                box.y0-clearance <= y <= box.y1+clearance):
            return True
    return False


def occupancy_mask(x: np.ndarray, y: np.ndarray, clearance: float = 0.20) -> np.ndarray:
    boxes = scene_boxes()
    mask = np.zeros((len(y), len(x)), dtype=bool)
    for iy, yy in enumerate(y):
        for ix, xx in enumerate(x):
            mask[iy, ix] = point_blocked(float(xx), float(yy), boxes, clearance)
    return mask


def compile_latex(project_dir: Path, main_name: str = "main.tex") -> Path:
    run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", main_name],
        cwd=project_dir, timeout=900)
    pdf = project_dir / Path(main_name).with_suffix(".pdf")
    if not pdf.is_file() or pdf.stat().st_size < 10_000:
        raise AssertionError(f"Compiled PDF missing or too small: {pdf}")
    return pdf


def pdf_pages(path: Path) -> int:
    from pypdf import PdfReader
    return len(PdfReader(str(path)).pages)


def zip_directory(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as zf:
        for p in sorted(source.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=f"{source.name}/{p.relative_to(source).as_posix()}")
    with zipfile.ZipFile(target, "r") as zf:
        bad = zf.testzip()
        if bad is not None:
            raise AssertionError(f"Corrupt ZIP member: {bad}")
        names = zf.namelist()
        required_suffixes = ["paper.pdf", "manuscript/main.tex", "code/run_all.py",
                             "data/results.json", "SHA256SUMS"]
        for suffix in required_suffixes:
            if not any(n.endswith(suffix) for n in names):
                raise AssertionError(f"ZIP lacks required member ending with {suffix}")
        for info in zf.infolist():
            if not info.is_dir() and info.file_size == 0:
                raise AssertionError(f"ZIP contains empty file: {info.filename}")


def clean_latex_aux(project_dir: Path) -> None:
    for suffix in [".aux", ".bbl", ".blg", ".fdb_latexmk", ".fls", ".log", ".out"]:
        for p in project_dir.glob(f"*{suffix}"):
            p.unlink(missing_ok=True)


def make_release_report(package: Path, title: str, numerical_backend: str,
                        tests: dict) -> None:
    report = {
        "title": title,
        "paper_pages": pdf_pages(package / "paper.pdf"),
        "paper_sha256": sha256_file(package / "paper.pdf"),
        "numerical_backend": numerical_backend,
        "tests": tests,
        "file_count": sum(1 for p in package.rglob("*") if p.is_file()),
        "total_bytes": sum(p.stat().st_size for p in package.rglob("*") if p.is_file()),
        "verified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_text(package / "verification" / "release_check.json",
               json.dumps(report, indent=2) + "\n")
