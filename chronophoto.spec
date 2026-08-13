# -*- mode: python ; coding: utf-8 -*-

import sys
import tomllib
from pathlib import Path


project_root = Path(SPECPATH)
package_root = project_root / "src" / "chronophoto"
platform_icons = {
    "win32": project_root / "packaging" / "windows" / "chronophoto.ico",
    "darwin": project_root / "packaging" / "macos" / "chronophoto.icns",
}
platform_icon = platform_icons.get(sys.platform)
datas = [(str(package_root / "assets"), "chronophoto/assets")]
if sys.platform == "win32":
    nvidia_runtime = project_root / "build" / "vendor-python312" / "runtime"
    if nvidia_runtime.is_dir():
        datas.append((str(nvidia_runtime), "nvidia-runtime"))
with (project_root / "pyproject.toml").open("rb") as project_file:
    project_version = tomllib.load(project_file)["project"]["version"]

a = Analysis(
    [str(package_root / "__main__.py")],
    pathex=[str(project_root / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Chronophoto",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    icon=str(platform_icon) if platform_icon else None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="Chronophoto",
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="Chronophoto.app",
        icon=str(project_root / "packaging" / "macos" / "chronophoto.icns"),
        bundle_identifier="io.chronophoto.desktop",
        info_plist={
            "CFBundleDisplayName": "Chronophoto",
            "CFBundleShortVersionString": project_version,
            "CFBundleVersion": project_version,
            "NSHighResolutionCapable": True,
        },
    )
