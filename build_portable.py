"""Construye el paquete portable de la demo para el dia de la sustentacion.

Estrategia (opcion A): en lugar de congelar la aplicacion con PyInstaller — que
con PyTorch y Streamlit produce binarios de ~1 GB, lentos de arrancar y con alta
probabilidad de ser bloqueados por SmartScreen o el antivirus del equipo de la
sala — se construye una carpeta autocontenida con:

    DemoEMG/
        Iniciar_Demo.bat      Lanzador (doble clic)
        python/               Python embebido de python.org (no requiere instalar)
        app/                  demo_app.py, src/, assets/, .streamlit/
        app/data/             demo_cache.npz (sin los .mat originales)
        app/checkpoints/demo/ Pesos del ensemble, sin estado del optimizador

La carpeta funciona sin conexion, sin permisos de administrador y sin tocar el
Python del equipo anfitrion. Se puede copiar a un USB y ejecutar desde alli.

Uso:
    python build_portable.py                # descarga dependencias con pip
    python build_portable.py --copy-deps    # copia las del entorno actual (offline)
    python build_portable.py --no-torch     # sin PyTorch: usa el cache precomputado

Requiere conexion la primera vez, salvo con --copy-deps.
"""

from __future__ import annotations

import argparse
import compileall
import os
import shutil
import subprocess
import sys
import sysconfig
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent
DIST = ROOT / "dist" / "DemoEMG"
PY_DIR = DIST / "python"
APP_DIR = DIST / "app"

PY_VERSION = "3.13.1"
PY_EMBED_URL = f"https://www.python.org/ftp/python/{PY_VERSION}/python-{PY_VERSION}-embed-amd64.zip"
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"

# Dependencias con version EXACTA, no con rango. El paquete que se lleva a la
# sustentacion tiene que ser el mismo software que se probo: con rangos abiertos,
# una construccion posterior instala versiones mas nuevas y el resultado deja de
# estar verificado (por ejemplo, Streamlit 1.60 cambio el servidor a Uvicorn y la
# barra lateral dejo de renderizarse). Para actualizar estas versiones, cambialas
# aqui, reconstruye y vuelve a revisar la demo entera antes de confiar en ella.
DEPS_BASE = [
    "streamlit==1.54.0",
    "plotly==6.5.2",
    "numpy==2.3.2",
    "pandas==2.3.2",
    "scikit-learn==1.8.0",
    "scipy==1.17.0",
    "matplotlib==3.10.8",
]
# Variante CPU de PyTorch: la de CUDA anadiria mas de 2 GB que la demo nunca usa.
DEPS_TORCH = ["torch==2.10.0", "--index-url", "https://download.pytorch.org/whl/cpu"]

# Contenido de la aplicacion. Los .mat originales quedan deliberadamente fuera:
# el cache los sustituye y pesan mas de 100 MB cada uno.
APP_FILES = ["demo_app.py"]
APP_DIRS = ["assets", ".streamlit"]
SRC_MODULES = ["__init__.py", "config.py", "models_cyborg.py", "models_v18.py"]

LAUNCHER = r"""@echo off
title Demo EMG - Trabajo de Grado - Universidad del Valle

REM El directorio de trabajo tiene que ser app\, porque Streamlit busca su
REM configuracion en .streamlit\config.toml relativo al directorio actual. Si se
REM lanza desde otra carpeta, el tema oscuro y el resto de ajustes se ignoran.
cd /d "%~dp0app"

echo.
echo   ================================================================
echo    Clasificacion de gestos EMG con redes hibridas
echo    Trabajo de Grado - EIEE - Universidad del Valle
echo   ================================================================
echo.
echo   Iniciando la demo... la ventana del navegador se abrira sola.
echo   Para cerrar la demo, cierra esta ventana negra.
echo.

REM El archivo python313._pth ya fija las rutas del interprete embebido, que
REM ignora PYTHONHOME y PYTHONPATH: no hay que definirlos aqui.
REM PYTHONNOUSERSITE evita que, en un equipo que ya tenga Python instalado,
REM la demo cargue por error los paquetes de ese Python en lugar de los suyos.
set PYTHONNOUSERSITE=1
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

"%~dp0python\python.exe" -m streamlit run demo_app.py ^
    --server.headless=false ^
    --server.port=8501 ^
    --browser.gatherUsageStats=false

echo.
echo   La demo se ha cerrado. Pulsa una tecla para salir.
pause > nul
"""

README_DIST = """# Demo EMG — paquete portable

Trabajo de Grado · Escuela de Ingenieria Electrica y Electronica · Universidad del Valle

## Como ejecutarla

Doble clic en **Iniciar_Demo.bat**. El navegador se abre solo en
`http://localhost:8501`. Para cerrarla, cierra la ventana negra de consola.

No requiere instalar Python, ni conexion a internet, ni permisos de
administrador. La carpeta completa se puede copiar a un USB y ejecutar desde
cualquier equipo con Windows de 64 bits.

## Si el navegador no se abre solo

Abre manualmente `http://localhost:8501`.

## Si Windows muestra un aviso de SmartScreen

Es un archivo `.bat` de texto plano, no un ejecutable firmado. Pulsa
"Mas informacion" y luego "Ejecutar de todas formas". Puedes abrir el `.bat`
con el Bloc de notas para verificar que solo lanza Python.

## Si el puerto 8501 esta ocupado

Edita `Iniciar_Demo.bat` con el Bloc de notas y cambia `--server.port=8501`
por otro numero, por ejemplo `8600`.

## Antes de la sustentacion

- Abre la demo una vez en el equipo y con el proyector que vas a usar: el primer
  arranque carga los modelos y deja todo en cache.
- La interfaz esta pensada para 1280x720 o mas. Con menos ancho el navegador
  apila las columnas y sigue siendo utilizable, pero se lee mejor a pantalla
  completa (tecla F11).
- Los controles principales — regla de decision y umbral — estan en el cuerpo de
  la pagina, no en una barra lateral, precisamente para que no queden ocultos en
  resoluciones bajas.

## Contenido

- `python/` — Python embebido con las dependencias de la demo
- `app/` — codigo de la aplicacion y recursos
- `app/data/demo_cache.npz` — conjunto de prueba precomputado (S1, repeticion 6)
- `app/checkpoints/demo/` — pesos de los 4 modelos del ensemble
"""


def log(step: str, message: str) -> None:
    """Imprime un paso del proceso de construccion."""
    print(f"[{step}] {message}", flush=True)


def download(url: str, dest: Path) -> None:
    """Descarga un archivo mostrando su tamano final."""
    log("descarga", f"{url}")
    with urllib.request.urlopen(url, timeout=180) as response, dest.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    log("descarga", f"-> {dest.name} ({dest.stat().st_size/1e6:.1f} MB)")


def setup_embedded_python() -> None:
    """Descarga y prepara la distribucion embebida de Python.

    El paquete embebido trae un archivo `pythonXY._pth` que desactiva el sistema
    de rutas habitual: hay que anadir `Lib\\site-packages` y habilitar `import
    site` para que pip y las dependencias sean visibles.
    """
    if PY_DIR.exists():
        shutil.rmtree(PY_DIR)
    PY_DIR.mkdir(parents=True)

    zip_path = DIST / "python-embed.zip"
    download(PY_EMBED_URL, zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(PY_DIR)
    zip_path.unlink()

    pth_files = list(PY_DIR.glob("python*._pth"))
    if not pth_files:
        raise RuntimeError("La distribucion embebida no contiene el archivo ._pth esperado")
    pth = pth_files[0]
    # La primera entrada debe ser el zip de la biblioteca estandar (python313.zip);
    # sin ella el interprete no encuentra el modulo `encodings` y no arranca.
    pth.write_text(
        "\n".join([
            f"{pth.stem}.zip",
            ".",
            "Lib\\site-packages",
            "..\\app",
            "",
            "import site",
            "",
        ]),
        encoding="utf-8",
    )
    log("python", f"Python {PY_VERSION} embebido preparado en {PY_DIR}")

    bootstrap_pip()


def has_pip() -> bool:
    """Indica si el Python embebido tiene pip disponible."""
    resultado = subprocess.run(
        [str(PY_DIR / "python.exe"), "-m", "pip", "--version"],
        capture_output=True, env={**os.environ, "PYTHONNOUSERSITE": "1"},
    )
    return resultado.returncode == 0


def bootstrap_pip() -> None:
    """Instala pip en el Python embebido mediante get-pip.

    pip vive dentro de `Lib/site-packages`, de modo que cualquier limpieza de esa
    carpeta se lo lleva por delante. Por eso se comprueba y se reinstala en vez
    de darlo por hecho.
    """
    get_pip = DIST / "get-pip.py"
    download(GET_PIP_URL, get_pip)
    subprocess.run(
        [str(PY_DIR / "python.exe"), str(get_pip), "--no-warn-script-location"],
        check=True, env={**os.environ, "PYTHONNOUSERSITE": "1"},
    )
    get_pip.unlink()
    log("python", "pip instalado")


def pip_install(args: list[str]) -> None:
    """Instala paquetes dentro del Python embebido, aislado del anfitrion.

    El interprete embebido con `import site` habilitado tambien ve el
    directorio de paquetes de usuario del Python anfitrion. Sin aislarlo, pip
    da por satisfechos requisitos que en realidad no estan en el paquete y el
    resultado no funciona en otro equipo. `--target` fuerza el destino,
    `--ignore-installed` evita que los detecte como presentes y
    PYTHONNOUSERSITE saca el site de usuario del anfitrion de sys.path.
    """
    target = PY_DIR / "Lib" / "site-packages"
    env = {**os.environ, "PYTHONNOUSERSITE": "1"}
    cmd = [
        str(PY_DIR / "python.exe"), "-m", "pip", "install",
        "--target", str(target),
        "--ignore-installed",
        "--no-warn-script-location",
        *args,
    ]
    subprocess.run(cmd, check=True, env=env)


def install_deps(with_torch: bool) -> None:
    """Instala las dependencias de la demo en el Python embebido."""
    log("deps", "Instalando dependencias base (puede tardar varios minutos)...")
    pip_install(DEPS_BASE)
    if with_torch:
        log("deps", "Instalando PyTorch (variante CPU, ~200 MB)...")
        pip_install(DEPS_TORCH)
    log("deps", "Dependencias instaladas")


def copy_deps(with_torch: bool) -> None:
    """Copia las dependencias desde el entorno actual, sin usar la red.

    Alternativa para construir el paquete sin conexion. Solo es fiable si el
    Python anfitrion es de la misma version mayor.menor y arquitectura que el
    embebido, porque los modulos compilados no son intercambiables entre ellas.
    """
    host = f"{sys.version_info.major}.{sys.version_info.minor}"
    target_version = ".".join(PY_VERSION.split(".")[:2])
    if host != target_version:
        raise RuntimeError(
            f"El Python anfitrion es {host} y el embebido {target_version}. "
            "Las extensiones compiladas no son compatibles: usa el modo con pip."
        )

    site_packages = Path(sysconfig.get_paths()["purelib"])
    target = PY_DIR / "Lib" / "site-packages"
    target.mkdir(parents=True, exist_ok=True)

    paquetes = ["streamlit", "plotly", "numpy", "pandas", "sklearn", "scipy",
                "matplotlib", "altair", "pyarrow", "PIL", "tornado", "click",
                "blinker", "cachetools", "gitdb", "git", "jinja2", "markupsafe",
                "narwhals", "packaging", "protobuf", "google", "pydeck",
                "tenacity", "toml", "typing_extensions", "watchdog", "dateutil",
                "pytz", "tzdata", "attr", "attrs", "jsonschema", "referencing",
                "rpds", "six", "smmap", "joblib", "threadpoolctl", "cycler",
                "kiwisolver", "pyparsing", "fontTools", "contourpy"]
    if with_torch:
        paquetes += ["torch", "torchgen", "functorch", "sympy", "mpmath",
                     "networkx", "filelock", "fsspec"]

    copiados = 0
    for nombre in paquetes:
        origen = site_packages / nombre
        if origen.is_dir():
            shutil.copytree(origen, target / nombre, dirs_exist_ok=True)
            copiados += 1
        else:
            modulo = site_packages / f"{nombre}.py"
            if modulo.exists():
                shutil.copy2(modulo, target / modulo.name)
                copiados += 1
    # Los .dist-info son necesarios: streamlit consulta su propia version.
    for info in site_packages.glob("*.dist-info"):
        shutil.copytree(info, target / info.name, dirs_exist_ok=True)
    log("deps", f"{copiados} paquetes copiados desde el entorno actual")


def copy_app(with_torch: bool) -> None:
    """Copia el codigo y los recursos de la aplicacion al paquete."""
    if APP_DIR.exists():
        shutil.rmtree(APP_DIR)
    APP_DIR.mkdir(parents=True)

    for archivo in APP_FILES:
        shutil.copy2(ROOT / archivo, APP_DIR / archivo)

    for carpeta in APP_DIRS:
        origen = ROOT / carpeta
        if origen.exists():
            shutil.copytree(origen, APP_DIR / carpeta)

    src_dst = APP_DIR / "src"
    src_dst.mkdir()
    for modulo in SRC_MODULES:
        origen = ROOT / "src" / modulo
        if origen.exists():
            shutil.copy2(origen, src_dst / modulo)

    (APP_DIR / "data").mkdir()
    cache = ROOT / "data" / "demo_cache.npz"
    if not cache.exists():
        raise FileNotFoundError(
            "Falta data/demo_cache.npz. Generalo antes con:  python -m src.build_demo_cache (repositorio clasificador-semg)"
        )
    shutil.copy2(cache, APP_DIR / "data" / "demo_cache.npz")

    if with_torch:
        ckpt_src = ROOT / "checkpoints" / "demo"
        if not ckpt_src.exists():
            raise FileNotFoundError(
                "Falta checkpoints/demo/. Generalo con:  python -m src.build_demo_cache (repositorio clasificador-semg)"
            )
        ckpt_dst = APP_DIR / "checkpoints" / "demo"
        ckpt_dst.mkdir(parents=True)
        for pth in ckpt_src.glob("*.pth"):
            shutil.copy2(pth, ckpt_dst / pth.name)

    loro = ROOT / "checkpoints" / "loro" / "loro_cv_summary.json"
    if loro.exists():
        destino = APP_DIR / "checkpoints" / "loro"
        destino.mkdir(parents=True, exist_ok=True)
        shutil.copy2(loro, destino / loro.name)

    log("app", f"Aplicacion copiada a {APP_DIR}")


def write_launcher() -> None:
    """Escribe el lanzador .bat y el README del paquete."""
    (DIST / "Iniciar_Demo.bat").write_text(LAUNCHER, encoding="utf-8")
    (DIST / "LEEME.md").write_text(README_DIST, encoding="utf-8")
    log("app", "Lanzador Iniciar_Demo.bat y LEEME.md escritos")


def trim_package() -> None:
    """Elimina del paquete lo que con seguridad no hace falta en tiempo de ejecucion.

    La poda es deliberadamente conservadora. Patrones genericos como `**/testing`
    o `**/include` parecen inocuos pero rompen el paquete: `numpy/testing` es un
    submodulo real del que depende scipy al importarse. Aqui solo se eliminan las
    cabeceras de C++ y las bibliotecas de enlace de PyTorch — necesarias para
    compilar extensiones, nunca para inferir — junto con los bytecodes y simbolos
    de depuracion.
    """
    site_packages = PY_DIR / "Lib" / "site-packages"
    if not site_packages.exists():
        return

    objetivos: list[Path] = [
        site_packages / "torch" / "include",
        site_packages / "functorch" / "csrc",
    ]
    # Los __pycache__ se conservan a proposito: sin ellos, el primer arranque en
    # el equipo de la sala tendria que recompilar todo Streamlit y PyTorch.
    objetivos += list(site_packages.rglob("*.pdb"))
    objetivos += list((site_packages / "torch" / "lib").glob("*.lib")) \
        if (site_packages / "torch" / "lib").exists() else []

    liberado = 0
    for ruta in objetivos:
        if not ruta.exists():
            continue
        try:
            if ruta.is_dir():
                liberado += sum(f.stat().st_size for f in ruta.rglob("*") if f.is_file())
                shutil.rmtree(ruta)
            else:
                liberado += ruta.stat().st_size
                ruta.unlink()
        except (OSError, PermissionError):
            continue
    log("limpieza", f"{liberado/1e6:.0f} MB eliminados del paquete")


def verify_package(with_torch: bool) -> None:
    """Comprueba que el paquete arranca por si mismo antes de darlo por bueno.

    Importa las dependencias y carga el cache con el interprete embebido y el
    site de usuario desactivado, que es exactamente la situacion del equipo de
    la sustentacion. Sin esta comprobacion, un recorte demasiado agresivo o una
    dependencia ausente solo se descubririan al abrir la demo delante del jurado.

    Raises:
        RuntimeError: Si el paquete no supera la comprobacion.
    """
    modulos = ["numpy", "scipy", "pandas", "sklearn", "plotly", "streamlit", "matplotlib"]
    if with_torch:
        modulos.append("torch")

    # Las versiones instaladas deben coincidir con las fijadas: si pip resolvio
    # otra cosa, el paquete no es el que se probo y hay que saberlo aqui, no en
    # medio de la sustentacion.
    esperadas = {
        spec.split("==")[0].replace("scikit-learn", "sklearn"): spec.split("==")[1]
        for spec in DEPS_BASE if "==" in spec
    }
    if with_torch:
        esperadas["torch"] = DEPS_TORCH[0].split("==")[1]

    script = "\n".join([
        "import sys",
        f"for m in {modulos!r}:",
        "    __import__(m)",
        f"esperadas = {esperadas!r}",
        "for mod, version in esperadas.items():",
        "    real = __import__(mod).__version__.split('+')[0]",
        "    assert real == version, f'{mod}: se esperaba {version} y hay {real}'",
        "print(f'OK versiones fijadas: {len(esperadas)} paquetes')",
        "import numpy as np",
        "from sklearn.metrics import balanced_accuracy_score",
        "d = np.load('data/demo_cache.npz')",
        "acc = float((d['probs_mean'].argmax(1) == d['labels']).mean() * 100)",
        "assert 70.0 < acc < 75.0, f'accuracy inesperada en el cache: {acc}'",
        "print(f'OK ventanas={len(d[\"labels\"])} accuracy={acc:.2f}%')",
    ])
    if with_torch:
        script += "\n".join([
            "",
            "import torch",
            "from src.models_cyborg import CyborgHybrid",
            "m = CyborgHybrid(manual_features_dim=48)",
            "ck = torch.load('checkpoints/demo/best_model_cyborg.pth',",
            "                weights_only=False, map_location='cpu')",
            "m.load_state_dict(ck['model_state_dict']); m.eval()",
            "with torch.no_grad():",
            "    out = m(torch.from_numpy(d['signals'][:1]), torch.from_numpy(d['feat48'][:1]))",
            "print('OK inferencia en vivo, logits', tuple(out.shape))",
        ])

    log("verificacion", "Ejecutando el paquete de forma aislada...")
    resultado = subprocess.run(
        [str(PY_DIR / "python.exe"), "-c", script],
        cwd=str(APP_DIR),
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
        capture_output=True, text=True,
    )
    if resultado.returncode != 0:
        print(resultado.stdout)
        print(resultado.stderr, file=sys.stderr)
        raise RuntimeError(
            "El paquete portable no supera la comprobacion. Revisa el error de arriba "
            "antes de llevarlo a la sustentacion."
        )
    for linea in resultado.stdout.strip().splitlines():
        log("verificacion", linea)


def folder_size_mb(path: Path) -> float:
    """Tamano total de una carpeta en megabytes."""
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--copy-deps", action="store_true",
                        help="Copia las dependencias del entorno actual en vez de usar pip")
    parser.add_argument("--no-torch", action="store_true",
                        help="Excluye PyTorch; la demo usara las probabilidades del cache")
    parser.add_argument("--skip-python", action="store_true",
                        help="Reutiliza el Python embebido ya descargado")
    parser.add_argument("--app-only", action="store_true",
                        help="Solo recopia el codigo de la demo y vuelve a verificar, "
                             "sin tocar el interprete ni las dependencias")
    args = parser.parse_args()

    with_torch = not args.no_torch
    DIST.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("PAQUETE PORTABLE DE LA DEMO — Trabajo de Grado UV")
    print("=" * 64)
    print(f"Destino:  {DIST}")
    print(f"PyTorch:  {'incluido (inferencia en vivo)' if with_torch else 'excluido (solo cache)'}")
    print(f"Origen:   {'entorno actual' if args.copy_deps else 'pip'}")
    print()

    if args.app_only:
        if not (PY_DIR / "python.exe").exists():
            raise SystemExit(
                "No hay un paquete previo que actualizar: ejecuta el script sin --app-only."
            )
        log("python", "Modo --app-only: interprete y dependencias intactos")
    else:
        if not args.skip_python or not (PY_DIR / "python.exe").exists():
            setup_embedded_python()
        else:
            log("python", "Reutilizando el Python embebido existente")
            if not args.copy_deps and not has_pip():
                log("python", "pip no esta presente en el interprete embebido; reinstalando")
                bootstrap_pip()

        if args.copy_deps:
            copy_deps(with_torch)
        else:
            install_deps(with_torch)

    copy_app(with_torch)
    write_launcher()
    trim_package()

    # Precompilar deja los .pyc listos: el primer arranque en la sala no tiene que
    # compilar Streamlit y PyTorch, que son decenas de miles de modulos.
    log("compilacion", "Generando bytecode para acelerar el primer arranque...")
    compileall.compile_dir(str(APP_DIR), quiet=2, force=True)
    compileall.compile_dir(str(PY_DIR / "Lib" / "site-packages"), quiet=2, workers=0)

    verify_package(with_torch)

    print()
    print("=" * 64)
    print("PAQUETE LISTO")
    print("=" * 64)
    print(f"  Carpeta:  {DIST}")
    print(f"  Tamano:   {folder_size_mb(DIST):.0f} MB")
    print(f"  Lanzador: {DIST / 'Iniciar_Demo.bat'}")
    print()
    print("  Pruebalo ahora con doble clic en Iniciar_Demo.bat, y copia la")
    print("  carpeta DemoEMG completa al USB de la sustentacion.")


if __name__ == "__main__":
    main()
