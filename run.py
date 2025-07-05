#!/usr/bin/env python3
import os
import sys
import subprocess

# Ruta base del proyecto (este archivo está en la raíz)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Script a ejecutar, relativo a la raíz
if len(sys.argv) < 2:
    print("Uso: python run.py <path/al/script.py> [args...]")
    sys.exit(1)

script_to_run = sys.argv[1]
args = sys.argv[2:]

# Comando completo
command = [
    sys.executable,
    os.path.join(PROJECT_ROOT, script_to_run),
    *args
]

# Ejecuta con PYTHONPATH como la raíz del proyecto
env = os.environ.copy()
env["PYTHONPATH"] = PROJECT_ROOT

subprocess.run(command, env=env, cwd=PROJECT_ROOT)
