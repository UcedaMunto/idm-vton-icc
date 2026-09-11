#!/usr/bin/env python3
"""Calculadora reproducible del plan de tiempo del entrenamiento IDM-VTON V14.

Los valores por defecto son los MEDIDOS en la cadena V14 (2026-09-08/09):
  - 42,8 s/paso de media; se usa 43,0 s/paso (redondeo conservador).
  - 500 pasos/bloque -> ~6,0 h por bloque.
  - overhead ~0,15 h por bloque (carga de modelos + guardado + revision).

Uso:
    python3 plan_tiempo_v14.py
    SECONDS_PER_STEP=50 python3 plan_tiempo_v14.py

Variables de entorno:
    SECONDS_PER_STEP, STEPS_PER_BLOCK, OVERHEAD_H_PER_BLOCK, TRAIN_PAIRS,
    BLOCK_DISK_GB, FREE_DISK_GB, MIN_FREE_GIB
"""
import math
import os

SECONDS_PER_STEP = float(os.environ.get("SECONDS_PER_STEP", "43.0"))
STEPS_PER_BLOCK = int(os.environ.get("STEPS_PER_BLOCK", "500"))
OVERHEAD_H_PER_BLOCK = float(os.environ.get("OVERHEAD_H_PER_BLOCK", "0.15"))
TRAIN_PAIRS = int(os.environ.get("TRAIN_PAIRS", "13563"))
BLOCK_DISK_GB = float(os.environ.get("BLOCK_DISK_GB", "5.1"))
FREE_DISK_GB = float(os.environ.get("FREE_DISK_GB", "93"))
MIN_FREE_GIB = float(os.environ.get("MIN_FREE_GIB", "40"))

MEASURED_SECONDS_PER_STEP = 42.8


def miles(n):
    """12345 -> '12.345' (separador de miles es-ES)."""
    return f"{n:,}".replace(",", ".")


def dec(x, decimals=1):
    """1234.5 -> '1.234,5' (miles '.' y decimal ',')."""
    return (
        f"{x:,.{decimals}f}".replace(",", "\x00").replace(".", ",").replace("\x00", ".")
    )


def horas(pasos):
    return pasos * SECONDS_PER_STEP / 3600.0


def bloques(pasos):
    return math.ceil(pasos / STEPS_PER_BLOCK)


def fmt_h(h):
    return f"{dec(h)} h ({dec(h / 24)} d)"


def fila_escenario(nombre, pasos):
    b = bloques(pasos)
    h = horas(pasos)
    total = h + b * OVERHEAD_H_PER_BLOCK
    print(
        f"| {nombre} | {miles(pasos)} | {fmt_h(h)} | {fmt_h(total)} | {b} |"
    )


def main():
    print("=" * 78)
    print("IDM-VTON V14 - plan de tiempo (calculadora)")
    print("=" * 78)
    print(f"s/paso usado          : {dec(SECONDS_PER_STEP)}  (medido: {dec(MEASURED_SECONDS_PER_STEP)})")
    print(f"pasos por bloque      : {miles(STEPS_PER_BLOCK)}")
    print(f"overhead por bloque   : {dec(OVERHEAD_H_PER_BLOCK, 2)} h")
    print(f"pares de train        : {miles(TRAIN_PAIRS)}")
    print(f"1 paso                : {dec(SECONDS_PER_STEP)} s")
    print(f"1 hora                : {dec(3600.0 / SECONDS_PER_STEP)} pasos")
    print(f"1 dia (24 h)          : {miles(int(round(3600.0 * 24 / SECONDS_PER_STEP)))} pasos")
    print(f"bloque de {miles(STEPS_PER_BLOCK):>4}       : {dec(horas(STEPS_PER_BLOCK))} h de computo")
    print()

    print("## Escenarios")
    print("| Objetivo | Pasos | Solo computo | + overhead | Bloques |")
    print("|---|---:|---:|---:|---:|")
    escenarios = [
        ("Smoke", 100),
        ("Piloto", 1_000),
        ("MVP", 5_000),
        ("Produccion corta", 10_000),
        ("Produccion estandar", 20_000),
        ("Produccion larga", 30_000),
        ("1 epoca", TRAIN_PAIRS),
        ("2 epocas", 2 * TRAIN_PAIRS),
        ("Receta original (130 epocas)", 130 * TRAIN_PAIRS),
    ]
    for nombre, pasos in escenarios:
        fila_escenario(nombre, pasos)
    print()

    print("## Por epoca")
    print("| Epocas | Pasos | Solo computo | + overhead |")
    print("|---:|---:|---:|---:|")
    for ep in (1, 2, 5, 10):
        pasos = ep * TRAIN_PAIRS
        b = bloques(pasos)
        h = horas(pasos)
        print(
            f"| {ep} | {miles(pasos)} | {fmt_h(h)} | {fmt_h(h + b * OVERHEAD_H_PER_BLOCK)} |"
        )
    print()

    print("## Disco")
    capacidad = (FREE_DISK_GB - MIN_FREE_GIB) / BLOCK_DISK_GB
    print(f"libres                 : {dec(FREE_DISK_GB)} GiB")
    print(f"minimo de seguridad    : {dec(MIN_FREE_GIB)} GiB")
    print(f"GB por bloque          : {dec(BLOCK_DISK_GB)} GiB")
    print(
        f"bloques sin limpieza   : {int(capacidad)} "
        f"(~{miles(int(capacidad) * STEPS_PER_BLOCK)} pasos)"
    )
    print()
    print("Nota: los valores por defecto son los medidos; ajustar con variables de")
    print("entorno para simular otro hardware o receta.")


if __name__ == "__main__":
    main()
