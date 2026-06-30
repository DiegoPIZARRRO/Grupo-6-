import os
import time
import math
import threading
from contextlib import contextmanager
from datetime import datetime

try:
    import psutil
except ImportError:
    psutil = None

try:
    import resource
except Exception:
    resource = None

registros_rendimiento = []

def _cpu_seconds_total():
    """
    CPU acumulada del proceso actual.
    En sistemas Unix/Linux intenta incluir también procesos hijos,
    útil para cross_val_score con n_jobs=-1.
    """
    total = time.process_time()
    if resource is not None:
        try:
            hijos = resource.getrusage(resource.RUSAGE_CHILDREN)
            total += hijos.ru_utime + hijos.ru_stime
        except Exception:
            pass
    return total

def _rss_mb_arbol_procesos():
    """
    Memoria RSS aproximada del proceso actual y, si existen, procesos hijos.
    """
    if psutil is None:
        return float("nan")
    try:
        proceso = psutil.Process(os.getpid())
        total = proceso.memory_info().rss
        for hijo in proceso.children(recursive=True):
            try:
                total += hijo.memory_info().rss
            except Exception:
                pass
        return total / (1024 ** 2)
    except Exception:
        return float("nan")

def _fmt_num(valor, decimales=3):
    if valor is None:
        return "No aplica"
    try:
        if math.isnan(valor):
            return "No disponible"
    except Exception:
        pass
    return f"{valor:.{decimales}f}"

@contextmanager
def medir_rendimiento(operacion, detalle="", unidades=None, unidad_nombre="registros"):
    """
    Mide el rendimiento de un bloque de código.

    Parámetros:
    - operacion: nombre del punto clave del pipeline.
    - detalle: descripción breve de lo que se ejecuta.
    - unidades: cantidad procesada, cuando aplica.
    - unidad_nombre: nombre de la unidad procesada, por ejemplo registros, filas, folds.
    """
    inicio_fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    t0 = time.perf_counter()
    cpu0 = _cpu_seconds_total()
    mem0 = _rss_mb_arbol_procesos()
    muestras_memoria = []
    if not math.isnan(mem0):
        muestras_memoria.append(mem0)

    detener_muestreo = threading.Event()

    def _muestrear_memoria():
        while not detener_muestreo.is_set():
            mem = _rss_mb_arbol_procesos()
            if not math.isnan(mem):
                muestras_memoria.append(mem)
            time.sleep(0.10)

    hilo = None
    if psutil is not None:
        hilo = threading.Thread(target=_muestrear_memoria, daemon=True)
        hilo.start()

    estado = "OK"
    error = ""
    try:
        yield
    except Exception as exc:
        estado = "ERROR"
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        detener_muestreo.set()
        if hilo is not None:
            hilo.join(timeout=0.3)

        t1 = time.perf_counter()
        cpu1 = _cpu_seconds_total()
        mem1 = _rss_mb_arbol_procesos()

        tiempo_s = t1 - t0
        cpu_s = max(cpu1 - cpu0, 0)
        cpu_pct = (cpu_s / tiempo_s) * 100 if tiempo_s > 0 else float("nan")

        if math.isnan(mem0) or math.isnan(mem1):
            delta_mem = float("nan")
        else:
            delta_mem = mem1 - mem0

        pico_mem = max(muestras_memoria) if muestras_memoria else mem1

        if unidades is not None and unidades != 0:
            try:
                latencia_ms = (tiempo_s / unidades) * 1000
            except Exception:
                latencia_ms = float("nan")
        else:
            latencia_ms = float("nan")

        registros_rendimiento.append({
            "Fecha/hora": inicio_fecha,
            "Operación": operacion,
            "Detalle": detalle,
            "Tiempo ejecución (s)": tiempo_s,
            "CPU proceso aprox. (%)": cpu_pct,
            "CPU proceso (s)": cpu_s,
            "RAM inicial (MB)": mem0,
            "RAM final (MB)": mem1,
            "RAM pico aprox. (MB)": pico_mem,
            "RAM delta (MB)": delta_mem,
            "Unidades procesadas": unidades,
            "Unidad": unidad_nombre,
            "Latencia aprox. (ms/unidad)": latencia_ms,
            "Estado": estado,
            "Log": "Sin errores" if estado == "OK" else error
        })

def actualizar_unidades_operacion(operacion, unidades, unidad_nombre="registros"):
    """
    Permite actualizar la cantidad de unidades procesadas cuando se conoce después de ejecutar el bloque.
    """
    for registro in reversed(registros_rendimiento):
        if registro["Operación"] == operacion:
            registro["Unidades procesadas"] = unidades
            registro["Unidad"] = unidad_nombre
            tiempo_s = registro["Tiempo ejecución (s)"]
            registro["Latencia aprox. (ms/unidad)"] = (tiempo_s / unidades) * 1000 if unidades else float("nan")
            break

def registrar_estabilidad_cv(operacion, scores):
    """
    Agrega indicadores de estabilidad para operaciones de validación cruzada.
    """
    import numpy as np

    scores = np.array(scores)
    media = float(np.mean(scores))
    desviacion = float(np.std(scores))
    minimo = float(np.min(scores))
    maximo = float(np.max(scores))
    variabilidad_pct = (desviacion / media) * 100 if media != 0 else float("nan")

    if variabilidad_pct < 5:
        estabilidad = "Alta: baja variabilidad entre folds"
    elif variabilidad_pct < 10:
        estabilidad = "Media: variabilidad moderada entre folds"
    else:
        estabilidad = "Revisar: alta variabilidad entre folds"

    for registro in reversed(registros_rendimiento):
        if registro["Operación"] == operacion:
            registro["F1 promedio CV"] = media
            registro["F1 desviación CV"] = desviacion
            registro["F1 mínimo CV"] = minimo
            registro["F1 máximo CV"] = maximo
            registro["Variabilidad CV (%)"] = variabilidad_pct
            registro["Estabilidad CV"] = estabilidad
            break