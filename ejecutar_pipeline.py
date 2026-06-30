import subprocess
import sys

def ejecutar_script(ruta_script):
    print(f"--- Iniciando: {ruta_script} ---")
    try:
        # sys.executable asegura que se use el mismo entorno virtual/Python actual
        # check=True detiene la ejecución lanzando una excepción si el script falla
        subprocess.run([sys.executable, ruta_script], check=True)
        print(f"--- Finalizado con éxito: {ruta_script} ---\n")
    except subprocess.CalledProcessError:
        print(f"*** Error crítico durante la ejecución de {ruta_script} ***")
        print("El pipeline se ha detenido para evitar inconsistencias en la base de datos.")
        sys.exit(1)

def main():
    print("Iniciando pipeline automatizado Telco\n")
    
    # Lista ordenada de las etapas del pipeline
    scripts = [
        "scripts/ingesta.py",
        "scripts/limpieza_transformacion.py",
        "scripts/validacion_estructural_semantica.py",
        "scripts/carga_telco_supabase.py",
        "scripts/train_telco_model.py"
    ]

    for script in scripts:
        ejecutar_script(script)
        
    print("Todas las etapas del pipeline se han ejecutado correctamente.")

if __name__ == "__main__":
    main()