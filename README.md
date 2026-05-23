# Pipeline automatizado para el entrenamiento de un modelo IA.

Este proyecto tiene como objetivo desarrollar pipeline automatizado para entrenar un modelo de IA con multiples fuentes de datos.
---

## Componentes del sistema

- **Archivo CSV**: Dataset a procesar.
- **Scripts de procesamiento**: ingesta, limpieza y transformación, validación de datos y carga a un entorno en SupaBase.
- **Base de datos SupaBase**: para la carga y consulta estructurada del dataset.
- **Documentación**: diseño técnico completo + planificación.

---

## Tecnologías utilizadas

- Python 3  
- Pandas / Scikit-learn  
- SupaBase  
- Docker  
- Git / GitHub  
- Ms Project (planificación)

---

## Pipeline implementado

| Etapa | Descripción |
|-------|-------------|
| 1. Diseño e instalación | Estructura de carpetas, setup del entorno, definición de herramientas |
| 2. Ingesta | Lectura desde CSV y carga a memoria |
| 3. Limpieza | Eliminación de duplicados, tratamiento de nulos, revisión de tipos |
| 4. Transformación | Creación de variables como días sin reposición, tasa de ventas, etc. |
| 5. Validación | Revisión de rangos, tipos, coherencia; validación básica |
| 6. Carga en Supabase | Subida del dataset limpio y validado a la base de datos |

---

## 📂 Estructura del repositorio

```
Grupo-6-/
├── .github/workflows
|   └── ci.yml
├── app/
│   └── __init__.py
|   └── main.py
├── scripts/
│   ├── ingesta.py
│   ├── limpieza_transformacion.py
│   ├── validacion_estructural_semantica.py
│   └── carga_telco_supabase.py
├── data/
│   └── raw/
|     └── telco.csv
├── tests/
│   └── test_health.py
├── .dockerignore
├── .env.example
├── .gitignore
├── Dockerfile
├── README.md
├── render.yaml
├── requirements.txt

```

---

## Cómo ejecutar el sistema (entorno ya instalado)

1. Clonar el repositorio  
   `https://github.com/DiegoPIZARRRO/Grupo-6-`

2. Entrar a la carpeta del proyecto  
   `cd Grupo-6-`
   
3. Crear un archivo .env con el formato de env.example usando tus credenciales de SupaBase                                
   `type nul > .env`

4. Ejecutar cada script del pipiline en el siguiente orden:  
   `python scripts/ingesta.py`  
   `python scripts/limpieza_transformacion.py`  
   `python scripts/validacion_estructural_semantica.py`                
   `python scripts/carga_telco_supabase.py`

---

## Documentación técnica

El documento de diseño técnico está disponible en: 
[Diseño tecnico_ FreshRoute.docx](https://github.com/user-attachments/files/26389107/Diseno.tecnico_.FreshRoute.docx)

---

## Equipo

- Integrante 1 – Benjamín Castañeda
- Integrante 2 – Diego Pizarro  
- Integrante 3 – Ángel Durand


## Validación y limpieza del dataset de Telco

**Fecha de ejecución:** 2026-05-22 13:41:10

### Archivos utilizados

- Dataset de entrada: `data\processed\telco_preprocesado.csv`
- Dataset limpio generado: `data\processed\telco_limpio.csv`
- Reporte de validación: `reports\reporte_errores_validacion.csv`

### Resumen del procesamiento

- Filas originales: 7043
- Filas finales: 7032
- Filas eliminadas: 11
- Problemas registrados en logs: 11

### Transformaciones aplicadas

- Estandarización de nombres de columnas.
- Limpieza y normalización de columnas de texto.
- Conversión de columnas de fecha cuando corresponde.
- Conversión de columnas numéricas cuando corresponde.
- Eliminación de filas vacías.
- Eliminación de registros duplicados.
- Eliminación de registros con valores nulos.
- Validación de rangos permitidos para columnas numéricas relevantes.
- Creación de columnas derivadas cuando existen variables suficientes.
- Generación de reporte de validación para revisar decisiones tomadas durante la limpieza.

### Rol dentro del pipeline de datos

Esta etapa permite mejorar la calidad semántica y técnica del dataset antes de su uso en procesos posteriores,
como carga a base de datos, análisis exploratorio, construcción de modelos o generación de reportes.
Además, deja evidencia reproducible de las reglas aplicadas y de los problemas detectados.
