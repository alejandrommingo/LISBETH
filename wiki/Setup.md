# ⚙️ Guía de Instalación y Configuración

Esta guía te ayudará a configurar el entorno de **LISBETH** en una máquina Linux local.

---

## 📋 Requisitos Previos

Asegúrate de tener instalado:
*   **Python 3.12** o superior.
*   **Git**.
*   **Virtualenv** (recomendado: módulo `venv` nativo).

Verifica tu versión de Python:
```bash
python3 --version
```

---

## 🚀 Paso a Paso

### 1. Clonar el Repositorio

```bash
git clone https://github.com/alejandrommingo/LISBETH.git
cd LISBETH
```

### 2. Crear Entorno Virtual

Es crítico usar un entorno virtual para aislar las dependencias híbridas (Data Engineering + NLP).

```bash
# Crear entorno en la carpeta .venv
python3 -m venv .venv

# Activar el entorno
source .venv/bin/activate
```

### 3. Instalar Dependencias

El proyecto utiliza `pyproject.toml` para definir dependencias. Instálalas con pip:

```bash
pip install -r requirements.txt
# O si prefieres instalar en modo editable:
pip install -e .
```

Si vas a desarrollar o ejecutar tests, instala las dependencias opcionales:

```bash
pip install -e ".[dev]"
```

---

## 🔧 Configuración del Entorno (.env)

El sistema utiliza variables de entorno para configuración sensible o específica del despliegue.

1.  Crea un archivo `.env` en la raíz del proyecto.
2.  (Opcional) Define las siguientes variables si necesitas sobreescribir los defaults:

```ini
# .env example

# Directorio de salida por defecto
LISBETH_OUTPUT_DIR=data/

# Configuración de GDELT
GDELT_MAX_RECORDS=250
REQUEST_TIMEOUT=30

# Dominios permitidos (separados por coma si fuera lista, pero el código lo maneja interno)
# Generalmente se maneja vía domains.py, pero puedes configurar flags de debug aquí.
LOG_LEVEL=INFO
```
*Nota: La mayoría de configuraciones tienen valores por defecto sensatos en `src/news_harvester/config.py`.*

---

## ✅ Verificación

Para asegurar que todo está correctamente instalado:

1.  **Verificar CLI**:
    ```bash
    # Debería mostrar la ayuda del CLI general
    python src/cli.py --help
    ```

2.  **Ejecutar Tests** (Si instalaste dependencias dev):
    ```bash
    pytest
    ```

Si ves los mensajes de ayuda y los tests pasan (o se ejecutan sin errores de importación), ¡estás listo!

---

## 🆘 Solución de Problemas Comunes

**Error: `ModuleNotFoundError: No module named 'src'`**
*   Asegúrate de ejecutar los comandos desde la **raíz** del proyecto (`LISBETH/`).
*   Verifica que `PYTHONPATH` incluya el directorio actual: `export PYTHONPATH=$PYTHONPATH:.`

**Error al instalar `torch` o `transformers`**
*   Estas librerías pueden ser pesadas. Asegúrate de tener `pip` actualizado: `pip install --upgrade pip`.
*   Si usas GPU, verifica la compatibilidad de versiones CUDA en la documentación de PyTorch.
