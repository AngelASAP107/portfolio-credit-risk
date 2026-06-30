# Imagen base oficial de Python — versión slim para reducir tamaño
FROM python:3.9-slim

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Instalar dependencia del sistema que necesita LightGBM en Linux
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Copiar dependencias primero — Docker cachea esta capa si requirements.txt no cambia
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código y el modelo
COPY src/ ./src/
COPY models/ ./models/

# Puerto que expone la API
EXPOSE 8000

# Comando que ejecuta el contenedor al arrancar
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
