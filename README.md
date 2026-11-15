# App Streamlit – Tamizaje de presión arterial elevada (ACA 2)

Aplicación que estima la probabilidad de presión arterial elevada (≥ 130/80 mmHg) con redes neuronales: un MLP con embeddings y un FT-Transformer, entrenados con NHANES 2015-2016. Los datos, las métricas y el historial de predicciones viven en una base SQLite.

## Estructura

```
app_presion/
├── app.py                  # interfaz Streamlit
├── base_datos.py           # consultas a SQLite
├── modelos_nn.py           # arquitecturas PyTorch (las mismas del notebook)
├── requirements.txt        # lo genera el notebook con las versiones exactas
├── nhanes_presion.db       # lo genera el notebook
└── artefactos/             # lo genera el notebook
    ├── config.json
    ├── mlp.pt, ft.pt       # pesos de las redes
    ├── preprocesador_nn.joblib
    ├── regresion_logistica.joblib
    └── *.png               # gráficas para el póster
```

## Cómo armar la app

1. Ejecutar completo `ACA_2_Aprendizaje_Automatico.ipynb` en Google Colab. Al final se descarga `app_presion.zip`.
2. El zip ya trae una carpeta `app_presion/`. Descomprimirlo en la carpeta que **contiene** a esta, para que ambas se combinen: deben quedar juntos `app.py`, `base_datos.py`, `artefactos/`, `nhanes_presion.db` y el `requirements.txt` nuevo.
3. Probar en local (opcional):
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```

## Despliegue en Streamlit Community Cloud

1. Subir la carpeta a un repositorio de GitHub (los archivos pesan menos de 2 MB, no requieren Git LFS).
2. En share.streamlit.io: **Create app** → elegir el repositorio → *Main file path*: `app.py` (o `app_presion/app.py` si la carpeta no está en la raíz).
3. En **Advanced settings**, elegir Python 3.12.
4. `requirements.txt` instala PyTorch en versión CPU, que es más liviana que la versión con GPU.

**Nota:** el historial de predicciones se guarda en `nhanes_presion.db`. En Streamlit Cloud el disco es temporal, así que el historial se reinicia cuando la app se vuelve a desplegar.
