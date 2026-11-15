"""
base_datos.py - Acceso a la base SQLite de la app (nhanes_presion.db).

Tablas que crea el notebook: nhanes_crudo, diccionario, datos_limpios,
particiones, metricas_modelos, historial_entrenamiento, curva_roc y
predicciones (vacía, la llena la app).
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

RUTA_DB = Path(__file__).parent / "nhanes_presion.db"

# Consultas del análisis exploratorio que se muestran en la app
CONSULTAS_EDA = {
    "Por grupo de edad": """
SELECT CASE WHEN edad < 30 THEN '18-29' WHEN edad < 40 THEN '30-39' WHEN edad < 50 THEN '40-49'
            WHEN edad < 60 THEN '50-59' WHEN edad < 70 THEN '60-69' ELSE '70-80' END AS grupo,
       COUNT(*) AS personas,
       ROUND(100.0 * AVG(presion_elevada), 1) AS pct_elevada
FROM datos_limpios
GROUP BY grupo
ORDER BY grupo""",
    "Por categoría de IMC": """
SELECT CASE WHEN imc IS NULL THEN '5. Sin dato' WHEN imc < 18.5 THEN '1. Bajo peso'
            WHEN imc < 25 THEN '2. Normal' WHEN imc < 30 THEN '3. Sobrepeso'
            ELSE '4. Obesidad' END AS grupo,
       COUNT(*) AS personas,
       ROUND(100.0 * AVG(presion_elevada), 1) AS pct_elevada
FROM datos_limpios
GROUP BY grupo
ORDER BY grupo""",
    "Por sexo": """
SELECT sexo AS grupo,
       COUNT(*) AS personas,
       ROUND(100.0 * AVG(presion_elevada), 1) AS pct_elevada
FROM datos_limpios
GROUP BY sexo""",
    "Por consumo de tabaco": """
SELECT CASE fuma WHEN 'Sí' THEN 'Ha fumado' WHEN 'No' THEN 'No ha fumado' ELSE 'Sin dato' END AS grupo,
       COUNT(*) AS personas,
       ROUND(100.0 * AVG(presion_elevada), 1) AS pct_elevada
FROM datos_limpios
GROUP BY grupo
ORDER BY personas DESC""",
}

CATEGORICAS = ["sexo", "fuma", "alcohol", "educacion", "estado_civil"]


def conectar():
    return sqlite3.connect(RUTA_DB)


def leer(sql, params=()):
    with conectar() as con:
        return pd.read_sql_query(sql, con, params=params)


def existe_tabla(nombre):
    return not leer("SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?", (nombre,)).empty


def asegurar_tabla_predicciones():
    """Por si la base se copió sin la tabla (el notebook ya la crea)."""
    with conectar() as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS predicciones (
            id INTEGER PRIMARY KEY AUTOINCREMENT, fecha TEXT NOT NULL,
            edad INTEGER, sexo TEXT, imc REAL, cintura_cm REAL, fuma TEXT, alcohol TEXT,
            educacion TEXT, estado_civil TEXT, personas_hogar INTEGER, ratio_ingreso_pobreza REAL,
            modelo TEXT NOT NULL, probabilidad REAL NOT NULL, umbral REAL NOT NULL,
            resultado TEXT NOT NULL, probabilidad_lr REAL)""")


def guardar_prediccion(entrada, modelo, probabilidad, umbral, resultado, probabilidad_lr):
    asegurar_tabla_predicciones()
    columnas = ["edad", "sexo", "imc", "cintura_cm", "fuma", "alcohol", "educacion",
                "estado_civil", "personas_hogar", "ratio_ingreso_pobreza"]
    valores = [None if pd.isna(entrada[c]) else entrada[c] for c in columnas] + [modelo, float(probabilidad), float(umbral),
                                                resultado, float(probabilidad_lr)]
    with conectar() as con:
        con.execute(
            f"INSERT INTO predicciones (fecha, {', '.join(columnas)}, modelo, probabilidad, umbral, "
            f"resultado, probabilidad_lr) VALUES (datetime('now', 'localtime'), "
            f"{', '.join('?' * len(valores))})", valores)


def predicciones_recientes(limite=200):
    asegurar_tabla_predicciones()
    return leer("SELECT * FROM predicciones ORDER BY id DESC LIMIT ?", (limite,))


def resumen_predicciones():
    asegurar_tabla_predicciones()
    return leer("""SELECT COUNT(*) AS total,
                          COALESCE(SUM(resultado = 'Elevada'), 0) AS elevadas,
                          ROUND(AVG(probabilidad), 3) AS prob_promedio
                   FROM predicciones""").iloc[0]


def borrar_predicciones():
    with conectar() as con:
        con.execute("DELETE FROM predicciones")


def metricas(conjunto="prueba"):
    return leer("SELECT * FROM metricas_modelos WHERE conjunto = ? ORDER BY auc DESC", (conjunto,))


def historial(modelo):
    return leer("SELECT * FROM historial_entrenamiento WHERE modelo = ? ORDER BY epoca", (modelo,))


def modelos_con_historial():
    return leer("SELECT DISTINCT modelo FROM historial_entrenamiento")["modelo"].tolist()


def curva_roc():
    return leer("SELECT modelo, fpr, tpr FROM curva_roc")


def diccionario():
    return leer("SELECT codigo, nombre, descripcion, origen, decision FROM diccionario")


def resumen_datos():
    return leer("""SELECT (SELECT COUNT(*) FROM nhanes_crudo) AS registros_crudos,
                          COUNT(*) AS registros_limpios,
                          ROUND(100.0 * AVG(presion_elevada), 1) AS pct_elevada
                   FROM datos_limpios""").iloc[0]


def datos_prueba():
    """Conjunto de prueba (el mismo del notebook) leído desde las tablas
    datos_limpios y particiones. Devuelve (X, y)."""
    datos = leer("""SELECT d.* FROM datos_limpios d
                    JOIN particiones p ON p.SEQN = d.SEQN
                    WHERE p.conjunto = 'test'
                    ORDER BY d.SEQN""")
    datos[CATEGORICAS] = datos[CATEGORICAS].astype(object).where(datos[CATEGORICAS].notna(), np.nan)
    return datos.drop(columns=["SEQN", "presion_elevada"]), datos["presion_elevada"].astype(int)
