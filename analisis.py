"""
analisis.py - Cálculos de evaluación para la pestaña "Desempeño del modelo".

No depende de Streamlit ni de PyTorch: recibe etiquetas reales y
probabilidades y devuelve tablas listas para mostrar.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

METRICAS = ["accuracy", "precision", "recall", "especificidad", "f1", "auc"]
NOMBRES_METRICAS = {"accuracy": "Accuracy", "precision": "Precisión", "recall": "Recall",
                    "especificidad": "Especificidad", "f1": "F1", "auc": "ROC-AUC"}


def metricas_umbral(y_real, prob, umbral):
    """Matriz de confusión y métricas para un umbral dado."""
    y_real = np.asarray(y_real).astype(int)
    pred = (np.asarray(prob) >= umbral).astype(int)
    vn, fp, fn, vp = confusion_matrix(y_real, pred, labels=[0, 1]).ravel()
    recall = vp / (vp + fn) if vp + fn else 0.0
    precision = vp / (vp + fp) if vp + fp else 0.0
    especificidad = vn / (vn + fp) if vn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"umbral": float(umbral), "accuracy": (vp + vn) / len(y_real), "precision": precision,
            "recall": recall, "especificidad": especificidad, "f1": f1,
            "vp": int(vp), "vn": int(vn), "fp": int(fp), "fn": int(fn)}


def curva_umbral(y_real, prob, desde=0.05, hasta=0.95, paso=0.01):
    """Recall, especificidad, precisión y F1 para cada umbral."""
    umbrales = np.round(np.arange(desde, hasta + paso / 2, paso), 2)
    return pd.DataFrame([metricas_umbral(y_real, prob, u) for u in umbrales])


def ic_se_traslapan(fila_a, fila_b):
    """True si los intervalos de confianza del AUC de dos modelos se traslapan."""
    return not (fila_a["auc_ic_sup"] < fila_b["auc_ic_inf"] or fila_b["auc_ic_sup"] < fila_a["auc_ic_inf"])


def brecha_mejor_epoca(historial):
    """Pérdida y AUC en la mejor época, y la brecha validación - entrenamiento."""
    mejor = int(historial["mejor_epoca"].iloc[0])
    fila = historial.loc[historial["epoca"] == mejor].iloc[0]
    return {"mejor_epoca": mejor, "epocas": int(historial["epoca"].max()),
            "loss_train": fila["loss_train"], "loss_val": fila["loss_val"],
            "brecha": fila["loss_val"] - fila["loss_train"],
            "auc_train": fila["auc_train"], "auc_val": fila["auc_val"]}
