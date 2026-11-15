"""modelos_nn.py - Arquitecturas neuronales y predictor del proyecto."""
import json
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn

NUMERICAS = ["edad", "imc", "cintura_cm", "ratio_ingreso_pobreza", "personas_hogar"]
CATEGORICAS = ["sexo", "fuma", "alcohol", "educacion", "estado_civil"]
VARIABLES = NUMERICAS + CATEGORICAS

# 1. MLP profundo con embeddings para las variables categóricas

class MLPTabular(nn.Module):
    """Perceptrón multicapa: [numéricas + embeddings] -> capas densas -> 1 logit."""

    def __init__(self, n_num, cardinalidades, emb_dim=4, capas=(64, 32, 16), dropout=0.2):
        super().__init__()
        self.embs = nn.ModuleList(
            [nn.Embedding(c, min(emb_dim, c)) for c in cardinalidades]
        )
        d_in = n_num + sum(e.embedding_dim for e in self.embs)
        bloques, previo = [], d_in
        for h in capas:
            bloques += [nn.Linear(previo, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            previo = h
        bloques.append(nn.Linear(previo, 1))
        self.red = nn.Sequential(*bloques)

    def forward(self, x_num, x_cat):
        partes = [x_num] + [emb(x_cat[:, i]) for i, emb in enumerate(self.embs)]
        return self.red(torch.cat(partes, dim=1)).squeeze(1)


# 2. FT-Transformer (Gorishniy): cada variable se vuelve un token

class TokenizadorNumerico(nn.Module):
    """Convierte cada variable numérica x_j en un vector: x_j * w_j + b_j."""

    def __init__(self, n_num, d):
        super().__init__()
        self.w = nn.Parameter(torch.empty(n_num, d))
        self.b = nn.Parameter(torch.zeros(n_num, d))
        nn.init.normal_(self.w, std=0.02)

    def forward(self, x_num):                      
        return x_num.unsqueeze(-1) * self.w + self.b  


class FTTransformer(nn.Module):
    """Tokens de variables + token [CLS] -> encoder Transformer -> 1 logit."""

    def __init__(self, n_num, cardinalidades, d=32, capas=2, cabezas=4, dropout=0.1):
        super().__init__()
        self.num_tok = TokenizadorNumerico(n_num, d)
        self.cat_embs = nn.ModuleList([nn.Embedding(c, d) for c in cardinalidades])
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.cls, std=0.02)
        capa = nn.TransformerEncoderLayer(
            d_model=d, nhead=cabezas, dim_feedforward=2 * d,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(capa, num_layers=capas, enable_nested_tensor=False)
        self.cabeza = nn.Sequential(nn.LayerNorm(d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, x_num, x_cat):
        tokens = [self.num_tok(x_num)] + [
            emb(x_cat[:, i]).unsqueeze(1) for i, emb in enumerate(self.cat_embs)
        ]
        x = torch.cat(tokens, dim=1)                      
        cls = self.cls.expand(x.size(0), -1, -1)
        x = self.encoder(torch.cat([cls, x], dim=1))       
        return self.cabeza(x[:, 0]).squeeze(1)             


ARQUITECTURAS = {"mlp": MLPTabular, "ft": FTTransformer}


def construir_modelo(tipo, n_num, cardinalidades, hiper):
    return ARQUITECTURAS[tipo](n_num, cardinalidades, **hiper)



# 3. Preprocesamiento -> tensores

def a_tensores(Z, n_num=len(NUMERICAS)):
    """Salida del ColumnTransformer -> (x_num float32, x_cat int64)."""
    
    Z = np.asarray(Z, dtype=np.float32)
    x_num = torch.tensor(Z[:, :n_num], dtype=torch.float32)
    x_cat = torch.tensor(Z[:, n_num:] + 1, dtype=torch.int64)
    return x_num, x_cat


class PredictorPresion:
    """Une preprocesador + red + umbral. Recibe un DataFrame con VARIABLES."""

    def __init__(self, preprocesador, modelo, umbral, nombre=""):
        self.prep = preprocesador
        self.modelo = modelo.eval()
        self.umbral = float(umbral)
        self.nombre = nombre

    @torch.no_grad()
    def predict_proba(self, df):
        x_num, x_cat = a_tensores(self.prep.transform(df[VARIABLES]))
        return torch.sigmoid(self.modelo(x_num, x_cat)).numpy()

    def predict(self, df):
        return (self.predict_proba(df) >= self.umbral).astype(int)


def cargar_predictores(carpeta):
    """Lee artefactos/config.json y devuelve {nombre: PredictorPresion}."""
    carpeta = Path(carpeta)
    config = json.loads((carpeta / "config.json").read_text(encoding="utf-8"))
    prep = joblib.load(carpeta / config["preprocesador"])
    predictores = {}
    for nombre, m in config["modelos"].items():
        red = construir_modelo(m["tipo"], len(NUMERICAS), config["cardinalidades"], m["hiper"])
        red.load_state_dict(torch.load(carpeta / m["archivo"], map_location="cpu"))
        predictores[nombre] = PredictorPresion(prep, red, m["umbral"], nombre)
    return predictores, config
