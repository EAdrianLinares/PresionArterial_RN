"""
app.py - Tamizaje de presión arterial elevada con redes neuronales (ACA 2, CUN).

Ejecutar localmente:  streamlit run app.py
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import base_datos as bd
from modelos_nn import VARIABLES, cargar_predictores

CARPETA_ART = Path(__file__).parent / "artefactos"
NO_APLICA = "No aplica (menor de 20)"
NO_RESPONDE = "No responde"

st.set_page_config(page_title="Tamizaje de presión arterial", page_icon="🩺", layout="wide")


@st.cache_resource
def cargar_modelos():
    predictores, config = cargar_predictores(CARPETA_ART)
    lr = joblib.load(CARPETA_ART / config["regresion_logistica"])
    return predictores, config, lr


predictores, config, lr = cargar_modelos()
categorias = config["categorias"]

# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🩺 Presión arterial")
    st.caption("ACA 2 · Aprendizaje Automático · CUN")
    nombres = list(predictores)
    principal = config["modelo_principal"]
    nombre_modelo = st.selectbox(
        "Red neuronal", nombres, index=nombres.index(principal),
        help="El modelo marcado como principal fue el de mejor AUC en validación.")
    predictor = predictores[nombre_modelo]
    st.markdown(
        f"**Umbral de decisión:** {predictor.umbral:.2f}  \n"
        f"Elegido con validación para lograr un recall ≥ {config['recall_minimo']:.0%}: "
        "se prioriza no dejar pasar personas con presión alta.")
    st.divider()
    st.markdown(
        "**Integrantes**  \nEyner Adrián Linares Lopez  \nDaniel Alejandro Roncancio Rodriguez  \n"
        "Oscar Mateo Tautiva Rozo  \nKarol Ginneth Tautiva Rozo")
    st.caption("Docente: Hamilton Rivera Flor")

st.title("Detección temprana de presión arterial elevada")
st.write(
    "Estimación del riesgo de presión arterial ≥ 130/80 mmHg con redes neuronales entrenadas "
    "con la encuesta NHANES 2015-2016 (CDC, EE. UU.). **Es una herramienta de tamizaje, no un "
    "diagnóstico:** la única forma de saberlo es medirse la presión.")

tab_riesgo, tab_modelo, tab_datos, tab_historial = st.tabs(
    ["🔎 Evaluar riesgo", "📈 Desempeño del modelo", "🗄️ Datos (SQL)", "🕘 Historial"])

# ---------------------------------------------------------------------------
# 1) Evaluar riesgo
# ---------------------------------------------------------------------------
def opciones(columna, edad):
    """Categorías aprendidas por el modelo, adaptadas a la edad de la persona."""
    cats = [c for c in categorias[columna] if c != NO_RESPONDE]
    if columna in ("educacion", "estado_civil"):
        if edad < 20:
            return [NO_APLICA]
        cats = [c for c in cats if c != NO_APLICA]
    return cats


def si_no(etiqueta, clave):
    valor = st.radio(etiqueta, ["Sí", "No", "Prefiero no responder"], horizontal=True, key=clave)
    return NO_RESPONDE if valor == "Prefiero no responder" else valor


with tab_riesgo:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Datos personales")
        edad = st.number_input("Edad (años)", min_value=18, max_value=80, value=45,
                               help="El modelo se entrenó con adultos de 18 a 80 años.")
        sexo = st.radio("Sexo", ["Hombre", "Mujer"], horizontal=True)
        educacion = st.selectbox("Nivel educativo", opciones("educacion", edad))
        estado_civil = st.selectbox("Estado civil", opciones("estado_civil", edad))
    with col2:
        st.subheader("Medidas corporales")
        calcular_imc = st.toggle("Calcular el IMC con peso y estatura", value=True)
        if calcular_imc:
            peso = st.number_input("Peso (kg)", 30.0, 250.0, 75.0, step=0.5)
            estatura = st.number_input("Estatura (cm)", 120.0, 220.0, 168.0, step=0.5)
            imc = round(peso / (estatura / 100) ** 2, 1)
            st.info(f"IMC calculado: **{imc}**")
        else:
            imc = st.number_input("Índice de masa corporal (IMC)", 12.0, 70.0, 26.0, step=0.1)
        cintura = st.number_input("Circunferencia de la cintura (cm)", 50.0, 180.0, 92.0, step=0.5,
                                  help="Medida a la altura del ombligo.")
    with col3:
        st.subheader("Hábitos y hogar")
        fuma = si_no("¿Ha fumado al menos 100 cigarrillos en su vida?", "fuma")
        alcohol = si_no("¿Ha tomado 12 o más tragos en algún año?", "alcohol")
        personas = st.slider("Personas que viven en su hogar", 1, 7, 3,
                             help="En la encuesta, 7 significa 7 o más.")
        ingreso = st.slider("Ingreso familiar frente a la línea de pobreza", 0.0, 5.0, 2.5, 0.1,
                            help="1 = ingreso igual a la línea de pobreza de EE. UU.; "
                                 "5 = cinco veces o más (tope de la encuesta).")

    entrada = {"edad": int(edad), "imc": float(imc), "cintura_cm": float(cintura),
               "ratio_ingreso_pobreza": float(ingreso), "personas_hogar": int(personas),
               "sexo": sexo, "fuma": fuma, "alcohol": alcohol,
               "educacion": educacion, "estado_civil": estado_civil}
    df_entrada = pd.DataFrame([entrada])[VARIABLES]

    if st.button("Calcular riesgo", type="primary", use_container_width=True):
        prob = float(predictor.predict_proba(df_entrada)[0])
        prob_lr = float(lr["modelo"].predict_proba(df_entrada)[:, 1][0])
        resultado = "Elevada" if prob >= predictor.umbral else "Normal"
        try:
            bd.guardar_prediccion(entrada, nombre_modelo, prob, predictor.umbral, resultado, prob_lr)
        except Exception as error:   # la predicción se muestra aunque falle el registro
            st.warning(f"No se pudo guardar en el historial: {error}")
        st.session_state["ultimo"] = {"prob": prob, "prob_lr": prob_lr, "resultado": resultado,
                                      "modelo": nombre_modelo, "umbral": predictor.umbral,
                                      "entrada": entrada}

    if "ultimo" in st.session_state:
        r = st.session_state["ultimo"]
        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric(f"Probabilidad ({r['modelo']})", f"{r['prob']:.0%}")
        m2.metric("Regresión logística (ACA 1)", f"{r['prob_lr']:.0%}",
                  delta=f"{(r['prob_lr'] - r['prob']) * 100:+.0f} pp frente a la red", delta_color="off")
        m3.metric("Umbral de alerta", f"{r['umbral']:.0%}")
        st.progress(min(max(r["prob"], 0.0), 1.0))
        if r["resultado"] == "Elevada":
            st.error("**Riesgo alto de presión arterial elevada.** Se recomienda medirse la presión "
                     "en un centro de salud o farmacia y consultar con un profesional.")
        else:
            st.success("**Riesgo por debajo del umbral de alerta.** Aun así, se recomienda medirse "
                       "la presión al menos una vez al año.")

        with st.expander("¿Cómo cambia el riesgo si cambia el IMC o la edad?"):
            base = pd.DataFrame([r["entrada"]])[VARIABLES]
            rango_imc = np.arange(18, 45.5, 0.5)
            sim_imc = pd.concat([base.assign(imc=v) for v in rango_imc], ignore_index=True)
            rango_edad = np.arange(20, 81)
            sim_edad = pd.concat([base.assign(edad=int(v)) for v in rango_edad], ignore_index=True)
            if r["entrada"]["edad"] >= 20:   # no se mezclan menores con categorías de adultos
                e1, e2 = st.columns(2)
                e2.line_chart(pd.DataFrame({"Probabilidad": predictor.predict_proba(sim_edad)},
                                           index=pd.Index(rango_edad, name="Edad")))
            else:
                e1 = st.container()
            e1.line_chart(pd.DataFrame({"Probabilidad": predictor.predict_proba(sim_imc)},
                                       index=pd.Index(rango_imc, name="IMC")))
            st.caption("Se dejan fijos los demás datos y solo se varía una medida.")

# ---------------------------------------------------------------------------
# 2) Desempeño del modelo
# ---------------------------------------------------------------------------
with tab_modelo:
    st.subheader("Métricas en el conjunto de prueba")
    st.caption("La prueba (15 % de los datos) no se usó para entrenar ni para elegir umbrales.")
    met = bd.metricas("prueba")
    vista = met[["modelo", "umbral", "accuracy", "precision", "recall", "especificidad", "f1",
                 "auc", "fn"]].copy()
    vista["AUC IC 95 %"] = [f"{a:.3f} – {b:.3f}" for a, b in met[["auc_ic_inf", "auc_ic_sup"]].values]
    vista = vista.rename(columns={"modelo": "Modelo", "umbral": "Umbral", "accuracy": "Accuracy",
                                  "precision": "Precisión", "recall": "Recall",
                                  "especificidad": "Especificidad", "f1": "F1", "auc": "ROC-AUC",
                                  "fn": "Falsos negativos"})
    st.dataframe(vista.round(3), hide_index=True, use_container_width=True)

    st.subheader("Curvas de aprendizaje")
    redes_hist = bd.modelos_con_historial()
    red_sel = st.selectbox("Modelo", redes_hist,
                           index=redes_hist.index(principal) if principal in redes_hist else 0)
    hist = bd.historial(red_sel).set_index("epoca")
    st.caption(f"Mejor época (pesos conservados): {int(hist['mejor_epoca'].iloc[0])} "
               f"de {len(hist)} entrenadas.")
    c1, c2, c3 = st.columns(3)
    for col, metrica, titulo in [(c1, "loss", "Pérdida"), (c2, "acc", "Exactitud"), (c3, "auc", "ROC-AUC")]:
        col.markdown(f"**{titulo}**")
        col.line_chart(hist[[f"{metrica}_train", f"{metrica}_val"]].rename(
            columns={f"{metrica}_train": "Entrenamiento", f"{metrica}_val": "Validación"}))

    c4, c5 = st.columns(2)
    with c4:
        st.markdown(f"**Matriz de confusión – {red_sel}**")
        fila = met[met["modelo"] == red_sel].iloc[0]
        matriz = np.array([[fila["vn"], fila["fp"]], [fila["fn"], fila["vp"]]]).astype(int)
        fig, eje = plt.subplots(figsize=(4, 3.4))
        eje.imshow(matriz, cmap="Blues")
        for (i, j), v in np.ndenumerate(matriz):
            eje.text(j, i, v, ha="center", va="center",
                     color="white" if v > matriz.max() / 2 else "black", fontsize=13)
        eje.set_xticks([0, 1], ["Normal", "Elevada"]); eje.set_yticks([0, 1], ["Normal", "Elevada"])
        eje.set_xlabel("Predicción"); eje.set_ylabel("Real")
        st.pyplot(fig)
    with c5:
        st.markdown("**Curvas ROC – prueba**")
        roc = bd.curva_roc()
        fig, eje = plt.subplots(figsize=(4.5, 3.4))
        for nombre, grupo in roc.groupby("modelo"):
            auc = met.loc[met["modelo"] == nombre, "auc"]
            eje.plot(grupo["fpr"], grupo["tpr"], label=f"{nombre} ({auc.iloc[0]:.3f})" if len(auc) else nombre)
        eje.plot([0, 1], [0, 1], "k--", lw=1)
        eje.set_xlabel("1 - especificidad"); eje.set_ylabel("Recall"); eje.legend(fontsize=7)
        st.pyplot(fig)

# ---------------------------------------------------------------------------
# 3) Datos (SQL)
# ---------------------------------------------------------------------------
with tab_datos:
    resumen = bd.resumen_datos()
    d1, d2, d3 = st.columns(3)
    d1.metric("Registros en la encuesta", f"{int(resumen['registros_crudos']):,}".replace(",", "."))
    d2.metric("Con presión medida (usados)", f"{int(resumen['registros_limpios']):,}".replace(",", "."))
    d3.metric("Con presión elevada", f"{resumen['pct_elevada']:.1f} %")
    st.caption("Los datos viven en una base SQLite (`nhanes_presion.db`). "
               "Estas consultas se ejecutan en vivo sobre la tabla `datos_limpios`.")
    for titulo, sql in bd.CONSULTAS_EDA.items():
        st.markdown(f"**{titulo}**")
        q1, q2 = st.columns([1, 1.3])
        q1.code(sql.strip(), language="sql")
        q2.bar_chart(bd.leer(sql).set_index("grupo")["pct_elevada"], y_label="% con presión elevada")
    with st.expander("Diccionario de datos (tabla `diccionario`)"):
        st.dataframe(bd.diccionario(), hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
# 4) Historial de predicciones
# ---------------------------------------------------------------------------
with tab_historial:
    res = bd.resumen_predicciones()
    h1, h2, h3 = st.columns(3)
    h1.metric("Predicciones registradas", int(res["total"]))
    h2.metric("Con riesgo alto", int(res["elevadas"]))
    h3.metric("Probabilidad promedio", "-" if pd.isna(res["prob_promedio"]) else f"{res['prob_promedio']:.0%}")
    registros = bd.predicciones_recientes()
    if registros.empty:
        st.info("Aún no hay predicciones. Use la pestaña **Evaluar riesgo**.")
    else:
        st.dataframe(registros, hide_index=True, use_container_width=True)
        b1, b2 = st.columns(2)
        b1.download_button("Descargar CSV", registros.to_csv(index=False).encode("utf-8"),
                           "predicciones.csv", "text/csv")
        if b2.button("Borrar historial"):
            bd.borrar_predicciones()
            st.rerun()
    st.caption("Cada predicción se guarda con un INSERT en la tabla `predicciones`. En Streamlit "
               "Community Cloud el disco es temporal, así que el historial se reinicia cuando la app "
               "se vuelve a desplegar.")
