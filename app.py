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

import analisis as an
import base_datos as bd
from modelos_nn import VARIABLES, cargar_predictores

CARPETA_ART = Path(__file__).parent / "artefactos"
NO_APLICA = "No aplica (menor de 20)"
NO_RESPONDE = "No responde"
NO_RESPONDER = "Prefiero no responder"
COLORES = ["#0F2E4A", "#C8372D", "#E39B3A", "#7FA7C9", "#5B6B7A"]

st.set_page_config(page_title="Tamizaje de presión arterial", page_icon="🩺", layout="wide")
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9})


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
    st.subheader("Opciones técnicas")
    nombre_modelo = st.selectbox(
        "Red neuronal", nombres, index=nombres.index(principal),
        help="El modelo principal es el de mejor AUC en validación.")
    predictor = predictores[nombre_modelo]
    st.caption(
        f"Umbral de decisión: {predictor.umbral:.2f}. Si la probabilidad de la red es igual o mayor, "
        f"la app muestra «riesgo alto». Se eligió con validación para detectar al menos el "
        f"{config['recall_minimo']:.0%} de los casos.")
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
    ["🔎 Evaluar riesgo", "📈 Desempeño del modelo", "🗄️ Datos", "🕘 Historial"])


# ---------------------------------------------------------------------------
# 1) Evaluar riesgo
# ---------------------------------------------------------------------------
def opciones(columna, edad):
    """Categorías aprendidas por el modelo, adaptadas a la edad, con opción de no responder."""
    if columna in ("educacion", "estado_civil") and edad < 20:
        return [NO_APLICA]
    cats = [c for c in categorias[columna] if c not in (NO_RESPONDE, NO_APLICA)]
    return [NO_RESPONDER] + cats


def a_valor(respuesta):
    """'Prefiero no responder' -> vacío: el modelo lo completa como en el entrenamiento."""
    return np.nan if respuesta == NO_RESPONDER else respuesta


with tab_riesgo:
    st.subheader("Datos principales")
    st.caption("La edad, el sexo, el IMC y la cintura concentran casi todo el poder predictivo del modelo "
               "(ver *Desempeño del modelo → ROC e importancia*).")
    col1, col2 = st.columns(2)
    with col1:
        edad = st.number_input("Edad (años)", min_value=18, max_value=80, value=45,
                               help="El modelo se entrenó con adultos de 18 a 80 años.")
        sexo = st.radio("Sexo", ["Hombre", "Mujer"], horizontal=True)
        cintura = st.number_input("Circunferencia de la cintura (cm)", 50.0, 180.0, 92.0, step=0.5,
                                  help="Medida a la altura del ombligo.")
    with col2:
        calcular_imc = st.toggle("Calcular el IMC con peso y estatura", value=True)
        if calcular_imc:
            c_peso, c_estatura = st.columns(2)
            peso = c_peso.number_input("Peso (kg)", 30.0, 250.0, 75.0, step=0.5)
            estatura = c_estatura.number_input("Estatura (cm)", 120.0, 220.0, 168.0, step=0.5)
            imc = round(peso / (estatura / 100) ** 2, 1)
            st.info(f"IMC calculado: **{imc}**")
        else:
            imc = st.number_input("Índice de masa corporal (IMC)", 12.0, 70.0, 26.0, step=0.1)

    with st.expander("Más datos (opcionales)"):
        st.caption("Estos datos casi no cambian el resultado. Si no los responde, el modelo los completa "
                   "a partir de personas parecidas del entrenamiento, igual que hizo con los datos faltantes "
                   "de la encuesta.")
        o1, o2 = st.columns(2)
        with o1:
            fuma = o1.selectbox("¿Ha fumado al menos 100 cigarrillos en su vida?", [NO_RESPONDER, "Sí", "No"])
            alcohol = o1.selectbox("¿Ha tomado 12 o más tragos en algún año?", [NO_RESPONDER, "Sí", "No"])
            educacion = o1.selectbox("Nivel educativo", opciones("educacion", edad))
            estado_civil = o1.selectbox("Estado civil", opciones("estado_civil", edad))
        with o2:
            con_hogar = o2.checkbox("Indicar personas en el hogar")
            personas = o2.slider("Personas que viven en su hogar", 1, 7, 3, disabled=not con_hogar,
                                 help="En la encuesta, 7 significa 7 o más.")
            con_ingreso = o2.checkbox("Indicar ingreso familiar")
            ingreso = o2.slider("Ingreso familiar frente a la línea de pobreza", 0.0, 5.0, 2.5, 0.1,
                                disabled=not con_ingreso,
                                help="1 = ingreso igual a la línea de pobreza de EE. UU.; "
                                     "5 = cinco veces o más (tope de la encuesta).")

    entrada = {"edad": int(edad), "imc": float(imc), "cintura_cm": float(cintura),
               "ratio_ingreso_pobreza": float(ingreso) if con_ingreso else np.nan,
               "personas_hogar": int(personas) if con_hogar else np.nan,
               "sexo": sexo, "fuma": a_valor(fuma), "alcohol": a_valor(alcohol),
               "educacion": a_valor(educacion), "estado_civil": a_valor(estado_civil)}
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
        imc_r = r["entrada"]["imc"]
        categoria_imc = ("bajo peso" if imc_r < 18.5 else "peso normal" if imc_r < 25
                         else "sobrepeso" if imc_r < 30 else "obesidad")
        if r["resultado"] == "Elevada":
            st.error("### Riesgo alto de presión arterial elevada\n"
                     "Sus datos se parecen a los de personas que tenían la presión arterial elevada "
                     "(130/80 mmHg o más). **Le recomendamos medirse la presión** en un centro de salud o "
                     "una farmacia y, si sale alta, consultar con un profesional de la salud.")
        else:
            st.success("### Riesgo bajo de presión arterial elevada\n"
                       "Sus datos no muestran señales de alerta. Aun así, la presión alta muchas veces no da "
                       "síntomas: **se recomienda medirla al menos una vez al año.**")
        st.markdown(f"**Su IMC:** {imc_r} ({categoria_imc}, según la clasificación de la OMS).")
        st.caption("Este resultado es una orientación, no un diagnóstico. La única forma de saber si tiene la "
                   "presión alta es medirla.")

        with st.expander("Detalles técnicos del resultado"):
            st.caption("Información para el equipo y el docente; no está pensada para el paciente.")
            m1, m2, m3 = st.columns(3)
            m1.metric(f"Probabilidad ({r['modelo']})", f"{r['prob']:.0%}")
            m2.metric("Regresión logística (ACA 1)", f"{r['prob_lr']:.0%}",
                      delta=f"{(r['prob_lr'] - r['prob']) * 100:+.0f} pp frente a la red", delta_color="off")
            m3.metric("Umbral de alerta", f"{r['umbral']:.0%}")
            st.caption("La app muestra «riesgo alto» cuando la probabilidad de la red es igual o mayor que el "
                       "umbral. Durante el entrenamiento se dio más peso a los casos de presión elevada, por lo "
                       "que la probabilidad sirve para ordenar el riesgo, no como un porcentaje literal.")
            st.markdown("**¿Cómo cambia la probabilidad si cambia el IMC o la edad?**")
            base = pd.DataFrame([r["entrada"]])[VARIABLES]
            rango_imc = np.arange(18, 45.5, 0.5)
            sim_imc = pd.concat([base.assign(imc=v) for v in rango_imc], ignore_index=True)
            e1, e2 = st.columns(2)
            e1.line_chart(pd.DataFrame({"Probabilidad": predictor.predict_proba(sim_imc)},
                                       index=pd.Index(rango_imc, name="IMC")))
            if r["entrada"]["edad"] >= 20:   # no se mezclan menores con categorías de adultos
                rango_edad = np.arange(20, 81)
                sim_edad = pd.concat([base.assign(edad=int(v)) for v in rango_edad], ignore_index=True)
                e2.line_chart(pd.DataFrame({"Probabilidad": predictor.predict_proba(sim_edad)},
                                           index=pd.Index(rango_edad, name="Edad")))
            st.caption("Se dejan fijos los demás datos y solo se varía una medida.")


# ---------------------------------------------------------------------------
# 2) Desempeño del modelo
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Calculando probabilidades en el conjunto de prueba…")
def probabilidades_prueba(nombre):
    X, y = bd.datos_prueba()
    if nombre in predictores:
        prob = predictores[nombre].predict_proba(X)
    else:
        prob = lr["modelo"].predict_proba(X[VARIABLES])[:, 1]
    return np.asarray(y), np.asarray(prob)


def matriz_confusion(ax, vn, fp, fn, vp, titulo):
    matriz = np.array([[vn, fp], [fn, vp]]).astype(int)
    total_fila = matriz.sum(axis=1, keepdims=True)
    ax.imshow(matriz, cmap="Blues")
    for (i, j), v in np.ndenumerate(matriz):
        pct = v / total_fila[i, 0] if total_fila[i, 0] else 0
        ax.text(j, i, f"{v}\n({pct:.0%})", ha="center", va="center", fontsize=11,
                color="white" if v > matriz.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Normal", "Elevada"]); ax.set_yticks([0, 1], ["Normal", "Elevada"])
    ax.set_xlabel("Predicción"); ax.set_ylabel("Real"); ax.set_title(titulo, fontsize=10)
    for lado in ax.spines.values():
        lado.set_visible(False)


with tab_modelo:
    met = bd.metricas("prueba")
    met_val = bd.metricas("validacion")
    modelos = met["modelo"].tolist()
    nombre_lr = next((m for m in modelos if "logíst" in m.lower()), None)
    color_de = {m: COLORES[i % len(COLORES)] for i, m in enumerate(
        sorted(modelos, key=lambda m: (m != nombre_modelo, m != nombre_lr, m)))}
    fila_red = met[met["modelo"] == nombre_modelo].iloc[0]
    fila_lr = met[met["modelo"] == nombre_lr].iloc[0] if nombre_lr else None

    st.subheader(f"{nombre_modelo}: resultados en el conjunto de prueba")
    st.caption(f"Prueba: {int(fila_red[['vp', 'vn', 'fp', 'fn']].sum())} personas que el modelo nunca vio "
               "durante el entrenamiento ni al elegir el umbral. Las flechas comparan con la regresión "
               "logística del ACA 1.")
    k = st.columns(5)
    for col, clave, nombre, inverso in [(k[0], "auc", "ROC-AUC", False), (k[1], "recall", "Recall", False),
                                        (k[2], "especificidad", "Especificidad", False),
                                        (k[3], "f1", "F1", False), (k[4], "fn", "Falsos negativos", True)]:
        valor = fila_red[clave]
        texto = f"{int(valor)}" if clave == "fn" else f"{valor:.3f}"
        delta = None
        if fila_lr is not None:
            d = valor - fila_lr[clave]
            delta = f"{int(d):+d}" if clave == "fn" else f"{d:+.3f}"
        col.metric(nombre, texto, delta=delta, delta_color="inverse" if inverso else "normal")

    if fila_lr is not None:
        traslape = an.ic_se_traslapan(fila_red, fila_lr)
        st.info(
            f"**Lectura:** el AUC de {nombre_modelo} ({fila_red['auc']:.3f}) frente al de la regresión logística "
            f"({fila_lr['auc']:.3f}) difiere en {fila_red['auc'] - fila_lr['auc']:+.3f}, y sus intervalos de "
            f"confianza del 95 % {'se traslapan: la diferencia no es concluyente' if traslape else 'no se traslapan'}. "
            f"Con su umbral ({fila_red['umbral']:.2f}), la red deja pasar {int(fila_red['fn'])} casos de presión "
            f"elevada frente a {int(fila_lr['fn'])} de la regresión logística, a cambio de "
            f"{int(fila_red['fp'] - fila_lr['fp']):+d} falsas alarmas.")

    sub_comp, sub_curvas, sub_umbral, sub_roc = st.tabs(
        ["Comparación de modelos", "Curvas de entrenamiento", "Umbral y errores", "ROC e importancia"])

    # --- Comparación de modelos -------------------------------------------
    with sub_comp:
        vista = met[["modelo", "umbral"] + an.METRICAS + ["fn", "fp"]].copy()
        vista["AUC IC 95 %"] = [f"{a:.3f} – {b:.3f}" for a, b in met[["auc_ic_inf", "auc_ic_sup"]].values]
        vista = vista.rename(columns={"modelo": "Modelo", "umbral": "Umbral", "fn": "Falsos negativos",
                                      "fp": "Falsos positivos", **an.NOMBRES_METRICAS})
        st.dataframe(vista.round(3), hide_index=True, use_container_width=True)

        g1, g2 = st.columns([1.4, 1])
        with g1:
            st.markdown("**Métricas por modelo**")
            fig, ax = plt.subplots(figsize=(6.4, 3.4))
            ancho = 0.8 / len(modelos)
            x = np.arange(len(an.METRICAS))
            for i, m in enumerate(modelos):
                fila = met[met["modelo"] == m].iloc[0]
                ax.bar(x + i * ancho - 0.4 + ancho / 2, [fila[c] for c in an.METRICAS], ancho,
                       label=m, color=color_de[m])
            ax.set_xticks(x, [an.NOMBRES_METRICAS[c] for c in an.METRICAS])
            ax.set_ylim(0, 1); ax.legend(fontsize=7, ncol=2, loc="upper left"); ax.grid(axis="y", alpha=0.3)
            st.pyplot(fig)
            st.caption("El recall alto y la especificidad baja son intencionales: en un tamizaje es peor "
                       "dejar pasar un caso que pedir una medición de más.")
        with g2:
            st.markdown("**AUC con intervalo de confianza del 95 %**")
            orden = met.sort_values("auc")
            fig, ax = plt.subplots(figsize=(4.4, 3.4))
            for i, (_, fila) in enumerate(orden.iterrows()):
                ax.errorbar(fila["auc"], i, xerr=[[fila["auc"] - fila["auc_ic_inf"]], [fila["auc_ic_sup"] - fila["auc"]]],
                            fmt="o", color=color_de[fila["modelo"]], capsize=4, markersize=7)
            ax.set_yticks(range(len(orden)), orden["modelo"]); ax.set_xlabel("ROC-AUC (prueba)")
            ax.grid(axis="x", alpha=0.3)
            st.pyplot(fig)
            st.caption("Si las barras se cruzan, la diferencia entre modelos puede deberse al azar del muestreo "
                       "(bootstrap de 1.000 remuestreos).")

        st.markdown("**Validación frente a prueba** (si los valores son parecidos, el modelo generaliza)")
        comp = met_val[["modelo", "auc", "recall", "especificidad"]].merge(
            met[["modelo", "auc", "recall", "especificidad"]], on="modelo", suffixes=(" validación", " prueba"))
        comp = comp.rename(columns={"modelo": "Modelo"})
        comp.columns = [c.replace("auc", "AUC").replace("recall", "Recall").replace("especificidad", "Especificidad")
                        for c in comp.columns]
        st.dataframe(comp.round(3), hide_index=True, use_container_width=True)

    # --- Curvas de entrenamiento ------------------------------------------
    with sub_curvas:
        redes_hist = bd.modelos_con_historial()
        red_sel = st.selectbox("Modelo", redes_hist,
                               index=redes_hist.index(nombre_modelo) if nombre_modelo in redes_hist else 0)
        hist = bd.historial(red_sel)
        info = an.brecha_mejor_epoca(hist)
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Mejor época", info["mejor_epoca"], help="Pesos conservados por el early stopping.")
        b2.metric("Épocas entrenadas", info["epocas"])
        b3.metric("AUC validación (mejor época)", f"{info['auc_val']:.3f}")
        b4.metric("Brecha de pérdida (val − train)", f"{info['brecha']:+.4f}",
                  help="Cuanto más grande, más memoriza la red los datos de entrenamiento.")

        fig, ejes = plt.subplots(1, 3, figsize=(12, 3.2))
        for ax, metrica, titulo in zip(ejes, ["loss", "acc", "auc"], ["Pérdida (BCE ponderada)",
                                                                      "Exactitud (umbral 0,5)", "ROC-AUC"]):
            ax.plot(hist["epoca"], hist[f"{metrica}_train"], color=COLORES[0], label="Entrenamiento")
            ax.plot(hist["epoca"], hist[f"{metrica}_val"], color=COLORES[1], label="Validación")
            ax.axvline(info["mejor_epoca"], color="gray", ls="--", lw=1, label="Mejor época")
            ax.set_title(titulo, fontsize=10); ax.set_xlabel("Época"); ax.grid(alpha=0.3)
        ejes[0].legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        st.caption("La pérdida de entrenamiento sigue bajando, pero lo que importa es la de validación: cuando "
                   "deja de mejorar, la red empieza a memorizar y el early stopping se queda con la mejor época.")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Pérdida de validación de todas las redes**")
            fig, ax = plt.subplots(figsize=(5.6, 3.2))
            for i, m in enumerate(redes_hist):
                h = bd.historial(m)
                ax.plot(h["epoca"], h["loss_val"], label=m, color=COLORES[(i + 1) % len(COLORES)])
            ax.set_xlabel("Época"); ax.set_ylabel("Pérdida de validación"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
            st.pyplot(fig)
        with c2:
            st.markdown("**Experimento de aumento de datos (ruido gaussiano)**")
            if "MLP" in redes_hist and "MLP sin aumento" in redes_hist:
                con, sin = an.brecha_mejor_epoca(bd.historial("MLP")), an.brecha_mejor_epoca(bd.historial("MLP sin aumento"))
                tabla_aum = pd.DataFrame({
                    "": ["Con ruido", "Sin ruido"],
                    "Mejor época": [con["mejor_epoca"], sin["mejor_epoca"]],
                    "AUC validación": [round(con["auc_val"], 4), round(sin["auc_val"], 4)],
                    "Brecha de pérdida": [round(con["brecha"], 4), round(sin["brecha"], 4)]})
                st.dataframe(tabla_aum, hide_index=True, use_container_width=True)
                reduccion = 1 - con["brecha"] / sin["brecha"] if sin["brecha"] else 0
                st.caption(f"El ruido cambió la brecha entre validación y entrenamiento en {-reduccion:+.0%} "
                           f"y el AUC de validación en {con['auc_val'] - sin['auc_val']:+.4f}. "
                           "Una brecha menor indica menos sobreajuste.")
            else:
                st.caption("No hay historial del experimento con y sin ruido.")

    # --- Umbral y errores ---------------------------------------------------
    with sub_umbral:
        st.caption("Mueva el umbral para ver el intercambio entre detectar más casos (recall) y dar menos falsas "
                   "alarmas (especificidad). Se calcula en vivo sobre el conjunto de prueba guardado en SQLite.")
        opciones_modelo = list(predictores) + ([nombre_lr] if nombre_lr else [])
        u1, u2 = st.columns([1, 2])
        modelo_umbral = u1.selectbox("Modelo a analizar", opciones_modelo,
                                     index=opciones_modelo.index(nombre_modelo))
        umbral_modelo = (predictores[modelo_umbral].umbral if modelo_umbral in predictores
                         else float(lr["umbral"]))
        umbral = u2.slider("Umbral de decisión", 0.05, 0.95, float(round(umbral_modelo, 2)), 0.01,
                           help=f"El umbral elegido en validación para este modelo es {umbral_modelo:.2f}.")
        y_test, p_test = probabilidades_prueba(modelo_umbral)
        m_u = an.metricas_umbral(y_test, p_test, umbral)

        q = st.columns(5)
        q[0].metric("Recall", f"{m_u['recall']:.3f}")
        q[1].metric("Especificidad", f"{m_u['especificidad']:.3f}")
        q[2].metric("Precisión", f"{m_u['precision']:.3f}")
        q[3].metric("Falsos negativos", m_u["fn"], help="Personas con presión alta que el modelo no detecta.")
        q[4].metric("Falsos positivos", m_u["fp"], help="Personas sanas a las que se les recomendaría medirse.")

        v1, v2, v3 = st.columns([1, 1.3, 1.3])
        with v1:
            fig, ax = plt.subplots(figsize=(3.4, 3.2))
            matriz_confusion(ax, m_u["vn"], m_u["fp"], m_u["fn"], m_u["vp"], f"Matriz de confusión (umbral {umbral:.2f})")
            st.pyplot(fig)
        with v2:
            curva = an.curva_umbral(y_test, p_test)
            fig, ax = plt.subplots(figsize=(4.6, 3.2))
            for clave, color in [("recall", COLORES[1]), ("especificidad", COLORES[0]), ("precision", COLORES[2])]:
                ax.plot(curva["umbral"], curva[clave], label=an.NOMBRES_METRICAS[clave], color=color)
            ax.axvline(umbral, color="gray", ls="--", lw=1)
            ax.set_xlabel("Umbral"); ax.set_ylim(0, 1.02); ax.legend(fontsize=8); ax.grid(alpha=0.3)
            ax.set_title("Métricas según el umbral", fontsize=10)
            st.pyplot(fig)
        with v3:
            fig, ax = plt.subplots(figsize=(4.6, 3.2))
            bins = np.linspace(0, 1, 26)
            ax.hist(p_test[y_test == 0], bins=bins, alpha=0.65, color=COLORES[3], label="Presión normal")
            ax.hist(p_test[y_test == 1], bins=bins, alpha=0.65, color=COLORES[1], label="Presión elevada")
            ax.axvline(umbral, color="black", ls="--", lw=1, label="Umbral")
            ax.set_xlabel("Probabilidad estimada"); ax.set_ylabel("Personas"); ax.legend(fontsize=8)
            ax.set_title("Distribución de probabilidades", fontsize=10)
            st.pyplot(fig)
        st.caption("A la derecha del umbral, el modelo da alerta. Las barras rojas a la izquierda son los falsos "
                   "negativos; las azules a la derecha, los falsos positivos. Cuanto menos se mezclan las dos "
                   "distribuciones, mejor discrimina el modelo.")

    # --- ROC e importancia ------------------------------------------------
    with sub_roc:
        r1, r2 = st.columns(2)
        with r1:
            st.markdown("**Curvas ROC – prueba**")
            roc = bd.curva_roc()
            fig, ax = plt.subplots(figsize=(4.8, 4.2))
            for nombre, grupo in roc.groupby("modelo"):
                auc = met.loc[met["modelo"] == nombre, "auc"]
                ax.plot(grupo["fpr"], grupo["tpr"], color=color_de.get(nombre, COLORES[-1]),
                        label=f"{nombre} ({auc.iloc[0]:.3f})" if len(auc) else nombre)
            ax.plot([0, 1], [0, 1], "k--", lw=1)
            ax.set_xlabel("1 − especificidad"); ax.set_ylabel("Recall (sensibilidad)")
            ax.legend(fontsize=7); ax.grid(alpha=0.3)
            st.pyplot(fig)
            st.caption("Cuanto más se acerca la curva a la esquina superior izquierda, mejor ordena el modelo "
                       "el riesgo. La diagonal punteada equivale a adivinar al azar.")
        with r2:
            st.markdown("**Importancia de las variables (permutación)**")
            ruta_imp = CARPETA_ART / "importancia_variables.png"
            if ruta_imp.exists():
                st.image(str(ruta_imp), use_container_width=True)
            st.caption("Se desordena una variable a la vez y se mide cuánto cae el AUC. La edad domina en la red "
                       "y en la regresión logística; le siguen IMC, sexo y cintura. Por eso el formulario pide "
                       "esos cuatro datos y deja el resto como opcional.")


# ---------------------------------------------------------------------------
# 3) Datos
# ---------------------------------------------------------------------------
with tab_datos:
    resumen = bd.resumen_datos()
    d1, d2, d3 = st.columns(3)
    d1.metric("Registros en la encuesta", f"{int(resumen['registros_crudos']):,}".replace(",", "."))
    d2.metric("Con presión medida (usados)", f"{int(resumen['registros_limpios']):,}".replace(",", "."))
    d3.metric("Con presión elevada", f"{resumen['pct_elevada']:.1f} %")
    st.caption("Los datos viven en una base SQLite (`nhanes_presion.db`) dentro de la app. Cada gráfica se "
               "calcula en vivo con una consulta a la tabla `datos_limpios`.")
    for titulo, sql in bd.CONSULTAS_EDA.items():
        st.markdown(f"**Presión elevada {titulo.lower()}**")
        tabla = bd.leer(sql)
        q1, q2 = st.columns([1.4, 1])
        q1.bar_chart(tabla.set_index("grupo")["pct_elevada"], y_label="% con presión elevada",
                     color=COLORES[1])
        q2.dataframe(tabla.rename(columns={"grupo": "Grupo", "personas": "Personas",
                                           "pct_elevada": "% con presión elevada"}),
                     hide_index=True, use_container_width=True)
    with st.expander("Diccionario de datos"):
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
    st.caption("Cada predicción se guarda con un INSERT en la tabla `predicciones`. Los datos opcionales que "
               "no se respondieron quedan vacíos. En Streamlit Community Cloud el disco es temporal, así que el "
               "historial se reinicia cuando la app se vuelve a desplegar.")
