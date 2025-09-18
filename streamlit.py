import streamlit as st
import pandas as pd
from joblib import load
import numpy as np

# -------------------------PROCESO DE DESPLIEGUE------------------------------
# En consola:
# pip install scikit-learn==1.3.2

# 01 --------------------------Load the model-------------------------------------------
clf = load('modelo_logistico_pipeline.joblib')

# 02---------------- Variables globales para los campos del formulario-----------------------
work_type_options = ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked']
hypertension_options = [0, 1]
stroke_options = [0, 1]
work_type = ''
age = 0
hypertension = ''
avg_glucose_level = 1.0
bmi = 1.0
stroke = ''


# 03 Reseteo------------- Función para resetear inputs ---------------------------------------
error_flag = False
    
def reset_inputs():
    global work_type, age, hypertension, avg_glucose_level, bmi, stroke, error_flag
    work_type = ''
    age = 0
    hypertension = ''
    avg_glucose_level = 1.0
    bmi = 1.0
    stroke = '' 
    error_flag = False
    
reset_inputs()    
# ------------------------Título centrado-------------------------------------------------
st.title("Modelo Predictivo de Ictus con Regresión Logística")
st.markdown("Este modelo predice la probabilidad de que un paciente sufra un ictus según parámetros como el tipo de trabajo, edad, hipertensión, nivel de glucosa, IMC y antecedentes de ictus.")
st.markdown("---")

# ----------------------- Función para validar los campos del formulario----------------------------
def validate_inputs():
    global error_flag
    if any(val < 0 for val in [age, avg_glucose_level, bmi]):
        st.error("No se permiten valores negativos. Por favor, ingrese valores válidos en todos los campos.")
        error_flag = True
    else:
        error_flag = False

# ------------------------------------ Formulario en dos columnas------------------------------------
with st.form("ictus_form"):
    col1, col2 = st.columns(2)

    # Input fields en la primera columna
    with col1:
        work_type = st.selectbox("**Tipo de Trabajo**", work_type_options)
        age = st.number_input("**Edad**", min_value=0.0, value=float(age), step=1.0) 
        avg_glucose_level = st.number_input("**Nivel promedio de glucosa**", min_value=1.0, value= avg_glucose_level, step= 1.0)
        bmi = st.number_input("**Índice de masa corporal (IMC)**", min_value=1.0, value=bmi, step=1.0)
        
    # Input fields en la segunda columna
    with col2:
        hypertension = st.selectbox("**¿Tiene hipertensión?**", hypertension_options)
        stroke = st.selectbox("**¿Ha sufrido un ictus previamente?**", stroke_options)

    # ----------------------------------------- Boton de Predecir-------------------------------------------------
    predict_button = st.form_submit_button("Predecir")

# Validar entradas SOLO si se presiona "Predecir"
if predict_button and not error_flag:
    # Crear DataFrame
    data = {
    'work_type': [work_type],
    'age': [age],
    'hypertension': [hypertension],
    'avg_glucose_level': [avg_glucose_level],
    'bmi': [bmi],
    'stroke': [stroke] 
    }
    df = pd.DataFrame(data)

    # Realizar predicción
    probabilities_classes = clf.predict_proba(df)[0]

    # Obtener la clase con la mayor probabilidad
    class_predicted = np.argmax(probabilities_classes)

    # Asignar salida y probabilidad según la clase predicha
    # En el script original: #Exited: 0 Cliente retenido;  1 Cliente cerró cuenta
    if class_predicted == 0:
        outcome = "Cliente Retenido"
        probability_churn = probabilities_classes[0]
        style_result = 'background-color: lightgreen; font-size: larger;'
    else:
        outcome = "Churn (Cliente cerró cuenta)"
        probability_churn = probabilities_classes[1]
        style_result = 'background-color: lightcoral; font-size: larger;'

    # Mostrar resultado con estilo personalizado
    result_html = f"<div style='{style_result}'>La predicción fue de clase '{outcome}' con una probabilidad de {round(float(probability_churn), 4)}</div>"
    st.markdown(result_html, unsafe_allow_html=True)

# --------------------------- Boton de Resetear-------------------------------------
if st.button("Resetear"):
    # Resetear inputs
    reset_inputs()

# streamlit run streamlit.py       en la consola
#Coindice con 06_Random_Forest_pipelines.ipynb