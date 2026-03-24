import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
from imblearn.over_sampling import SMOTE

# ==============================================================================
# 1. CONFIGURACIÓN Y CARGA DE DATOS
# ==============================================================================
ruta_csv = 'Ubicacion del la data'
ruta_graficas = 'Ubicacion de las graficas a guardar'

if not os.path.exists(ruta_graficas):
    os.makedirs(ruta_graficas)

def obtener_estacion(mes):
    """Asigna la estación climática basada en el mes (Hemisferio Norte)."""
    if mes in [12, 1, 2]: return 'INVIERNO'
    elif mes in [3, 4, 5]: return 'PRIMAVERA'
    elif mes in [6, 7, 8]: return 'VERANO'
    else: return 'OTONO'

try:
    # Cargamos 250,000 registros para un balance óptimo de datos
    df = pd.read_csv(ruta_csv, low_memory=False, nrows=250000)
    df.columns = [c.lower() for c in df.columns]
    print(f"✅ Dataset cargado correctamente. Registros: {len(df)}")
except FileNotFoundError:
    print(f"❌ Error: No se encontró el archivo CSV en {ruta_csv}"); exit()

# ==============================================================================
# 2. LIMPIEZA E INGENIERÍA DE VARIABLES (GEO + CLIMA + DEMOGRAFÍA)
# ==============================================================================
# Limpiamos nulos en variables críticas
df.dropna(subset=['latitude', 'longitude', 'law_cat_cd', 'age_group', 'perp_sex', 'perp_race'], inplace=True)

# Filtro geográfico estricto para la ciudad de Nueva York
df = df[(df['latitude'] > 40.4) & (df['latitude'] < 40.9) & 
        (df['longitude'] > -74.3) & (df['longitude'] < -73.7)]

# Creación de variables temporales y climáticas
df['arrest_date'] = pd.to_datetime(df['arrest_date'])
df['month'] = df['arrest_date'].dt.month
df['day_of_week'] = df['arrest_date'].dt.dayofweek
df['estacion'] = df['month'].apply(obtener_estacion)

# Target: 1 para Felonía (Grave), 0 para Delito Menor/Violación
df['target'] = (df['law_cat_cd'] == 'F').astype(int)

# ==============================================================================
# 3. CODIFICACIÓN Y PREPARACIÓN DE MATRICES
# ==============================================================================
# Definimos los predictores integrales
features = ['arrest_boro', 'age_group', 'perp_sex', 'perp_race', 'estacion', 
            'latitude', 'longitude', 'month', 'day_of_week']

# One-Hot Encoding: Transforma categorías (Raza, Sexo, Clima) en columnas binarias
X = pd.get_dummies(df[features], drop_first=True)
y = df['target']

# Limpieza de nombres de columnas para compatibilidad total con XGBoost
X.columns = [re.sub(r'[\[\]<>]', '_', str(col)) for col in X.columns]

# División de datos (70% Entrenamiento / 30% Prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Escalamiento de variables numéricas (Geografía y Tiempo)
scaler = StandardScaler()
num_vars = ['latitude', 'longitude', 'month', 'day_of_week']
X_train[num_vars] = scaler.fit_transform(X_train[num_vars])
X_test[num_vars] = scaler.transform(X_test[num_vars])

# Balanceo de Clases: SMOTE genera datos sintéticos para igualar Felonías y Delitos Menores
print("⚖️ Aplicando SMOTE para balancear las clases...")
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

# ==============================================================================
# 4. ENTRENAMIENTO DE LOS 4 MODELOS
# ==============================================================================
print("🤖 Entrenando modelos (esto puede tomar un momento)...")

modelos = {
    "Regr. Logística": LogisticRegression(max_iter=1000, solver='liblinear'),
    "Árbol Decisión": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1),
    "XGBoost": XGBClassifier(n_estimators=100, max_depth=6, eval_metric='logloss')
}

for name, model in modelos.items():
    model.fit(X_res, y_res)

# ==============================================================================
# 5. EVALUACIÓN Y GENERACIÓN DE EVIDENCIAS
# ==============================================================================
# A. Tabla Comparativa de Rendimiento
res_list = []
for name, model in modelos.items():
    y_pred = model.predict(X_test)
    res_list.append({'Modelo': name, 'Accuracy': accuracy_score(y_test, y_pred), 'F1-Score': f1_score(y_test, y_pred)})

df_res = pd.DataFrame(res_list).sort_values(by='F1-Score', ascending=False)
print("\n" + "="*50)
print("📊 RESULTADOS FINALES: GEO + CLIMA + DEMOGRAFÍA")
print("="*50)
print(df_res.to_string(index=False))
print("="*50)

# B. Gráfica de Importancia de Variables (Random Forest)
importancias = pd.Series(modelos["Random Forest"].feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
plt.figure(figsize=(12, 8))
sns.barplot(x=importancias, y=importancias.index, palette='viridis')
plt.title('Influencia de Variables: Geografía, Clima y Demografía (Top 15)')
plt.savefig(os.path.join(ruta_graficas, 'importancia_geo_clima.png'), bbox_inches='tight')
plt.close()

# C. Curva ROC Comparativa
plt.figure(figsize=(10, 7))
for name, model in modelos.items():
    probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc(fpr, tpr):.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.legend(); plt.title('Capacidad de Discriminación: Curva ROC'); plt.grid(alpha=0.3)
plt.savefig(os.path.join(ruta_graficas, 'roc_geo_clima.png'))
plt.close()

# ==============================================================================
# 6. PREDICCIÓN CONTEXTUAL EN VIVO
# ==============================================================================
def predecir_con_todo(boro, edad, sexo, raza, lat, lon, mes, dia):
    """Realiza una predicción cuádruple considerando el clima automático."""
    estacion = obtener_estacion(mes)
    nuevo = pd.DataFrame([[boro, edad, sexo, raza, estacion, lat, lon, mes, dia]], 
                         columns=['arrest_boro', 'age_group', 'perp_sex', 'perp_race', 'estacion', 
                                  'latitude', 'longitude', 'month', 'day_of_week'])
    
    # Preprocesamiento idéntico (Dummies -> Reindex -> Limpiar -> Escalar)
    nuevo_x = pd.get_dummies(nuevo).reindex(columns=X.columns, fill_value=0)
    nuevo_x.columns = [re.sub(r'[\[\]<>]', '_', str(col)) for col in nuevo_x.columns]
    nuevo_x[num_vars] = scaler.transform(nuevo_x[num_vars])
    
    mapa = {1: "🚨 FELONÍA (Grave)", 0: "✅ DELITO MENOR"}
    
    print(f"\n--- 🔎 ANÁLISIS DE PREDICCIÓN (Estación: {estacion}) ---")
    print(f"Sujeto: {raza} | {sexo} | Edad {edad} | Ubicación: ({lat}, {lon})")
    for name, model in modelos.items():
        print(f"{name}: {mapa[model.predict(nuevo_x)[0]]}")

# Ejemplo de prueba: Incidente en Verano (Mes 7) en el Bronx (B)
predecir_con_todo('B', '18-24', 'M', 'BLACK', 40.82, -73.89, 7, 5)

print(f"\n🚀 Proceso completado. Resultados y gráficas listos en: {ruta_graficas}")


# ==============================================================================
# EXPERIMENTO: IMPACTO DEL CLIMA (ENERO VS. JULIO)
# ==============================================================================
print("\n" + "!"*40)
print("🔬 EXPERIMENTO: ¿EL CLIMA CAMBIA LA PREDICCIÓN?")
print("!"*40)

# Datos comunes para la prueba
boro, edad, sexo, raza = 'K', '25-44', 'M', 'BLACK'
lat, lon = 40.65, -73.95 # Punto fijo en Brooklyn
dia = 5 # Sábado

print(f"\n❄️ CASO 1: INVIERNO (Enero - Mes 1)")
predecir_con_todo(boro, edad, sexo, raza, lat, lon, 1, dia)

print(f"\n☀️ CASO 2: VERANO (Julio - Mes 7)")
predecir_con_todo(boro, edad, sexo, raza, lat, lon, 7, dia)