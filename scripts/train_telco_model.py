from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from numpy import mean
from utils.rendimiento import medir_rendimiento, actualizar_unidades_operacion, registrar_estabilidad_cv


##########DEFINICIÓN DE FUNCIONES #####
def calcula_metricas(nombre_algoritmo,y_test,y_pred):
### CLASE POSITIVA (1) ###
    #Calculo la precisión del modelo
    from sklearn.metrics import precision_score
    precision = precision_score(y_test, y_pred)

    #Calculo la exactitud del modelo
    from sklearn.metrics import accuracy_score
    exactitud = accuracy_score(y_test, y_pred)

    #Calculo la sensibilidad del modelo
    from sklearn.metrics import recall_score
    sensibilidad = recall_score(y_test, y_pred)

    #Calculo el Puntaje F1 del modelo
    from sklearn.metrics import f1_score
    puntajef1 = f1_score(y_test, y_pred)

    #Calculo la curva ROC - AUC del modelo
    from sklearn.metrics import roc_auc_score
    roc_auc = roc_auc_score(y_test, y_pred)

### CLASE NEGATIVA (0) ###
    from sklearn.metrics import confusion_matrix
    labels = [1, 0]
    cm = confusion_matrix(y_test, y_pred, labels = labels)
    tp,fn,fp,tn = cm.ravel()

    #True Negative Rate / specificity
    TNR_recall_espec = round(tn/(tn+fp),6)

    #Negative Predictive Value
    NPV_precision = round(tn/(tn+fn),6)

    #Cálculo del F1 para la clase negativa
    F1_neg = round(2 * (NPV_precision * TNR_recall_espec) / (NPV_precision + TNR_recall_espec),2)

    ## Vector de Desempeño - Resumen de Métricas del Modelo
    df_1 = pd.DataFrame({'Modelo': [nombre_algoritmo],'Clase ':[1],'Exactitud': [exactitud],'Precisión': [precision],
                   'Sensibilidad': [sensibilidad], 'F1': [puntajef1],'AUC': [roc_auc]})
    df_2 = pd.DataFrame({'Modelo': '','Clase ':[0],'Exactitud': '','Precisión': [NPV_precision],
                   'Sensibilidad': [TNR_recall_espec], 'F1': [F1_neg],'AUC': ''})
   # df = df_1.append(df_2, ignore_index = True)
    df = pd.concat([df_1, df_2], ignore_index=True)

    return df

def SMOTE_RUS (X_train,y_train,sample_smote,sample_rus):
    import imblearn
    sm = imblearn.over_sampling.SMOTE(sampling_strategy=sample_smote,random_state=11)
    rus = imblearn.under_sampling.RandomUnderSampler(sampling_strategy=sample_rus,random_state=11)
    steps = [('SMOTE', sm), ('RUS', rus)] # Ponemos ambos pasos
    sm_rus = imblearn.pipeline.Pipeline(steps=steps) # Creamos un pipeline para realizar ambas tareas
    X_train_sm_rus, y_train_sm_rus = sm_rus.fit_resample(X_train, y_train)
    return X_train_sm_rus,y_train_sm_rus, sm_rus

def main():

        #Carga de datos depurados
    with medir_rendimiento("Carga de datos", detalle='pd.read_excel("data/processed/telco_limpio.csv")'):
        data = pd.read_excel("data/processed/telco_limpio.csv")

    actualizar_unidades_operacion("Carga de datos", len(data), "filas")

    with medir_rendimiento("Preparación de variables X/y", detalle="Separación de variable objetivo y variables predictoras", unidades=len(data), unidad_nombre="filas"):
        y = data['churn']
        X = data.drop(['churn','customerid','cantidad_features_activas','fecha_procesamiento'], axis = 1)

        #Captura los nombres de las variables independientes
        cols = X.columns

    # Se separa el dataset en entrenamiento y prueba.
    # IMPORTANTE:
    # El conjunto de prueba queda reservado para la evaluación final del modelo.
    # Por eso, la validación cruzada se realizará solamente sobre X_train_raw e y_train_raw.
    with medir_rendimiento("Separación Train/Test", detalle="train_test_split estratificado", unidades=len(X), unidad_nombre="filas"):
        X_train_raw, X_test_raw, y_train_raw, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=0,
            stratify=y
        )

    # IMPORTANTE:
    # El escalador se ajusta solamente con el conjunto de entrenamiento original.
    # Así se evita fuga de datos desde el conjunto de prueba.
    escalar = MinMaxScaler()

    with medir_rendimiento("Escalamiento MinMax", detalle="fit_transform en train y transform en test", unidades=len(X_train_raw) + len(X_test_raw), unidad_nombre="filas"):
        X_train = escalar.fit_transform(X_train_raw)  # aprende mínimos y máximos solo desde entrenamiento
        X_test = escalar.transform(X_test_raw)        # aplica la misma escala al conjunto de prueba

    # Invocar función SMOTE + RUS solamente sobre el conjunto de entrenamiento escalado
    # IMPORTANTE:
    # El balanceo no se aplica sobre X_test, porque el conjunto de prueba debe representar datos no vistos.
    with medir_rendimiento("Balanceo SMOTE + RUS", detalle="SMOTE(sampling_strategy=0.5) + RandomUnderSampler(sampling_strategy=0.75)", unidades=len(X_train), unidad_nombre="filas train originales"):
        X_train, y_train, pipeline_balanceo = SMOTE_RUS(X_train, y_train_raw, 0.5, 0.85)

    actualizar_unidades_operacion("Balanceo SMOTE + RUS", len(y_train), "filas train balanceadas")


    ### ÁRBOL DE DECISIÓN ###

    # Instanciar el clasificador basado en Arboles de decisión
    from sklearn.tree import DecisionTreeClassifier
    tree_clf = DecisionTreeClassifier(max_depth=8,criterion = 'entropy')

    #Entreno el modelo
    with medir_rendimiento("Entrenamiento Decision Tree", detalle="tree_clf.fit(X_train, y_train)", unidades=len(X_train), unidad_nombre="filas train balanceadas"):
        tree_clf.fit(X_train, y_train)

    #Realizo una predicción
    with medir_rendimiento("Predicción Decision Tree", detalle="tree_clf.predict(X_test)", unidades=len(X_test), unidad_nombre="filas test"):
        y_pred = tree_clf.predict(X_test)

    #Obtiene las métricas del modelo
    with medir_rendimiento("Métricas Decision Tree", detalle="precision, accuracy, recall, f1 y matriz de métricas", unidades=len(y_test), unidad_nombre="filas test"):
        df_metricas_tree = calcula_metricas('Tree',y_test,y_pred)

    # Pipeline completo para validación cruzada:
    # En cada fold se ajusta el escalador solo con el subconjunto de entrenamiento del fold.
    # Luego se aplica SMOTE + RUS solo sobre ese subconjunto de entrenamiento.
    # Finalmente se entrena y evalúa el modelo evitando fuga de datos.
    #
    # IMPORTANTE:
    # Como existe un conjunto de prueba final separado, la validación cruzada se realiza
    # solamente sobre X_train_raw e y_train_raw, no sobre X e y completos.
    pipeline_tree = Pipeline([
        ('scaler', MinMaxScaler()),
        ('SMOTE', SMOTE(sampling_strategy=0.5, random_state=11)),
        ('RUS', RandomUnderSampler(sampling_strategy=0.85, random_state=11)),
        ('tree', DecisionTreeClassifier(max_depth=8, criterion='entropy'))
    ])

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

    with medir_rendimiento("Validación cruzada Decision Tree", detalle="RepeatedStratifiedKFold 10x3 con pipeline completo", unidades=30, unidad_nombre="folds"):
        tree_scores = cross_val_score(
            pipeline_tree,
            X_train_raw,
            y_train_raw,
            scoring='f1',
            cv=cv,
            n_jobs=-1
        )

    registrar_estabilidad_cv("Validación cruzada Decision Tree", tree_scores)

    ### REGRESIÓN LOGÍSTICA ###

    #Defino el algoritmo a utilizar
    from sklearn.linear_model import LogisticRegression
    lr_clf = LogisticRegression()

    #Entreno el modelo
    with medir_rendimiento("Entrenamiento Regresión Logística", detalle="lr_clf.fit(X_train, y_train)", unidades=len(X_train), unidad_nombre="filas train balanceadas"):
        lr_clf.fit(X_train, y_train)

    #Realizo una predicción
    with medir_rendimiento("Predicción Regresión Logística", detalle="lr_clf.predict(X_test)", unidades=len(X_test), unidad_nombre="filas test"):
        y_pred = lr_clf.predict(X_test)

    #Obtiene las métricas del modelo
    with medir_rendimiento("Métricas Regresión Logística", detalle="precision, accuracy, recall, f1 y matriz de métricas", unidades=len(y_test), unidad_nombre="filas test"):
        df_metricas_lr = calcula_metricas('Regresión Logística',y_test,y_pred)

    # Pipeline completo para Regresión Logística en validación cruzada.
    # El escalado y el balanceo se realizan dentro de cada fold, evitando fuga de datos.
    #
    # IMPORTANTE:
    # Como existe un conjunto de prueba final separado, la validación cruzada se realiza
    # solamente sobre X_train_raw e y_train_raw, no sobre X e y completos.
    pipeline_lr = Pipeline([
        ('scaler', MinMaxScaler()),
        ('SMOTE', SMOTE(sampling_strategy=0.5, random_state=11)),
        ('RUS', RandomUnderSampler(sampling_strategy=0.85, random_state=11)),
        ('lr', LogisticRegression(max_iter=1000))
    ])

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

    with medir_rendimiento("Validación cruzada Regresión Logística", detalle="RepeatedStratifiedKFold 10x3 con pipeline completo", unidades=30, unidad_nombre="folds"):
        lr_scores = cross_val_score(
            pipeline_lr,
            X_train_raw,
            y_train_raw,
            scoring='f1',
            cv=cv,
            n_jobs=-1
        )

    registrar_estabilidad_cv("Validación cruzada Regresión Logística", lr_scores)

    ### PERCEPTRÓN MULTICAPA ###

    # Perceptrón Multicapa MLP (neuronas, capas)
    # hidden_layer_sizes=(100,) indica una capa oculta con 100 neuronas.
    from sklearn.neural_network import MLPClassifier
    mlp_clf = MLPClassifier(random_state=0, hidden_layer_sizes=(100,), max_iter=1000)

    #Entreno el modelo
    with medir_rendimiento("Entrenamiento MLP", detalle="mlp_clf.fit(X_train, y_train)", unidades=len(X_train), unidad_nombre="filas train balanceadas"):
        mlp_clf.fit(X_train, y_train)

    #Realizo una predicción
    with medir_rendimiento("Predicción MLP", detalle="mlp_clf.predict(X_test)", unidades=len(X_test), unidad_nombre="filas test"):
        y_pred = mlp_clf.predict(X_test)

    #Obtiene las métricas del modelo
    with medir_rendimiento("Métricas MLP", detalle="precision, accuracy, recall, f1 y matriz de métricas", unidades=len(y_test), unidad_nombre="filas test"):
        df_metricas_mlp = calcula_metricas('MLP',y_test,y_pred)

    # Pipeline completo para MLP en validación cruzada.
    # El escalado y el balanceo se realizan dentro de cada fold, evitando fuga de datos.
    #
    # IMPORTANTE:
    # Como existe un conjunto de prueba final separado, la validación cruzada se realiza
    # solamente sobre X_train_raw e y_train_raw, no sobre X e y completos.
    pipeline_mlp = Pipeline([
        ('scaler', MinMaxScaler()),
        ('SMOTE', SMOTE(sampling_strategy=0.5, random_state=11)),
        ('RUS', RandomUnderSampler(sampling_strategy=0.85, random_state=11)),
        ('mlp', MLPClassifier(random_state=0, hidden_layer_sizes=(100,), max_iter=1000))
    ])

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

    with medir_rendimiento("Validación cruzada MLP", detalle="RepeatedStratifiedKFold 10x3 con pipeline completo", unidades=30, unidad_nombre="folds"):
        mlp_scores = cross_val_score(
            pipeline_mlp,
            X_train_raw,
            y_train_raw,
            scoring='f1',
            cv=cv,
            n_jobs=-1
        )

    registrar_estabilidad_cv("Validación cruzada MLP", mlp_scores)

    # Consolida todas las métricas obtenidas antes del proceso de validación cruzada
    with medir_rendimiento("Consolidación métricas modelos", detalle="pd.concat de métricas Tree, LR y MLP", unidades=3, unidad_nombre="modelos"):
        df_metricas_consolidadas = pd.concat(
            [df_metricas_tree, df_metricas_lr, df_metricas_mlp],
            ignore_index=True
        )
        
    # ============================================================
    # 1. IMPRESIÓN DE EVALUACIÓN A PRIORI (TABLA COMPLETA)
    # ============================================================
    print("\n" + "="*80)
    print("EVALUACIÓN A PRIORI (Tabla Consolidada de Métricas)")
    print("="*80)
    print(df_metricas_consolidadas.to_string(index=False))
    print("="*80 + "\n")

    # Guardamos todos los resultados obtenido en los procesos de Cross Validation en una lista
    scores = [tree_scores,lr_scores,mlp_scores]

    #Nombres de los algoritmos utilizados
    modelos = ["Decision Tree","Logistic Regression","Multilayer Perceptron"]

    #Crea diccionario modelos y promedio del score
    with medir_rendimiento("Cálculo F1 promedio por modelo", detalle="np.mean sobre scores de validación cruzada", unidades=3, unidad_nombre="modelos"):
        dict_score_means = dict(
            zip(
                modelos,
                map(
                    np.mean,
                    scores
                )
            )
        )
        df_score_means = pd.DataFrame.from_dict(dict_score_means,orient = 'index',  columns = ['Score mean'])

    # ============================================================
    # 2. SELECCIÓN DE MEJOR MODELO Y VALIDACIÓN CRUZADA
    # ============================================================
    with medir_rendimiento("Selección mejor modelo", detalle="Comparación por F1 promedio de cross validation", unidades=3, unidad_nombre="modelos"):
        f1_tree = np.mean(tree_scores)
        f1_lr = np.mean(lr_scores)
        f1_mlp = np.mean(mlp_scores)

        resultados_f1 = pd.DataFrame({
            "Modelo": ["Decision Tree", "Logistic Regression", "Multilayer Perceptron"],
            "F1 Promedio": [f1_tree, f1_lr, f1_mlp]
        })

    print("RESULTADOS VALIDACIÓN CRUZADA (F1 Promedio a través de 30 iteraciones)")
    print("="*80)
    print(resultados_f1.to_string(index=False))
    print("="*80 + "\n")

    with medir_rendimiento("Asignación clasificador final", detalle="Asignar mejor_clf según modelo con mayor F1", unidades=1, unidad_nombre="modelo seleccionado"):
        mejor_indice = resultados_f1["F1 Promedio"].idxmax()
        mejor_modelo_nombre = resultados_f1.loc[mejor_indice, "Modelo"]
        mejor_f1 = resultados_f1.loc[mejor_indice, "F1 Promedio"]

        pipelines = {
            "Decision Tree": pipeline_tree,
            "Logistic Regression": pipeline_lr,
            "Multilayer Perceptron": pipeline_mlp
        }

        mejor_pipeline = pipelines[mejor_modelo_nombre]

    print(f"---> EL MODELO GANADOR ES: {mejor_modelo_nombre} con un F1 de {mejor_f1:.4f} <--- \n")

    #Entrenamiento final del pipeline con mejor f1
    with medir_rendimiento("Entrenamiento pipeline final", detalle="mejor_pipeline.fit(X_train_raw, y_train_raw)", unidades=len(X_train_raw), unidad_nombre="filas train originales"):
        mejor_pipeline.fit(X_train_raw, y_train_raw)


    # ============================================================
    # 3. SIMULACIÓN DE INFERENCIA CON REGISTRO FICTICIO
    # ============================================================
    print("SIMULACIÓN DE PREDICCIÓN CON VALORES FICTICIOS")
    print("="*80)
    
    # Crear un registro ficticio basado en la estructura de X
    registro_ficticio = X.iloc[[0]].copy()
    registro_ficticio['tenure'] = 2          # Lleva solo 2 meses
    registro_ficticio['monthlycharges'] = 95.0 # Paga muy caro
    registro_ficticio['totalcharges'] = 190.0
    registro_ficticio['contract'] = 0        # Contrato mes a mes (alto riesgo)
    
    print("Registro ficticio ingresado (Cliente de alto riesgo):")
    print(registro_ficticio.to_string(index=False) + "\n")
    
    # Realizar predicción directamente 
    prediccion = mejor_pipeline.predict(registro_ficticio)
    print(f"Clasificación predicha (0=Se queda, 1=Abandona): {prediccion[0]}\n")
    
    # Obtener probabilidades exactas
    probabilidades = mejor_pipeline.predict_proba(registro_ficticio)
    df_probabilidades = pd.DataFrame({
        "Clase CHURN": mejor_pipeline.classes_,
        "Probabilidad": probabilidades[0]
    })
    
    print("Probabilidades exactas de la decisión:")
    print(df_probabilidades.to_string(index=False))
    print("="*80 + "\n")

    # ------------------------------------------------------------
    # Guardar el pipeline en un archivo .joblib
    # ------------------------------------------------------------
    NOMBRE_ARCHIVO = "predictor_churn_pipeline.joblib"
    ARTIFACTS_DIR = Path("artifacts")
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    with medir_rendimiento("Guardado modelo Joblib", detalle=f"joblib.dump(mejor_pipeline, {NOMBRE_ARCHIVO})", unidades=1, unidad_nombre="archivo"):
        joblib.dump(mejor_pipeline, ARTIFACTS_DIR / NOMBRE_ARCHIVO)

if __name__ == "__main__":
    main()