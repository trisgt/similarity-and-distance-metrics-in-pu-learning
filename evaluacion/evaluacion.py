from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, auc

# Función para dar información al usuario sobre la evaluación actual:
def informacion_evaluacion(
        nombre,
        clases_positivas = "-",
        porc_positivos = "-",
        metrica = "-",
        met_neg_fiables = "-",
        met_aprendizaje = "-",
        guardar = False,
        path_guardado = "modelo.txt"):
    
    # Información para el usuario:
    print(f"\nEvaluacion del modelo: \"{nombre}\"")
    print(f"\tClases Positivas (PU): {clases_positivas}")
    print(f"\tPorcentaje de Positivos (PU): {porc_positivos}")
    print(f"\tMetrica Utilizada (PU): {metrica}")
    print(f"\tMetodo de extraccion de Negativos Fiables (PU): {met_neg_fiables}")
    print(f"\tMetodo de aprendizaje: {met_aprendizaje}\n")

    # Si la opción está marcada, guardamos los resultados en un .txt:
    if (guardar):
        f = open(path_guardado, "a")

        # Información para el usuario:
        f.write(f"Evaluacion del modelo: \"{nombre}\"")
        f.write(f"\n\tClases Positivas (PU): {clases_positivas}")
        f.write(f"\n\tPorcentaje de Positivos (PU): {porc_positivos}")
        f.write(f"\n\tMetrica Utilizada (PU): {metrica}")
        f.write(f"\n\tMetodo de extraccion de Negativos Fiables (PU): {met_neg_fiables}")
        f.write(f"\n\tMetodo de aprendizaje: {met_aprendizaje}\n\n")

        print(f"Informacion guardada en \"{path_guardado}\"")


# Función para evaluar los resultados de un modelo tras entrenamiento.
# Se usan precision, recall y F1-Score (accuracy no es relevante):
def evaluacion_dataset(y_true, y_pred, y_score):
    # Para PR-ROC, calculamos los valores de precision y recall a partir de la curva:
    vals_precision, vals_recall, _ = precision_recall_curve(y_true, y_score)

    metricas = {
        #"Precision": precision_score(y_true, y_pred, zero_division = 0), # Ignorar
        #"Recall": recall_score(y_true, y_pred, zero_division = 0), # Ignorar
        "F1-Score": f1_score (y_true, y_pred, zero_division = 0),
        "PR-AUC": auc(vals_recall, vals_precision) # Añadir PR-AUC (No ROC-AUC)
    }

    return metricas


# Función para calcular las medias y desviaciones típicas tras K-fold: