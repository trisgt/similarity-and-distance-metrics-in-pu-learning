from sklearn.metrics import f1_score, precision_recall_curve, auc

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
    print(f"\nEvaluacion: {nombre}")
    print(f"\tClases Positivas (PU): {clases_positivas}")
    print(f"\tPorcentaje de Positivos (PU): {porc_positivos}")
    print(f"\tMetrica Utilizada (PU): {metrica}")
    print(f"\tMetodo de extraccion de Negativos Fiables (PU): {met_neg_fiables}")
    print(f"\tMetodo de aprendizaje: {met_aprendizaje}\n")

    # Si la opción está marcada, guardamos los resultados en un .txt:
    if guardar:
        with open(path_guardado, "a") as f:
            # Información para el usuario:
            f.write(f"Evaluacion: {nombre}")
            f.write(f"\n\tClases Positivas (PU): {clases_positivas}")
            f.write(f"\n\tPorcentaje de Positivos (PU): {porc_positivos}")
            f.write(f"\n\tMetrica Utilizada (PU): {metrica}")
            f.write(f"\n\tMetodo de extraccion de Negativos Fiables (PU): {met_neg_fiables}")
            f.write(f"\n\tMetodo de aprendizaje: {met_aprendizaje}\n\n")

        print(f"Informacion guardada en \"{path_guardado}\"")


# Función para evaluar los resultados de un modelo tras entrenamiento, con F1-Score y PR-AUC:
def evaluacion_dataset(y_test, y_pred, y_score):
    # Para PR-AUC, calculamos los valores de precision y recall a partir de la curva:
    vals_precision, vals_recall, _ = precision_recall_curve(y_test, y_score)

    metricas = {
        "F1-Score": f1_score(y_test, y_pred, zero_division = 0),
        "PR-AUC": auc(vals_recall, vals_precision)
    }

    return metricas