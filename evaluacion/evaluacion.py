from sklearn.metrics import f1_score, precision_recall_curve, auc

# Función para dar información al usuario sobre la evaluación actual:
def informacion_evaluacion(
        nombre,
        cl_positivas,
        porc_positivos,
        met_neg_fiables,
        k, porc_neg_fiables,
        metrica,
        met_aprendizaje,
        guardar = False,
        path_guardado = "resultados.txt"):

    # Información para el usuario:
    informacion = (
        f"Evaluacion: {nombre}\n"
        f"\tCl. positivas (PU): {cl_positivas}\n"
        f"\tPorc. positivos (PU): {porc_positivos}\n"
        f"\tMet. negativos fiables (TS): {met_neg_fiables}"
    )

    # "k" solo se utiliza si el método de RN no es Rocchio:
    if met_neg_fiables != "Rocchio":
        informacion += f"\t(K = {k})"


    # El porcentaje de RN solo se utiliza si el método no es Rocchio o CRNE:
    if met_neg_fiables not in ("Rocchio", "CRNE"):
        informacion += f" (Porc. RN: {porc_neg_fiables})"

    # Se añade el resto de la información:
    informacion += (
        f"\n\tMetr. distancia (TS): {metrica}\n"
        f"\tMet. aprendizaje: {met_aprendizaje}\n"
    )

    # Se muestra la información:
    print(f"\n{informacion}")

    # Si la opción está marcada, guardamos los resultados en un .txt:
    if guardar:
        with open(path_guardado, "a") as f:
            f.write("\n" + informacion + "\n")

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