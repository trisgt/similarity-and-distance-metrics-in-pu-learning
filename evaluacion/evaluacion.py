import os
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

# Función para evaluar los resultados de un modelo tras entrenamiento:
def evaluacion_dataset(
        y_true,
        y_pred,
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
    print(f"\tClases Positivas: {clases_positivas}")
    print(f"\tPorcentaje de Positivos: {porc_positivos}")
    print(f"\tMetrica Utilizada: {metrica}")
    print(f"\tMetodo de extraccion de Negativos Fiables: {met_neg_fiables}")
    print(f"\tMetodo de aprendizaje: {met_aprendizaje}\n")

    # Se imprime el reporte:
    reporte = classification_report(y_true, y_pred, zero_division = 0)
    print(reporte)

    # Si la opción está marcada, guardamos los resultados en un .txt:
    if (guardar):
        f = open(path_guardado, "a")

        # Información para el usuario:
        f.write(f"Evaluacion del modelo: \"{nombre}\"")
        f.write(f"\n\tClases Positivas: {clases_positivas}")
        f.write(f"\n\tPorcentaje de Positivos: {porc_positivos}")
        f.write(f"\n\tMetrica Utilizada: {metrica}")
        f.write(f"\n\tMetodo de extraccion de Negativos Fiables: {met_neg_fiables}")
        f.write(f"\n\tMetodo de aprendizaje: {met_aprendizaje}\n\n")

        # Se imprime el reporte:
        f.write(reporte)
        f.write("\n\n")
        f.close()

        print(f"Reporte guardado en \"{path_guardado}\"")

    # Métricas devueltas, utilizadas en el K-fold:
    metricas = {
        "precision": precision_score(y_true, y_pred, zero_division = 0),
        "recall": recall_score(y_true, y_pred, zero_division = 0),
        "f1": f1_score(y_true, y_pred, zero_division = 0),
    }

    return metricas