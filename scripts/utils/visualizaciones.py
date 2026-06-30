def matriz_confusion(y_test, y_pred, nombre_modelo="Modelo", guardar_en=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    matriz = confusion_matrix(y_test, y_pred, labels=[1, 0])

    plt.figure(figsize=(6, 4))

    ax = sns.heatmap(
        matriz,
        annot=True,
        cmap="Blues",
        fmt="g"
    )

    ax.set_title(f"Matriz de Confusión - {nombre_modelo}\n")
    ax.set_xlabel("\nPredicción")
    ax.set_ylabel("Real")

    ax.xaxis.set_ticklabels(["Churn 1", "No churn 0"])
    ax.yaxis.set_ticklabels(["Churn 1", "No churn 0"])

    plt.tight_layout()

    if guardar_en:
        plt.savefig(guardar_en, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def plot_scores(scores, ideal=0.8, nombre_modelo="Modelo", guardar_en=None):
    """
    Dibuja los scores obtenidos en validación cruzada.

    Parámetros:
    - scores: arreglo con los scores de cross validation.
    - ideal: umbral esperado del proyecto.
    - nombre_modelo: nombre del algoritmo evaluado.
    - guardar_en: ruta donde guardar la imagen. Si es None, muestra el gráfico.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    scores = np.array(scores).flatten()

    k = list(range(1, len(scores) + 1))
    valores = scores.tolist()

    fig = plt.figure(figsize=(10, 5))

    plt.plot(k, valores, color="steelblue", label="Score por fold")
    plt.plot(k, [ideal for _ in k], color="mediumblue", label=f"Umbral ideal: {ideal}")
    plt.plot(k, [scores.mean() for _ in k], color="brown", linestyle="dashed", label=f"Promedio: {scores.mean():.6f}")

    plt.xlabel("k-folds")
    plt.ylabel("score")
    plt.suptitle(f"Scores de {nombre_modelo} mediante k-fold estratificado")
    plt.title(f"Score ideal del proyecto F1 = {ideal}")
    plt.legend()
    plt.tight_layout()

    if guardar_en:
        plt.savefig(guardar_en, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
