def modele_2(py,grad):
    import matplotlib.pyplot as plt

    name=['GradientBoosting','Pytorch Linear Model']
    score=[grad,py]
    plt.figure(figsize=(10, 6))
    plt.bar(name, score, color='green')

    for i, value in enumerate(score):
        plt.text(i, value + 0.5, f"{value:.2f}%", ha='center', va='bottom', fontsize=10, color='black')

    plt.title("Score accuracy par modèle")
    plt.xlabel("Modèle")
    plt.ylabel("Accuracy (%)")
    plt.tight_layout()
    plt.show()
