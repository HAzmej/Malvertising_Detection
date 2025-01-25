def evaluation(models,X_test, y_test):
  from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
  import matplotlib.pyplot as plt
  import time
  from numpy import ravel

  import os
  if not os.path.exists('./plot'):
     os.makedirs('./plot')
  
  best_mod=None
  best_acc=0
  acc_av=[]

  for name, model in models:
    a=time.time()
      

      
    y_pred = model.predict(X_test)
    b=time.time()

    accuracy = accuracy_score(y_test, y_pred)
    acc_av.append(accuracy)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
      
    if accuracy>best_acc:
      best_acc=accuracy
      best_mod=model
      output_file = f"./plot/{name}_{accuracy*100:.2f}_evaluation.txt"

        # Sauvegarder les résultats dans un fichier texte
      with open(output_file, 'w') as f:
        f.write(f"Evaluation results for {name}:\n")
        f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
        f.write(f"Classification Report:\n{report}\n")
        f.write(f"Confusion Matrix:\n{conf_matrix}\n")
        f.write(f"Temps d'execution: {b-a:.2f} seconds\n")
      

    print(f"Results saved in {output_file}")

  print("sauvegarde des performance dans ./plot/[Modele]_[score]_evaluation.txt")
  
  plt.figure(figsize=(10, 6))
  mo = [name for name, model in models]

  acc_av_percent = [round(acc * 100, 2) for acc in acc_av]
  plt.bar(mo, acc_av_percent, color='green')

  for i, value in enumerate(acc_av_percent):
    plt.text(i, value + 0.5, f"{value:.2f}%", ha='center', va='bottom', fontsize=10, color='black')

  plt.title("Score accuracy par modèle")
  plt.xlabel("Modèle")
  plt.ylabel("Accuracy (%)")
  plt.savefig(f'./plot/Comparaison_Evaluation.png')  
  plt.close() 
  print(best_mod)
  return best_mod , best_acc

