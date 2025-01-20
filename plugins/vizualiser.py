def show_dataset(features):
  import matplotlib.pyplot as plt
  import seaborn as sb
  for y in features:
    plt.figure(figsize=(10,6))
    sb.countplot(x=y, data=features)
    plt.xlabel(y)
    plt.ylabel("frequency")
    plt.show()

def heatmap(features):
  import matplotlib.pyplot as plt
  import seaborn as sb
  corr=features.corr()
  plt.figure(figsize=(16,10))
  sb.heatmap(corr, annot=True)
  plt.title('Correlation heatmap')
  plt.show()