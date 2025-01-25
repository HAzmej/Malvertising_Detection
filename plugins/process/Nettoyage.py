def nettoie(dataset):
  dataset = dataset.drop(columns=['ad_screenshot'])
  dataset['HasObfuscation'] = dataset['HasObfuscation'].astype(int)

  return dataset