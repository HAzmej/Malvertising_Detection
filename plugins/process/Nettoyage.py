def nettoie(dataset):
  dataset = dataset.drop(columns=['ad_screenshot','page_screenshot'])

  dataset['page_type'] = dataset['page_type'].replace({
    'home': 0,
    'article':1
    })
  dataset['IsHTTPS'] = dataset['IsHTTPS'].astype(int)
  dataset['HasObfuscation'] = dataset['HasObfuscation'].astype(int)

  return dataset