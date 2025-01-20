def load_plugins(plugin_dir="plugins"):
  """
  Charge dynamiquement tous les plug-ins du dossier spécifié.
  """
  import importlib
  import os
  plugins = {}
  for file in os.listdir(plugin_dir):
    if file.endswith(".py") and file != "__init__.py":
      plugin_name = file[:-3]  # Retirer l'extension .py
      module = importlib.import_module(f"{plugin_dir}.{plugin_name}")
      plugins[plugin_name] = module
  return plugins
