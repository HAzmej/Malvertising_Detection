# Utiliser une image de base Python
FROM python:3.9-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers requis dans le conteneur
COPY M1/MicroService1.py ./MicroService1.py

# Installer les dépendances requises
COPY M1/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port sur lequel FastAPI sera exécuté
EXPOSE 50002

# Lancer l'application
CMD ["python", "MicroService1.py"]
