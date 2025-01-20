#!/bin/bash
if [ "$#" -ne 1 ]; then
    echo "Utilisation : $0 <chemin_vers_fichier_csv>"
    exit 1
fi

CSV_PATH=$1

python main.py && python Extension.py "$CSV_PATH"
