import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("./M1/emissions.csv")
df1=pd.read_csv("./M2/emissions.csv")
df2=pd.read_csv("./M3/emissions.csv")
one_hot_encoder = df.iloc[0, :] 


word2vec = df.iloc[1, :]        
easyocr = df1.iloc[0, :]  
bert = df1.iloc[1, :]  
predict = df2.iloc[0, :]

df_new = pd.DataFrame({
    'OneHotEncoder': one_hot_encoder,
    'Word2Vec': word2vec,
    'EasyOCR': easyocr,
    'BERT': bert
})

################    Emission
plt.figure(figsize=(10, 6))

plt.bar("One Hot encoder", one_hot_encoder['emissions'], label='One Hot encoder (kg CO2eq)', color='blue')
plt.bar("Word2Vec", word2vec['emissions'], label='Word2Vec (kg CO2eq)', color='green')
plt.bar("Easyocr", easyocr['emissions'], label='Easyocr (kg CO2eq))', color='red')
plt.bar("BERT", bert['emissions'], label='BERT (kg CO2eq))', color='black')

plt.title('Émissions de CO2 pour différentes méthodes')
plt.ylabel('Émissions de CO2 (kg)')
plt.xlabel('Index')
plt.legend()
plt.grid(True)
plt.show()


####################    Energy Consumed
plt.figure(figsize=(10, 6))

plt.bar("One Hot encoder", one_hot_encoder['energy_consumed'], label='One Hot encoder (W/h énèrgie)', color='blue')
plt.bar("Word2Vec", word2vec['energy_consumed'], label='Word2Vec (W/h énèrgie)', color='green')
plt.bar("Easyocr", easyocr['energy_consumed'], label='Easyocr (W/h énèrgie)', color='red')
plt.bar("BERT", bert['energy_consumed'], label='BERT (W/h énèrgie)', color='black')


plt.title('Energie Consommée pour différentes méthodes')
plt.ylabel(' Energie Consommée (W/h)')
plt.xlabel('Index')
plt.legend()
plt.grid(True)
plt.show()

##############  Duration
plt.figure(figsize=(10, 6))

plt.bar("One Hot encoder", one_hot_encoder['duration'], color='blue')
plt.bar("Word2Vec", word2vec['duration'] ,color='green')
plt.bar("Easyocr", easyocr['duration'], color='red')
plt.bar("BERT", bert['duration'], color='black')


plt.title('Temps éxècution pour différentes méthodes en s')
plt.ylabel('Temps éxècution (s)')
plt.xlabel('Index')
plt.legend()
plt.grid(True)
plt.show()

########################### Emission_rate
plt.figure(figsize=(10, 6))

plt.bar("One Hot encoder", one_hot_encoder['emissions_rate'], label='One Hot encoder (kg CO2eq)', color='blue')
plt.bar("Word2Vec", word2vec['emissions_rate'], label='Word2Vec (kg CO2eq)', color='green')
plt.bar("Easyocr", easyocr['emissions_rate'], label='Easyocr (kg CO2eq))', color='red')
plt.bar("BERT", bert['emissions_rate'], label='BERT (kg CO2eq))', color='black')

plt.title('Émissions de CO2 pour différentes méthodes')
plt.ylabel('Émissions de CO2 (kg)')
plt.xlabel('Index')
plt.legend()
plt.grid(True)
plt.show()


########################### CPU_energy
plt.figure(figsize=(10, 6))

plt.bar("One Hot encoder", one_hot_encoder['cpu_energy'], label='One Hot encoder (Cpu energy)', color='blue')
plt.bar("Word2Vec", word2vec['cpu_energy'], label='Word2Vec (Cpu energy)', color='green')
plt.bar("Easyocr", easyocr['cpu_energy'], label='Easyocr (Cpu energy))', color='red')
plt.bar("BERT", bert['cpu_energy'], label='BERT (Cpu energy)', color='black')

plt.title('Cpu energy pour différentes méthodes')
plt.ylabel('Cpu energy de CO2 (kg)')
plt.xlabel('Index')
plt.legend()
plt.grid(True)
plt.show()


fig, axs = plt.subplots(3, 2, figsize=(14, 10))

# Émissions de CO2
axs[0, 0].bar("One Hot encoder", one_hot_encoder['emissions'], label='One Hot encoder (kg CO2eq)', color='blue')
axs[0, 0].bar("Word2Vec", word2vec['emissions'], label='Word2Vec (kg CO2eq)', color='green')
axs[0, 0].bar("Easyocr", easyocr['emissions'], label='Easyocr (kg CO2eq)', color='red')
axs[0, 0].bar("BERT", bert['emissions'], label='BERT (kg CO2eq)', color='black')
axs[0, 0].bar("Predict", predict['emissions'], label='PREDICT (kg CO2eq)', color='yellow')
axs[0, 0].set_title('Émissions de CO2 pour différentes méthodes')
axs[0, 0].set_ylabel('Émissions de CO2 (kg)')
axs[0, 0].set_xlabel('Index')
axs[0, 0].legend()
axs[0, 0].grid(True)

# Energie consommée
axs[0, 1].bar("One Hot encoder", one_hot_encoder['energy_consumed'], label='One Hot encoder (W/h énergie)', color='blue')
axs[0, 1].bar("Word2Vec", word2vec['energy_consumed'], label='Word2Vec (W/h énergie)', color='green')
axs[0, 1].bar("Easyocr", easyocr['energy_consumed'], label='Easyocr (W/h énergie)', color='red')
axs[0, 1].bar("BERT", bert['energy_consumed'], label='BERT (W/h énergie)', color='black')
axs[0, 1].bar("Predict", predict['energy_consumed'], label='PREDICT (W/h énergie)', color='yellow')
axs[0, 1].set_title('Énergie Consommée pour différentes méthodes')
axs[0, 1].set_ylabel('Énergie Consommée (W/h)')
axs[0, 1].set_xlabel('Index')
axs[0, 1].legend()
axs[0, 1].grid(True)

# Temps d'exécution
axs[1, 0].bar("One Hot encoder", one_hot_encoder['duration'], color='blue')
axs[1, 0].bar("Word2Vec", word2vec['duration'], color='green')
axs[1, 0].bar("Easyocr", easyocr['duration'], color='red')
axs[1, 0].bar("BERT", bert['duration'], color='black')
axs[1, 0].bar("Predict", predict['duration'], color='yellow')
axs[1, 0].set_title('Temps d\'exécution pour différentes méthodes en secondes')
axs[1, 0].set_ylabel('Temps d\'exécution (s)')
axs[1, 0].set_xlabel('Index')
axs[1, 0].grid(True)

# Taux d'émissions
axs[1, 1].bar("One Hot encoder", one_hot_encoder['emissions_rate'], label='One Hot encoder (kg CO2eq)', color='blue')
axs[1, 1].bar("Word2Vec", word2vec['emissions_rate'], label='Word2Vec (kg CO2eq)', color='green')
axs[1, 1].bar("Easyocr", easyocr['emissions_rate'], label='Easyocr (kg CO2eq)', color='red')
axs[1, 1].bar("BERT", bert['emissions_rate'], label='BERT (kg CO2eq)', color='black')
axs[1, 1].bar("Predict", predict['emissions_rate'], label='PREDICT (kg CO2eq)', color='yellow')
axs[1, 1].set_title('Émissions_rate de CO2 pour différentes méthodes')
axs[1, 1].set_ylabel('Émissions_rate de CO2 (kg)')
axs[1, 1].set_xlabel('Index')
axs[1, 1].legend()
axs[1, 1].grid(True)

# Cpu energy
axs[2, 0].bar("One Hot encoder", one_hot_encoder['cpu_energy'], label='One Hot encoder (cpu_energy)', color='blue')
axs[2, 0].bar("Word2Vec", word2vec['cpu_energy'], label='Word2Vec (cpu_energy)', color='green')
axs[2, 0].bar("Easyocr", easyocr['cpu_energy'], label='Easyocr (cpu_energy)', color='red')
axs[2, 0].bar("BERT", bert['cpu_energy'], label='BERT (cpu_energy)', color='black')
axs[2, 0].bar("Predict", predict['cpu_energy'], label='PREDICT (cpu_energy)', color='yellow')
axs[2, 0].set_title('Cpu energy pour différentes méthodes')
axs[2, 0].set_ylabel('Cpu energy')
axs[2, 0].set_xlabel('Index')
axs[2, 0].legend()
axs[2, 0].grid(True)

plt.tight_layout()
plt.show()