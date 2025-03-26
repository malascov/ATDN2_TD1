#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

df = pd.read_csv(r"C:\Users\Omar\Downloads\rendement_mais.csv")
df


# In[10]:


# 2.1 Mesures de tendance centrale pour 'rendement_t/ha'
rendement = df['RENDEMENT_T_HA']

moyenne = rendement.mean()
mediane = rendement.median()
mode = rendement.mode().iloc[0]

print("Moyenne du rendement :", moyenne)
print("Mediane du rendement :", mediane)
print("Mode du rendement :", mode)


# In[11]:


# 2.2 Mesures de dispersion pour 'rendement_t_ha'
ecart_type = rendement.std()
variance = rendement.var()
etendue = rendement.max() - rendement.min()

print("ecart-type du rendement :", ecart_type)
print("variance du rendement :", variance)
print("etendue du rendement :", etendue)


# In[18]:


# 2.3 Visualisations des données

#histogramme du rendement
plt.figure(figsize=(8, 4))
sns.histplot(rendement, kde=True)
plt.title("Histogramme du rendement (t/ha)")
plt.xlabel("Rendement (t/ha)")
plt.ylabel("Fréquence")
plt.show()

# Histogramme des precipitations
plt.figure(figsize=(8, 4))
sns.histplot(df['PRECIPITATIONS_MM'], kde=True, color='skyblue')
plt.title("Histogramme des précipitations (mm)")
plt.xlabel("Précipitations (mm)")
plt.ylabel("Fréquence")
plt.show()

#histogramme de la temperature
plt.figure(figsize=(8, 4))
sns.histplot(df['TEMPERATURE_C'], kde=True, color='salmon')
plt.title("Histogramme de la température (°C)")
plt.xlabel("Température (°C)")
plt.ylabel("Fréquence")
plt.show()

#boxplot pour identifier les outliers pour le rendement
plt.figure(figsize=(6, 4))
sns.boxplot(y=rendement)
plt.title("Boxplot du rendement (t/ha)")
plt.ylabel("Rendement (t/ha)")
plt.show()

#boxplot pour les precipitations
plt.figure(figsize=(6, 4))
sns.boxplot(y=df['PRECIPITATIONS_MM'], color='skyblue')
plt.title("Boxplot des précipitations (mm)")
plt.ylabel("Précipitations (mm)")
plt.show()

#boxplot pour la temperature
plt.figure(figsize=(6, 4))
sns.boxplot(y=df['TEMPERATURE_C'], color='salmon')
plt.title("Boxplot de la température (°C)")
plt.ylabel("Température (°C)")
plt.show()


# In[19]:


# 2.4 Calcul et visualisation de la matrice de corrélation
# selection des variables numériques
df_numeric = df.select_dtypes(include=[np.number])
correlation_matrix = df_numeric.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matrice de corrélation entre variables numériques")
plt.show()

# Analyse des impacts :
# On peut observer dans la matrice de corrélation les coefficients entre le rendement et les autres variables.
# Les variables présentant des coefficients de corrélation absolus plus élevés avec 'rendement_t_ha'
# sont potentiellement les plus influentes.


# In[23]:


# 3.1 - 3.2 
# Supposons que le DataFrame est déjà chargé dans df et que les colonnes
# s'appellent 'type_sol' et 'rendement_t/ha'. Adaptez les noms si nécessaire.

#selection des rendements pour chaque type de sol
rendement_argileux = df[df['TYPE_SOL'] == 'Argileux']['RENDEMENT_T_HA']
rendement_sableux  = df[df['TYPE_SOL'] == 'Sableux']['RENDEMENT_T_HA']
rendement_limoneux = df[df['TYPE_SOL'] == 'Limoneux']['RENDEMENT_T_HA']

#realisation du test ANOVA
f_stat, p_value = stats.f_oneway(rendement_argileux, rendement_sableux, rendement_limoneux)

print("Statistique F :", f_stat)
print("p-value :", p_value)


# In[29]:


# 4.1 
from sklearn.model_selection import train_test_split
df_clean = df.dropna(subset=['RENDEMENT_T_HA', 'SURFACE_HA', 'ENGRAIS_KG_HA', 'PRECIPITATIONS_MM', 'TEMPERATURE_C', 'TYPE_SOL'])

df_encoded = pd.get_dummies(df_clean, columns=['TYPE_SOL'], drop_first=True)
X = df_encoded.drop('RENDEMENT_T_HA', axis=1)
y = df_encoded['RENDEMENT_T_HA']

#80% entraînement, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# In[30]:


#4.2 Création et entraînement des modèles
# regression lineaire  : 
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)


# In[31]:


# random forest : 
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)


# In[36]:


#4.3 evaluation  
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Pour la regression lineaire
mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)

# Pour random forest
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

print("Modèle Régression Linéaire :")
print("MAE :", mae_lr)
print("RMSE :", rmse_lr)
print("R² :", r2_lr)

print("\nModèle Forêt Aléatoire :")
print("MAE :", mae_rf)
print("RMSE :", rmse_rf)
print("R² :", r2_rf)


# le modèle de forêt aléatoire serait plus performant car R² et rae sont plus petit 

# In[ ]:




