import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:/Users/homam/Downloads/movies_metadata.csv.zip", low_memory=False)
# Aufgabe 0:

print("Erste 5 Zeilen:")
print(df.head())

print("\nSpaltennamen:")
print(df.columns)


df = df.drop(['belongs_to_collection','homepage','tagline','popularity'],axis=1 ) #Column with a lot of missing values got deleted
print("\nFehlende Werte pro Spalte:")
print(df.isna().sum()) #The missing lines per column were summed


# Aufgabe 1 – Daten bereinigen


#Aufgabe 1:
# budget & revenue in numerische Werte umwandeln
df["budget"] = pd.to_numeric(df["budget"],downcast='integer', errors="coerce")
df["revenue"] = pd.to_numeric(df["revenue"],downcast='integer', errors="coerce")


# Profit berechnen
df["profit"] = df["revenue"] - df["budget"]

# return_ratio (Division durch 0 vermeiden)
df["return_ratio"] = df.apply(
    lambda x: x["revenue"] / x["budget"] if x["budget"] and x["budget"] > 0 # X ist eine Variable die Zeilen in einer Dataframe symbolisiert, die sich wie eine For loop verhählt
    #In den list klammer wird die spalte vorgegeben und das x davor sollen die zeilen sein

    else np.nan, # Wenn das x[budget] kleiner als null ist None(dasselbe wie null)
    axis=1
)
print("/////")

#Aufgabe 2:

df['profit_level'] = df['profit'].apply(
   lambda x: 'niedrig' if x <1 else 'mittel' if x <= 2 else 'hoch' # X mit vielen if müssen mit else abgeschlossen werden um das nächste if zu starten
)
print(df['profit_level'])

df['first_genre'] = df['genres'].apply(
    lambda x: x[0] if isinstance(x,list) and len(x) > 0  else None # with isinstance we can proof the data type, also we check the line if there is a genre he shows it but if none we get nothing back
)
print(df)
