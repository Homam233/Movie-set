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
print("\n")

#Aufgabe 2:

df['profit_level'] = df['profit'].apply(
   lambda x: 'niedrig' if x <1 else 'mittel' if x <= 2 else 'hoch' # X mit vielen if müssen mit else abgeschlossen werden um das nächste if zu starten
)
print(df['profit_level'])
print("\n")
df['first_genre'] = df['genres'].apply(
    lambda x: x[0] if isinstance(x,list) and len(x) > 0  else None # with isinstance we can proof the data type, also we check the line if there is a genre he shows it but if none we get nothing back
)
print(df['first_genre'])
print("\n")
# Source - https://stackoverflow.com/a
# Posted by chrisb, modified by community. See post 'Timeline' for change history
# Retrieved 2025-11-28, License - CC BY-SA 4.0

#df['release_date'] = pd.to_datetime(df['release_date'],format="%Y-%m-%d")
df['release_date'] = df['release_date'].fillna(value='2011-11-10')
g = df[(df['release_date'].str.contains('10395'))]


i = df.index[df['release_date'] == '2011-11-10'].tolist()
print(i)


#print(g)
print("\n")

print(df['release_date'].isnull().sum())
print("Homam")

#df['decade'] = (df['release_date'] // 10)*10 #The New column decade wich is based on the release_date got created
#print(df['decade'])


def top_n_by_profit(df,profit,name):
    return df[[name,profit]].nlargest(10,profit)
print("")
print("\n",top_n_by_profit(df,'profit','title'))

def average_rating_by_genre(df,name,value):
    return df.groupby([name], as_index=False)[value].mean()
print(average_rating_by_genre(df,'genres','revenue'))

class Movie:
    def __init__(self,df,title, release_year, budget, revenue, genres, profit,return_ratio):
        self.df = df
        self.title = title
        self.release_year = release_year
        self.budget = budget
        self.revenue = revenue
        self.genres = genres
        self.profit = profit
        self.return_ratio = return_ratio

    def blockbuster(self):
        x = []
        for i in range(len(self.df['return_ratio'])):
            if self.df['return_ratio'][i] > 5:
                x.append(self.df['return_ratio'])

            else:
                None
        return x


    def summary(self):
        return self.df.info(), self.df.dtypes

print('\n')

class MovieAnalyzer:
    def __init__(self,df):
        self.df = df

    def top_profits(self,name,profit):
        return df[[name,profit]].nlargest(10,profit)

    def rating_by_decade(self):
        return None
    def plot_budget_vs_revenue(self):
        return None


m = MovieAnalyzer

x = m.top_profits(df, 'title', 'profit')
print(x)

print(df.dtypes)
print(df['release_date'])

chosen_columns = ['title','release_date','profit'] # Ausgesuchten spalten in rheinfolge
existing_cols = [col for col in chosen_columns if col in df.columns] #fügt spalten in der gewünschten Rheienfolge in chosen_columns
print(df[existing_cols])




