import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# Aufgabe 0 – Daten laden & Überblick
# =====================================================

df = pd.read_csv(
    "C:/Users/homam/Downloads/movies_metadata.csv.zip",
    low_memory=False
)

print("Erste 5 Zeilen:")
print(df.head())

print("\nSpaltennamen:")
print(df.columns)

# Spalten mit vielen fehlenden Werten entfernen
df = df.drop(
    ['belongs_to_collection', 'homepage', 'tagline', 'popularity'],
    axis=1
)

print("\nFehlende Werte pro Spalte:")
print(df.isna().sum())


# =====================================================
# Aufgabe 1 – Daten bereinigen & neue Variablen
# =====================================================

# budget & revenue in numerische Werte umwandeln
df["budget"] = pd.to_numeric(df["budget"], downcast="integer", errors="coerce")
df["revenue"] = pd.to_numeric(df["revenue"], downcast="integer", errors="coerce")

# Profit berechnen
df["profit"] = df["revenue"] - df["budget"]

# return_ratio berechnen (Division durch 0 vermeiden)
df["return_ratio"] = df.apply(
    lambda x: x["revenue"] / x["budget"]
    if x["budget"] and x["budget"] > 0
    else np.nan,
    axis=1
)

print("\n")


# =====================================================
# Aufgabe 2 – Kategorisierung & Genres
# =====================================================

# Profit-Level festlegen
df["profit_level"] = df["profit"].apply(
    lambda x: "niedrig" if x < 1 else "mittel" if x <= 2 else "hoch"
)

print(df["profit_level"])
print("\n")

# Erstes Genre extrahieren (falls Liste vorhanden)
df["first_genre"] = df["genres"].apply(
    lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None
)

print(df["first_genre"])
print("\n")


# =====================================================
# Release Date Bereinigung
# =====================================================

df["release_date"] = df["release_date"].fillna("2011-11-10")

g = df[df["release_date"].str.contains("10395")]
i = df.index[df["release_date"] == "2011-11-10"].tolist()

print(i)
print("\n")
print(df["release_date"].isnull().sum())
print("Homam")


# =====================================================
# Funktionen
# =====================================================

def top_n_by_profit(df, profit, name):
    return df[[name, profit]].nlargest(10, profit)


print("\n", top_n_by_profit(df, "profit", "title"))


def average_rating_by_genre(df, name, value):
    return df.groupby([name], as_index=False)[value].mean()


print(average_rating_by_genre(df, "genres", "revenue"))


# =====================================================
# Klassen
# =====================================================

class Movie:
    def __init__(self, df, title, release_year, budget, revenue, genres, profit, return_ratio):
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
        for i in range(len(self.df["return_ratio"])):
            if self.df["return_ratio"][i] > 5:
                x.append(self.df["return_ratio"])
        return x

    def summary(self):
        return self.df.info(), self.df.dtypes


print("\n............")
print(df["release_date"])
print(df.isnull().sum())


class MovieAnalyzer:
    def __init__(self, df):
        self.df = df

    def top_profits(self, name, profit):
        return df[[name, profit]].nlargest(10, profit)

    def rating_by_decade(self, year_col, rating_col):
        df = self.df.copy()

        df["year"] = pd.to_numeric(df[year_col].str[:4], errors="coerce")
        df["year"] = df["year"].fillna(df["year"].median())
        df["decade"] = (df["year"] // 10) * 10

        return df.groupby("decade")[rating_col].mean().reset_index()

    def plot_budget_vs_revenue(self, budget_col="budget", revenue_col="revenue"):
        plt.figure(figsize=(8, 5))
        plt.scatter(self.df[budget_col], self.df[revenue_col])
        plt.xlabel("Budget")
        plt.ylabel("Revenue")
        plt.title("Budget vs Revenue")
        plt.grid(True)
        plt.show()


# =====================================================
# Analyse mit Klasse
# =====================================================

m = MovieAnalyzer(df)

x = m.top_profits("title", "profit")
g = m.rating_by_decade("release_date", "vote_average")
k = m.plot_budget_vs_revenue()

print(x, g, k)
print("############")
print(df.dtypes)
print(df["release_date"])


# =====================================================
# Spaltenauswahl
# =====================================================

chosen_columns = ["title", "release_date", "profit"]
existing_cols = [col for col in chosen_columns if col in df.columns]
print(df[existing_cols])

print("************")


# =====================================================
# Kleine Schleife (unverändert)
# =====================================================

i = 0
while i < len(df["return_ratio"]):
    i += 1
    print(df["return_ratio"])
    break


# =====================================================
# Aufgabe 6 – Auswertungen
# =====================================================

profit_level_count = df["profit_level"].value_counts()
print("\nAnzahl Filme pro profit_level:")
print(profit_level_count)

avg_imdb_genre = df.groupby("first_genre")["vote_average"].mean()
print("\nDurchschnittlicher IMDb-Score pro Genre:")
print(avg_imdb_genre)


# =====================================================
# Aufgabe 7 – Visualisierungen
# =====================================================

sns.set(style="whitegrid")

# Histogramm return_ratio
plt.figure(figsize=(8, 5))
sns.histplot(df["return_ratio"], bins=30, kde=True)
plt.title("Histogramm des Return Ratio")
plt.xlabel("Return Ratio")
plt.ylabel("Häufigkeit")
plt.show()

# Scatterplot Budget vs Revenue
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df["budget"], y=df["revenue"], alpha=0.5)
plt.title("Budget vs Revenue")
plt.xlabel("Budget")
plt.ylabel("Revenue")
plt.show()

# Boxplot Profit pro Decade
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x="decade", y="profit")
plt.title("Profit pro Decade")
plt.xlabel("Decade")
plt.ylabel("Profit")
plt.show()

# Countplot profit_level
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="profit_level")
plt.title("Anzahl Filme pro Profit Level")
plt.xlabel("Profit Level")
plt.ylabel("Anzahl Filme")
plt.show()

# Heatmap Korrelationen
corr_cols = ["budget", "revenue", "profit", "return_ratio", "vote_count"]
corr_matrix = df[corr_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Korrelations-Heatmap")
plt.show()
















