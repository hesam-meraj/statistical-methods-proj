import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("HousingData.xls") 



features = ["CRIM","ZN","INDUS",
            "CHAS","NOX","RM",
            "AGE","DIS","RAD"
            ,"TAX","PTRATIO"
            ,"B","LSTAT"]  
target = 'MEDV'  

for feature in features:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=df[feature], y=df[target])
    plt.xlabel(feature)
    plt.ylabel(target)
    plt.title(f"Scatter Plot: {feature} vs. {target}")
    plt.show()



sns.pairplot(df, x_vars=features, y_vars=['MEDV'], height=3, aspect=1, kind="scatter")
plt.show()

plt.figure(figsize=(12, 8))

corr_matrix = df.corr()


sns.heatmap(corr_matrix, annot=True, cmap="coolwarm",
            fmt=".2f", linewidths=0.5)


plt.title("Correlation Matrix of Boston Housing Dataset")

plt.show()



