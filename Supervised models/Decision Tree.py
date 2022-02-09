# %%

import pandas
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg

# %% Data fiddling

show_data = pandas.read_csv("shows.csv")

# Split data into columns
show_data[['Age', 'Experience', 'Rank', 'Nationality', 'Go']] = show_data[
    'Age;Experience;Rank;Nationality;Go'].str.split(';',
                                                    expand=True)
# Remove the first column
show_data.drop(show_data.columns[0], axis=1, inplace=True)

# %%

# Encode the categories into numericals

d = {'UK': 0, 'USA': 1, 'N': 2}
show_data['Nationality'] = show_data['Nationality'].map(d)
d = {'YES': 1, 'NO': 0}
show_data['Go'] = show_data['Go'].map(d)

print(show_data)

features = ['Age', 'Experience', 'Rank', 'Nationality']

X = show_data[features]
y = show_data['Go']

print(X)
print(y)

# %% Do the actual DT model and visualize it

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)
tree.plot_tree(dtree)
plt.show()

#%%
