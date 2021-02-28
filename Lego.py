#import
import pandas as pd
import numpy as np


#import file - df
df = pd.read_csv("lego_sets.csv")

#drop unwanted columns - df1
df1 = df.drop(["play_star_rating", "val_star_rating","prod_desc","prod_id","prod_long_desc", "num_reviews","country","set_name"], axis = "columns")

#print(df1.shape)


#convert difficulty to numerical values - still df1

def convertDifficulty(x):
  if x == 'Very Easy': return (1)
  elif x == 'Easy': return(2)
  elif x == 'Average': return(3)
  elif x == 'Challenging': return(4)
  elif x == 'Very Challenging': return(5)
  else: return None

df1['review_difficulty'] = df1['review_difficulty'].apply(convertDifficulty)

# Dealing with ages
def splitAge(age):
  if "+" in age:
    return int(age.split("+")[0])
  else:
    try: 
      return int(age.split("-")[0])
    except: 
      return int(str(age.split("-")[0])[0])

df1.ages = df1.ages.apply(splitAge).sort_values(ascending=False)


#Theme names
''''
print(df1.theme_name.value_counts().sort_values(ascending = False))
print(len(df1.theme_name.unique()))
'''

theme_stats = df1.theme_name.value_counts().sort_values(ascending = False)

theme_less_than_50 = theme_stats[theme_stats<=33]

df1.theme_name = df1.theme_name.apply(lambda x: 'other' if x in theme_less_than_50 else x)


#clean null values
df1 = df1.dropna(subset=['theme_name'])
df1.star_rating = df1.star_rating.fillna(4.0)
df1.review_difficulty = df1.review_difficulty.fillna(2.0)
#print(df1.isnull().sum())

#Turn Location into Numerical
dummies = pd.get_dummies(df1.theme_name)
df1 = pd.concat([df1, dummies.drop('other', axis = 'columns')], axis = 'columns')
df2 = df1.drop(['theme_name'], axis = 'columns')


# Apply linear LinearRegression -df2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = df2.drop('list_price',axis = 'columns')
y = df2.list_price
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)


'''

lr_lego = LinearRegression()
lr_lego.fit(X_train,y_train)
print(lr_lego.score(X_test,y_test))

# Cross Validation
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits = 5, test_size = 0.2, random_state = 0)
print(cross_val_score(LinearRegression(), X, y, cv=cv))

# Test other models
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_model_using_gridsearchcv(X,y):
    algos = {
        'linear_regression':{
            'model': LinearRegression(),
            'params': {'normalize': [True, False]}
        },
        'lasso':{
            'model': Lasso(),
            'params': {'alpha': [1, 2], 'selection': ['random', 'cyclic']
            }
        },
        'decision_tree':{
            'model': DecisionTreeRegressor(),
            'params': {'criterion': ['mse', 'friedman_mse'],'splitter': ['best', 'random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits = 5, test_size = 0.2, random_state=0)
    for algo_name, config in algos.items():
        gs = GridSearchCV(config['model'], config['params'], cv=cv, return_train_score = False)
        gs.fit(X, y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })
    return pd.DataFrame(scores, columns = ['model', 'best_score', 'best_params'])

print(find_best_model_using_gridsearchcv(X,y))
'''
# Decision Tree Fit
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
tree_lego = DecisionTreeRegressor(criterion = 'mse', splitter = 'random')
tree_lego.fit(X_train,y_train)
print(tree_lego.score(X_test,y_test))

cv = ShuffleSplit(n_splits = 5, test_size = 0.2, random_state = 0)
print(cross_val_score(DecisionTreeRegressor(), X, y, cv=cv))

# Predict 

def predict_price(theme, age, pieces, difficulty, rating):
    theme_index = np.where(X.columns == theme)[0][0]
    x = np.zeros(len(X.columns))
    x[0] = age
    x[1] = pieces
    x[2] = difficulty
    x[3] = rating
    if theme_index >= 0:
        x[theme_index] = 1
    return tree_lego.predict([x])[0]


print()
print()

predicted_price = predict_price('Star Wars™', 9, 1353, 3.4, 4.4)
print('The predicted price is ', predicted_price)
