from sklearn.linear_model import LinearRegression # Import for Foward Feature Selection
from sklearn.feature_selection import RFE #Import of Recursive Feature Selection
import numpy as np #Import of numpy for array manipulation
import pandas as pd #import of pandas for data manipulation
import matplotlib.pyplot as plt #import of matplot for plotting tools 
from sklearn.ensemble import RandomForestRegressor #import for random forest.
#Original Dataset
data2=  pd.read_csv("Final_Dataset_3.csv")
train= pd.read_csv("Final_Dataset_3.csv")
array= data2.values
#Data aranged in arrays for model fitting if needed
X = array[:,2:14]
Y = array[:,15]
df = data2.drop(['year','CPI','CPI_Core','Unnamed: 0'],1) #dataframe to refrence drop year since it is non essential for the model and cpi since that is our target variable
df2 = df.drop('inflation',1) # dataframe with global infaltion removed, this variable appears to be overpowering the model
#Missing Value Ratio
MISS= train.isnull().sum()/len(train)*100
print('Missing Value Ratio',MISS) #Prints Missing value ratios
# Variance 
V = train.var()
print('Variance',V)
# COR is the stored correlation values
COR=  data2.corr()
#Console setup to print all the correlation values
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#Display of correlation values   
    print(COR)
#Random Forests feature selection with global input    
model = RandomForestRegressor(random_state=1, max_depth=15)
model.fit(df,train.CPI_Core)
#Display of random forests model results with global inflation
features = df.columns
importances = model.feature_importances_
indices = np.argsort(importances)[-15:]  # top 15 features
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()
#Random forests with global inflation removed
model.fit(df2,Y)
#plot for RandomForest w/o Global inflation
features = df.columns
importances = model.feature_importances_
indices = np.argsort(importances)[-15:]  # top 15 features
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

# Recrusive Feature Elimination is a method that optimizes the model for a set of  attributes by removing them one at time and recaluating effectivness.
#RFE with global inflation
model = LinearRegression()# Defines that we using a linear and non logistic model
rfe = RFE(model, 6) #We are creating a model to highlight a 6 feature situation
fit = rfe.fit(df, train.CPI_Core)#fitiing the model with rfe
print("Num Features: %d"% fit.n_features_) # display number of target features
print("Selected Features: %s"% fit.support_) #display selected features
print("Feature Ranking: %s"% fit.ranking_)# show ranking of features 1, being chosen higher value worse the feature in this method

#RFe without global inflation

model = LinearRegression()
rfe = RFE(model, 6)
fit = rfe.fit(df2, train.CPI_Core)
print("Num Features: %d"% fit.n_features_) 
print("Selected Features: %s"% fit.support_) 
print("Feature Ranking: %s"% fit.ranking_)

#Forward Feature seletion

from sklearn.feature_selection import f_regression


ffs = f_regression(df, train.CPI_Core)# Foward feature selection
ffs2 = f_regression(df2, train.CPI_Core)


print(ffs) # display Rankings with F value array first and P value second. For this case I chose to reject the null Hypothesis if alpha is less than 0.1
print(ffs2)# same as before sans inflation

COR.to_csv('CorrelationValues', sep='\t')
np.savetxt("foo.csv", ffs, delimiter=",")


#PCA + FA
from sklearn.decomposition import PCA
col = []
feat=[1,2,3,4,5,6,7,8,9,10,11,12]
for i in feat:
    col.append(i)
    pca = PCA(n_components=i)
    principalComponents = pca.fit_transform(X)
    principalDf = pd.DataFrame(data = principalComponents
             , columns = col)
    finalDf = pd.concat([principalDf, train[['CPI']]], axis = 1)
    Filename = 'PCA'+ str(i) + '.csv'
    np.savetxt(Filename, finalDf, delimiter=",")
    print(i)