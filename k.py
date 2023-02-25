# Data cleaning & getting rid of irrelevant information before clustering
# Finding the optimal value of k
# Storing cluster to which the house belongs along with the data
from imp_lib import *
# Creating input features and target variable
df_new = df.copy()
df_new = df_new.drop(['TotalFloor','Bathroom',],axis=1)
print(df_new.head())
X = df_new
y = df_new.Price

# printing the shapes
print('X matrix dimensionality : ', X.shape)
print('y matrix dimensionality : ', y.shape)
# Performing train test split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 3)

# Confirming whether train test split is performed properly or not
print(X.shape)
print(y.shape)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Finding Optimal Value of k
# Empty list for appending rmse
rmse_val = []
for K in range(1,20):
    model = KNeighborsRegressor(n_neighbors = K)
    model.fit(X_train,y_train)
    pred = model.predict(X_test)
    error = np.sqrt(mean_squared_error(y_test, pred))
    rmse_val.append(error)
    print('RMSE value for k = ', K, 'is : ', error)

# Plot
k_range = range(1,20)
plt.plot(k_range, rmse_val)
plt.xlabel('K')
plt.ylabel('RMSE')
plt.show()

# Optimal Model
model = KNeighborsRegressor(n_neighbors = 4)
model.fit(X_train,y_train)
pred = model.predict(X_test)
error = np.sqrt(mean_squared_error(y_test, pred))
print('RMSE : ', error)

from sklearn.cluster import KMeans
cs = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    cs.append(kmeans.inertia_)
plt.plot(range(1, 11), cs)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('CS')
plt.show()

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans

error_rate = []
for i in range(1,40):
 knn = KNeighborsClassifier(n_neighbors=i)
 knn.fit(X_train,y_train)
 pred_i = knn.predict(X_test)
 error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', 
         marker='o',markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
print("Minimum error:-",min(error_rate),"at K =",error_rate.index(min(error_rate)))

acc = []
from sklearn import metrics
for i in range(1,40):
    neigh = KNeighborsClassifier(n_neighbors = i).fit(X_train,y_train)
    yhat = neigh.predict(X_test)
    acc.append(metrics.accuracy_score(y_test, yhat))
    
plt.figure(figsize=(10,6))
plt.plot(range(1,40),acc,color = 'blue',linestyle='dashed', 
         marker='o',markerfacecolor='red', markersize=10)
plt.title('accuracy vs. K Value')
plt.xlabel('K')
plt.ylabel('Accuracy')
print("Maximum accuracy:-",max(acc),"at K =",acc.index(max(acc)))