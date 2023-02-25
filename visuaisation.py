from imp_lib import *
# Scatter plot
plt.scatter('Floor','TotalFloor',marker='*',data=df)
plt.xlabel('Floor')
plt.ylabel('TotalFloor')
plt.title('Scatter Plot')
plt.show()

plt.scatter('Bedroom','Living.Room',marker='o',data=df)
plt.xlabel('Bedrom')
plt.ylabel('Living.Room')
plt.title('Scatter Plot')
plt.show()

plt.scatter('TotalFloor','Floor',marker='o',data=df)
plt.xlabel('TotalFloor')
plt.ylabel('Floor')
plt.title('Scatter Plot')
plt.show()

# Histogram
sns.histplot(df['Price'])
plt.title('Histogram')
plt.show()

# Barplot
plt.bar(df['Sqft'],df['Price'])
plt.xlabel('Sqft')
plt.ylabel('Price')
plt.title('Barplot')
plt.show()

sns.barplot(x = 'Price',
            y = 'TotalFloor',
            data = df)
plt.xlabel('Price')
plt.ylabel('TotalFloor')
plt.title('Barplot')
plt.show()
sns.scatterplot(x="TotalFloor",
                    y="Price",
                    data=df)

# Pair plot
sns.pairplot(df, hue ='Price',)
plt.show()

# Joint plot
sns.jointplot(x = "Floor", y = "TotalFloor",
              kind = "hist", data = df)

plt.title('Pair Plot')
# show the plot
plt.show()

# Joint plot
sns.jointplot(x = "Floor", y = "TotalFloor",
              kind = "hist", data = df)
# show the plot
plt.show()

# count plot
sns.countplot(x="TotalFloor", hue="Price", data=df)
plt.xlabel("TotalFloor")
plt.show()