import pandas as pd

# save filepath to variable for easier access
melbourne_file_path = "./melb_data.csv"
# read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path)
# print a summary of the data in Melbourne data
melbourne_data.describe()
# print(melbourne_data.describe())

# The Melbourne data has some missing values (some houses for which some variables weren't recorded.)
# We'll learn to handle missing values in a later tutorial.
# Your Iowa data doesn't have missing values in the columns you use.
# So we will take the simplest option for now, and drop houses from our data.
# Don't worry about this much for now, though the code is:

# dropna drops missing values (think of na as "not available")
melbourne_data = melbourne_data.dropna(axis=0)

# Pulling out data coloumns, here we set it as a target
y = melbourne_data.Price
# Features of the data
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]
# print(X.describe()), prints what we need

# PREDICT
from sklearn.tree import DecisionTreeRegressor

# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)
melbourne_model.fit(X, y)

# print("Making predictions for the following 5 houses:")
# print(X.head())
# print("The predictions are")
# print(melbourne_model.predict(X.head()))

# MODEL VALIDATION
from sklearn.metrics import mean_absolute_error

predicted_home_prices = melbourne_model.predict(X);
mean_absolute_error(y, predicted_home_prices)
# print(mean_absolute_error(y, predicted_home_prices)) = 434.7159

# Why is this wrong?
# Since models' practical value come from making predictions on new data,
# we measure performance on data that wasn't used to build the model.
# The most straightforward way to do this is to exclude some data from the model-building process,
# and then use those to test the model's accuracy on data it hasn't seen before.
# This data is called validation data.

from sklearn.model_selection import train_test_split

# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)


# print(mean_absolute_error(val_y, val_predictions))

# COMPENSATION - try messing with how deep the tree is
def get_mae(max_leaf_nodes, train_X, val_X, train_y,
            val_y):  # utility function that tells us the mae for a given tree depth
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return mae


for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" % (max_leaf_nodes, my_mae))

# Using a RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
# print(mean_absolute_error(val_y, melb_preds)) , 191669 better than DescisionTree
