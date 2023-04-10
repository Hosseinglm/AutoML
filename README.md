# AutoML
It automates the process of creating the model. Feeding your dataset into an AutoML tool, which will automatically split the data into training and testing sets, preprocess the data, and then try out multiple machine learning algorithms (such as decision trees, neural networks, and random forests) to see which one performs best on your dataset.<br/>
After the AutoML tool has finished running, it will select the best-performing machine learning model and provide you with its code or an API that you can use to make predictions on new data. This allows you to quickly and easily create a personalized recommendation system without needing to manually experiment with different machine learning algorithms and settings.<br/>
I'm using the iris dataset from scikit-learn as an example classification problem. I start by splitting the data into training and testing sets, and then create a Random Forest classifier. I define a hyperparameter grid to search over, and then use GridSearchCV (a scikit-learn function that performs hyperparameter tuning via cross-validation) to search over the grid and fit the best model. Finally, I print the best hyperparameters and the accuracy of the best model on the test set.<br/>
Result:<br/>
Best hyperparameters: {'max_depth': None, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 10}<br/>
Test set accuracy: 1.0
