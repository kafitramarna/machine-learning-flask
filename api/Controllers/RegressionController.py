import numpy as np
import matplotlib.pyplot as plt
import io
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import base64

class RegressionController:
    def __init__(self, X, y, feature_scaling=False, test_size=None, random_state=None):
        self.X = X
        self.y = y
        self.feature_scaling = feature_scaling
        self.test_size = test_size
        self.random_state = random_state
        
        if self.feature_scaling:
            self.scaler_X = StandardScaler()
            self.X_scaled = self.scaler_X.fit_transform(self.X)
        else:
            self.X_scaled = self.X
        
        if self.test_size is not None:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X_scaled, self.y, test_size=self.test_size, random_state=self.random_state)
            self.train_mode = True
        else:
            self.X_train = self.X_scaled
            self.y_train = self.y
            self.train_mode = False
        
    def _inverse_transform(self, X):
        if self.feature_scaling:
            return self.scaler_X.inverse_transform(X)
        return X

    def _plot_results(self, y_true, y_pred, title):
        # Create a BytesIO object to save the plot as bytes
        buf = io.BytesIO()
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        if self.feature_scaling:
            plt.scatter(self._inverse_transform(self.X_test), y_true, color='red', label='Actual')
            plt.scatter(self._inverse_transform(self.X_test), y_pred, color='blue', label='Predicted')
        else:
            plt.scatter(self.X_test, y_true, color='red', label='Actual')
            plt.scatter(self.X_test, y_pred, color='blue', label='Predicted')
        
        # Add title and labels
        plt.title(title)
        plt.xlabel('Feature')
        plt.ylabel('Target')
        plt.legend()
        plt.grid(True)
        
        # Save the plot to the BytesIO object in PNG format
        plt.savefig(buf, format='png')
        buf.seek(0)  # Rewind the buffer to the beginning
        
        # Convert the byte data to a Base64-encoded string
        image_bytes = buf.getvalue()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Close the plot and buffer
        plt.close()
        buf.close()
        
        # Return the Base64-encoded string
        return image_base64

    def linear_reg(self, fit_intercept=True , X_new=None):
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        results = {'model': model}
        if self.train_mode:
            y_pred = model.predict(self.X_test)
            r2 = r2_score(self.y_test, y_pred)
            results['r2'] = r2
            if self.X.shape[1] == 1:  # Check if X is 1-dimensional
                results['image'] = self._plot_results(self.y_test, y_pred, 'Linear Regression Results')
        if X_new is not None:
            y_new = model.predict([[20],[21]])
            # print(y_new)
            results['y_new'] = y_new.tolist()  # Convert numpy array to list for JSON serialization
        # print(results)
        return results

    def poly_reg(self, degree=2, include_bias=True, interaction_only=False):
        poly = PolynomialFeatures(degree=degree, include_bias=include_bias, interaction_only=interaction_only)
        X_poly_train = poly.fit_transform(self.X_train)
        
        if self.train_mode:
            X_poly_test = poly.transform(self.X_test)
        else:
            X_poly_test = X_poly_train
        
        model = LinearRegression()
        model.fit(X_poly_train, self.y_train)
        y_pred = model.predict(X_poly_test)
        
        results = {'model': model}
        
        if self.train_mode:
            r2 = r2_score(self.y_test, y_pred)
            results['r2'] = r2
            if self.X.shape[1] == 1:  # Check if X is 1-dimensional
                results['image'] = self._plot_results(self.y_test, y_pred, 'Polynomial Regression Results')
        
        return results

    def svm_reg(self, kernel='rbf', C=1.0, epsilon=0.1, gamma='scale'):
        model = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)
        model.fit(self.X_train, self.y_train)
        
        results = {'model': model}
        
        if self.train_mode:
            y_pred = model.predict(self.X_test)
            r2 = r2_score(self.y_test, y_pred)
            results['r2'] = r2
            if self.X.shape[1] == 1:  # Check if X is 1-dimensional
                results['image'] = self._plot_results(self.y_test, y_pred, 'SVR Results')
        
        return results

    def knn_reg(self, n_neighbors=5, weights='uniform', algorithm='auto'):
        model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)
        model.fit(self.X_train, self.y_train)
        
        results = {'model': model}
        
        if self.train_mode:
            y_pred = model.predict(self.X_test)
            r2 = r2_score(self.y_test, y_pred)
            results['r2'] = r2
            if self.X.shape[1] == 1:  # Check if X is 1-dimensional
                results['image'] = self._plot_results(self.y_test, y_pred, 'KNN Regression Results')
        
        return results

    def decision_tree_reg(self, criterion='squared_error', splitter='best', max_depth=None):
        model = DecisionTreeRegressor(criterion=criterion, splitter=splitter, max_depth=max_depth)
        model.fit(self.X_train, self.y_train)
        
        results = {'model': model}
        
        if self.train_mode:
            y_pred = model.predict(self.X_test)
            r2 = r2_score(self.y_test, y_pred)
            results['r2'] = r2
            if self.X.shape[1] == 1:  # Check if X is 1-dimensional
                results['image'] = self._plot_results(self.y_test, y_pred, 'Decision Tree Regression Results')
        
        return results

    def random_forest_reg(self, n_estimators=100, criterion='squared_error', max_depth=None):
        model = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth)
        model.fit(self.X_train, self.y_train)
        
        results = {'model': model}
        
        if self.train_mode:
            y_pred = model.predict(self.X_test)
            r2 = r2_score(self.y_test, y_pred)
            results['r2'] = r2
            if self.X.shape[1] == 1:  # Check if X is 1-dimensional
                results['image'] = self._plot_results(self.y_test, y_pred, 'Random Forest Regression Results')
        
        return results

    def gradient_boosting_reg(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
        model.fit(self.X_train, self.y_train)
        
        results = {'model': model}
        
        if self.train_mode:
            y_pred = model.predict(self.X_test)
            r2 = r2_score(self.y_test, y_pred)
            results['r2'] = r2
            if self.X.shape[1] == 1:  # Check if X is 1-dimensional
                results['image'] = self._plot_results(self.y_test, y_pred, 'Gradient Boosting Regression Results')
        
        return results

    def xgboost_reg(self, n_estimators=100, learning_rate=0.1, max_depth=3, objective='reg:squarederror'):
        model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, objective=objective)
        model.fit(self.X_train, self.y_train)
        
        results = {'model': model}
        
        if self.train_mode:
            y_pred = model.predict(self.X_test)
            r2 = r2_score(self.y_test, y_pred)
            results['r2'] = r2
            if self.X.shape[1] == 1:  # Check if X is 1-dimensional
                results['image'] = self._plot_results(self.y_test, y_pred, 'XGBoost Regression Results')
        
        return results