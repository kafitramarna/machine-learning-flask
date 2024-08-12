import numpy as np
import matplotlib.pyplot as plt
import io
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import base64

class RegressionController:
    def __init__(self, X, y, feature_scaling=False, test_size=0.2, random_state=42):
        self.X = X
        self.y = y
        self.feature_scaling = feature_scaling
        self.test_size = test_size
        self.random_state = random_state
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_scaled, self.y, test_size=self.test_size, random_state=self.random_state)
        self.train_mode = True
        
        if self.feature_scaling:
            self.scaler_X = StandardScaler()
            self.X_train = self.scaler_X.fit_transform(self.X_train)
            self.X_test = self.scaler_X.transform(self.X_test)
            
    def _inverse_transform(self, X):
        return self.scaler_X.inverse_transform(X)
    def _one_hot_encode(self):
        string_col = []
        for col in range(len(self.X[0])):
            if isinstance(self.X[0][col], str):
                string_col.append(col)
        if len(string_col) > 0:
            ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), string_col)], remainder='passthrough')
            self.X_train = np.array(ct.fit_transform(self.X_train))
            self.X_test = np.array(ct.transform(self.X_test))
             
        
    def _plot_results(self, title, model):
        # Create a BytesIO object to save the plot as bytes
        buf = io.BytesIO()
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        
        # Create a range of values for plotting the regression line
        X_range = np.linspace(np.min(self.X_train), np.max(self.X_train), 100).reshape(-1, 1)
        y_pred_range = model.predict(X_range)
        
        # Plot the training data
        if self.feature_scaling:
            plt.scatter(self._inverse_transform(self.X_train), self.y_train, color='red', label='Training Data')
            plt.scatter(self._inverse_transform(self.X_test), self.y_test, color='blue', label='Test Data')
        else:
            plt.scatter(self.X_train, self.y_train, color='red', label='Training Data')
            plt.scatter(self.X_test, self.y_test, color='blue', label='Test Data')
        
        # Plot the regression line
        plt.plot(X_range, y_pred_range, color='green', linewidth=3, label='Linear Regression')
        
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
    def _plot_poly_results(self, title, model, poly):
        # Create a BytesIO object to save the plot as bytes
        buf = io.BytesIO()
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        
        # Create a range of values for plotting the polynomial fit
        X_range = np.linspace(np.min(self.X_train), np.max(self.X_train), 100).reshape(-1, 1)
        X_range_poly = poly.transform(X_range)
        y_pred_range = model.predict(X_range_poly)
        
        # Plot training data
        if self.feature_scaling:
            plt.scatter(self._inverse_transform(self.X_train), self.y_train, color='red', label='Training Data')
            plt.scatter(self._inverse_transform(self.X_test), self.y_test, color='blue', label='Test Data')
        else:
            plt.scatter(self.X_train, self.y_train, color='red', label='Training Data')
            plt.scatter(self.X_test, self.y_test, color='blue', label='Test Data')
        
        # Plot the polynomial fit
        plt.plot(X_range, y_pred_range, color='green', linewidth=3, label='Polynomial Fit')
        
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

    def linear_reg(self, fit_intercept=True , X_new=None, copy_X=True, n_jobs=None, positive=False):
        model = LinearRegression(fit_intercept=fit_intercept, copy_X=copy_X, n_jobs=n_jobs, positive=positive)
        model.fit(self.X_train, self.y_train)
        results = {'model': model}
        
        if self.train_mode:
            y_pred = model.predict(self.X_test)
            r2 = r2_score(self.y_test, y_pred)
            mse = mean_squared_error(self.y_test, y_pred)
            results['r2'] = r2
            results['mse'] = mse
        if self.X.shape[1] == 1:
            # Update plotting to use the model directly
            results['image'] = self._plot_results('Linear Regression Results', model)
        
        if X_new is not None:
            y_new = model.predict(X_new)
            results['y_new'] = y_new.tolist()
            
        return results

    def poly_reg(self,X_new=None, degree=2, include_bias=True, interaction_only=False, order='C', copy_X=True, n_jobs=None, positive=False, fit_intercept=True):
        poly = PolynomialFeatures(degree=degree, include_bias=include_bias, interaction_only=interaction_only, order=order)
        model = LinearRegression(copy_X=copy_X, n_jobs=n_jobs, positive=positive, fit_intercept=fit_intercept)
        results = {'model': model}
        if self.train_mode:
            X_poly_train = poly.fit_transform(self.X_train)
            X_poly_test = poly.transform(self.X_test)
            model.fit(X_poly_train, self.y_train)
            y_pred = model.predict(X_poly_test)
            r2 = r2_score(self.y_test, y_pred)
            mse = mean_squared_error(self.y_test, y_pred)
            results['r2'] = r2
            results['mse'] = mse
        if self.X.shape[1] == 1: 
            results['image'] = self._plot_poly_results('Polynomial Regression Results',model,poly) 
        if X_new is not None:
            X_poly_new = poly.transform(X_new)
            y_new = model.predict(X_poly_new)
            results['y_new'] = y_new.tolist()
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