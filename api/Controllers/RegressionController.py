import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
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
    def __init__(self, X, y, feature_scaling=False, test_size=0.2, random_state=42, string_col=[]):
        self.X = X
        self.y = y
        self.feature_scaling = feature_scaling
        self.test_size = test_size
        self.random_state = random_state
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state)
        self.train_mode = True
        self.string_col = string_col
        if len(self.string_col) > 0:
            self.ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), self.string_col)], remainder='passthrough')
            self.X_train = np.array(self.ct.fit_transform(self.X_train),dtype=np.float64)
            self.X_test = np.array(self.ct.transform(self.X_test),dtype=np.float64)
        
        if self.feature_scaling:
            self.scaler_X = StandardScaler()
            self.X_train = self.scaler_X.fit_transform(self.X_train)
            self.X_test = self.scaler_X.transform(self.X_test)
            
    def _inverse_transform(self, X):
        return self.scaler_X.inverse_transform(X)
    
    def _is_able_to_encode(self):
        string_col = []
        for col in range(len(self.X[0])):
            if isinstance(self.X[0][col], str):
                string_col.append(col)
        return string_col
        
             
        
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
            plt.plot(self._inverse_transform(X_range), y_pred_range, color='green', linewidth=3, label='Linear Regression')
        else:
            plt.scatter(self.X_train, self.y_train, color='red', label='Training Data')
            plt.scatter(self.X_test, self.y_test, color='blue', label='Test Data')
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
        X_range = np.linspace(np.min(self.X_train if not self.feature_scaling else self._inverse_transform(self.X_train)), np.max(self.X_train if not self.feature_scaling else self._inverse_transform(self.X_train)), 100).reshape(-1, 1)
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

    def _plot_knn_results(self, title, model):
        # Create a BytesIO object to save the plot as bytes
        buf = io.BytesIO()
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        
        # Create a range of values for plotting the KNN prediction line
        X_range = np.linspace(np.min(self.X_train if not self.feature_scaling else self._inverse_transform(self.X_train)), 
                            np.max(self.X_train if not self.feature_scaling else self._inverse_transform(self.X_train)), 
                            100).reshape(-1, 1)
        if self.feature_scaling:
            X_range_scaled = self.scaler.transform(X_range)
            y_pred_range = model.predict(X_range_scaled)
            X_range_plot = self._inverse_transform(X_range)
        else:
            y_pred_range = model.predict(X_range)
            X_range_plot = X_range
        
        # Plot training and test data
        if self.feature_scaling:
            plt.scatter(self._inverse_transform(self.X_train), self.y_train, color='red', label='Training Data')
            plt.scatter(self._inverse_transform(self.X_test), self.y_test, color='blue', label='Test Data')
        else:
            plt.scatter(self.X_train, self.y_train, color='red', label='Training Data')
            plt.scatter(self.X_test, self.y_test, color='blue', label='Test Data')
        
        # Plot the KNN prediction line
        plt.plot(X_range_plot, y_pred_range, color='green', linewidth=3, label='KNN Prediction')
        
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
    def _plot_decision_tree_results(self, title, model):
        # Create a BytesIO object to save the plot as bytes
        buf = io.BytesIO()
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        
        # Create a range of values for plotting the Decision Tree prediction line
        X_range = np.linspace(np.min(self.X_train if not self.feature_scaling else self._inverse_transform(self.X_train)), 
                            np.max(self.X_train if not self.feature_scaling else self._inverse_transform(self.X_train)), 
                            100).reshape(-1, 1)
        if self.feature_scaling:
            X_range_scaled = self.scaler.transform(X_range)
            y_pred_range = model.predict(X_range_scaled)
            X_range_plot = self._inverse_transform(X_range)
        else:
            y_pred_range = model.predict(X_range)
            X_range_plot = X_range
        
        # Plot training and test data
        if self.feature_scaling:
            plt.scatter(self._inverse_transform(self.X_train), self.y_train, color='red', label='Training Data')
            plt.scatter(self._inverse_transform(self.X_test), self.y_test, color='blue', label='Test Data')
        else:
            plt.scatter(self.X_train, self.y_train, color='red', label='Training Data')
            plt.scatter(self.X_test, self.y_test, color='blue', label='Test Data')
        
        # Plot the Decision Tree prediction line
        plt.plot(X_range_plot, y_pred_range, color='green', linewidth=3, label='Decision Tree Prediction')
        
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
    def _plot_gradient_boosting_results(self, title, model):
        # Create a BytesIO object to save the plot as bytes
        buf = io.BytesIO()
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        
        # Create a range of values for plotting the Gradient Boosting prediction line
        X_range = np.linspace(np.min(self.X_train if not self.feature_scaling else self._inverse_transform(self.X_train)), 
                            np.max(self.X_train if not self.feature_scaling else self._inverse_transform(self.X_train)), 
                            100).reshape(-1, 1)
        if self.feature_scaling:
            X_range_scaled = self.scaler.transform(X_range)
            y_pred_range = model.predict(X_range_scaled)
            X_range_plot = self._inverse_transform(X_range)
        else:
            y_pred_range = model.predict(X_range)
            X_range_plot = X_range
        
        # Plot training and test data
        if self.feature_scaling:
            plt.scatter(self._inverse_transform(self.X_train), self.y_train, color='red', label='Training Data')
            plt.scatter(self._inverse_transform(self.X_test), self.y_test, color='blue', label='Test Data')
        else:
            plt.scatter(self.X_train, self.y_train, color='red', label='Training Data')
            plt.scatter(self.X_test, self.y_test, color='blue', label='Test Data')
        
        # Plot the Gradient Boosting prediction line
        plt.plot(X_range_plot, y_pred_range, color='green', linewidth=3, label='Gradient Boosting Prediction')
        
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
        model = LinearRegression(
                    fit_intercept=fit_intercept, 
                    copy_X=copy_X, 
                    n_jobs=n_jobs, 
                    positive=positive)
        model.fit(self.X_train, self.y_train)
        results = {'model': model}
        if self.train_mode:
            y_pred = model.predict(self.X_test)
            r2 = r2_score(self.y_test, y_pred)
            mse = mean_squared_error(self.y_test, y_pred)
            results['r2'] = r2
            results['mse'] = mse
        if self.X.shape[1] == 1:

            results['image'] = self._plot_results('Linear Regression Results', model)
        
        if X_new is not None:
            X_new = X_new if len(self.string_col) == 0 else np.array(self.ct.transform(X_new),dtype=np.float64)
            y_new = model.predict(X_new if not self.feature_scaling else self.scaler_X.transform(X_new))
            results['y_new'] = y_new.tolist()
            
        return results

    def poly_reg(self,X_new=None, degree=2, include_bias=True, interaction_only=False, order='C', copy_X=True, n_jobs=None, positive=False, fit_intercept=True):
        poly = PolynomialFeatures(
                    degree=degree, 
                    include_bias=include_bias, 
                    interaction_only=interaction_only, 
                    order=order)
        model = LinearRegression(
                    copy_X=copy_X, 
                    n_jobs=n_jobs, 
                    positive=positive, 
                    fit_intercept=fit_intercept)
        results = {'model': model}
        if self.train_mode:
            X_poly_train = poly.fit_transform(self.X_train)
            X_poly_test = poly.transform(self.X_test)
            model.fit(X_poly_train, self.y_train)
            y_pred = model.predict(X_poly_test) if not self.feature_scaling else model.predict(self._inverse_transform(X_poly_test))
            r2 = r2_score(self.y_test, y_pred)
            mse = mean_squared_error(self.y_test, y_pred)
            results['r2'] = r2
            results['mse'] = mse
        if self.X.shape[1] == 1: 
            results['image'] = self._plot_poly_results('Polynomial Regression Results',model,poly) 
        if X_new is not None:
            X_new  = X_new if len(self.string_col) == 0 else self.ct.transform(X_new)
            X_poly_new = poly.transform(X_new)
            y_new = model.predict(X_poly_new)
            results['y_new'] = y_new.tolist()
        return results

    def svm_reg(self, X_new=None, kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1 ):
        model = SVR(
                    kernel=kernel, 
                    degree=degree, 
                    gamma=gamma, 
                    coef0=coef0, 
                    tol=tol, 
                    C=C, 
                    epsilon=epsilon, 
                    shrinking=shrinking, 
                    cache_size=cache_size, 
                    verbose=verbose, 
                    max_iter=max_iter)
        model.fit(self.X_train, self.y_train)
        results = {'model': model}
        print(self.X_train)
        
        if self.train_mode:
            y_pred = model.predict(self.X_test)
            r2 = r2_score(self.y_test, y_pred)
            mse = mean_squared_error(self.y_test, y_pred)
            results['r2'] = r2
            results['mse'] = mse
        if self.X.shape[1] == 1:
            results['image'] = self._plot_results('Support Vector Regression Results', model)
        if X_new is not None:
            X_new = X_new if len(self.string_col) == 0 else np.array(self.ct.transform(X_new),dtype=np.float64)
            y_new = model.predict(X_new if not self.feature_scaling else self.scaler_X.transform(X_new))
            results['y_new'] = y_new.tolist()
            
        return results

    def knn_reg(self,X_new=None, n_neighbors=5, weights='uniform', algorithm='auto',leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None):
        model = KNeighborsRegressor(
                    n_neighbors=n_neighbors, 
                    weights=weights, 
                    algorithm=algorithm,
                    leaf_size=leaf_size, 
                    p=p, metric=metric, 
                    metric_params=metric_params, 
                    n_jobs=n_jobs)
        model.fit(self.X_train, self.y_train)
        
        results = {'model': model}
        
        if self.train_mode:
            y_pred = model.predict(self.X_test)
            r2 = r2_score(self.y_test, y_pred)
            mse = mean_squared_error(self.y_test, y_pred)
            results['r2'] = r2
            results['mse'] = mse
            
        if self.X.shape[1] == 1:  # Check if X is 1-dimensional
            results['image'] = self._plot_knn_results('K-Nearest Neighbors Regression Results', model)
        if X_new is not None:
            X_new = X_new if len(self.string_col) == 0 else np.array(self.ct.transform(X_new),dtype=np.float64)
            y_new = model.predict(X_new if not self.feature_scaling else self.scaler_X.transform(X_new))
            results['y_new'] = y_new.tolist()
        return results

    def decision_tree_reg(self,X_new=None, criterion='squared_error', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, ccp_alpha=0.0, monotonic_cst=None):
        model = DecisionTreeRegressor(
                    criterion=criterion,
                    splitter=splitter,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    min_weight_fraction_leaf=min_weight_fraction_leaf,
                    max_features=max_features,
                    random_state=random_state,
                    max_leaf_nodes=max_leaf_nodes,
                    min_impurity_decrease=min_impurity_decrease,
                    ccp_alpha=ccp_alpha,
                    monotonic_cst=monotonic_cst)
        model.fit(self.X_train, self.y_train)
        results = {'model': model}
        if self.train_mode:
            y_pred = model.predict(self.X_test)
            r2 = r2_score(self.y_test, y_pred)
            results['r2'] = r2
            mse = mean_squared_error(self.y_test, y_pred)
            results['mse'] = mse
        if self.X.shape[1] == 1:  # Check if X is 1-dimensional
            results['image'] = self._plot_decision_tree_results('Decision Tree Regression Results', model)
        if X_new is not None:
            X_new = X_new if len(self.string_col) == 0 else np.array(self.ct.transform(X_new),dtype=np.float64)
            y_new = model.predict(X_new if not self.feature_scaling else self.scaler_X.transform(X_new))
            results['y_new'] = y_new.tolist()
        
        return results

    def random_forest_reg(self, X_new=None, n_estimators=100,  criterion='squared_error', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=1.0, max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None, monotonic_cst=None):
        model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    criterion=criterion,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    min_weight_fraction_leaf=min_weight_fraction_leaf,
                    max_features=max_features,
                    max_leaf_nodes=max_leaf_nodes,
                    min_impurity_decrease=min_impurity_decrease,
                    bootstrap=bootstrap,
                    oob_score=oob_score,
                    n_jobs=n_jobs,
                    random_state=random_state,
                    verbose=verbose,
                    warm_start=warm_start,
                    ccp_alpha=ccp_alpha,
                    max_samples=max_samples,
                    monotonic_cst=monotonic_cst
                )
        model.fit(self.X_train, self.y_train)
        results = {'model': model}
        if self.train_mode:
            y_pred = model.predict(self.X_test)
            r2 = r2_score(self.y_test, y_pred)
            mse = mean_squared_error(self.y_test, y_pred)
            results['r2'] = r2
            results['mse'] = mse
        if self.X.shape[1] == 1:  # Check if X is 1-dimensional
            results['image'] = self._plot_decision_tree_results('Random Forest Regression Results', model)
        if X_new is not None:
            X_new = X_new if len(self.string_col) == 0 else np.array(self.ct.transform(X_new),dtype=np.float64)
            y_new = model.predict(X_new if not self.feature_scaling else self.scaler_X.transform(X_new))
            results['y_new'] = y_new.tolist()
        return results

    def gradient_boosting_reg(self, X_new=None, loss='squared_error', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, init=None, random_state=None, max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0):
        model = GradientBoostingRegressor(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    loss=loss,
                    subsample=subsample,
                    criterion=criterion,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    min_weight_fraction_leaf=min_weight_fraction_leaf,
                    min_impurity_decrease=min_impurity_decrease,
                    init=init,
                    random_state=random_state,
                    max_features=max_features,
                    alpha=alpha,
                    verbose=verbose,
                    max_leaf_nodes=max_leaf_nodes,
                    warm_start=warm_start,
                    validation_fraction=validation_fraction,
                    n_iter_no_change=n_iter_no_change,
                    tol=tol,
                    ccp_alpha=ccp_alpha
                )
        model.fit(self.X_train, self.y_train)
        
        results = {'model': model}
        if self.train_mode:
            y_pred = model.predict(self.X_test)
            r2 = r2_score(self.y_test, y_pred)
            mse = mean_squared_error(self.y_test, y_pred)
            results['r2'] = r2
            results['mse'] = mse
        if self.X.shape[1] == 1:  # Check if X is 1-dimensional
            results['image'] = self._plot_gradient_boosting_results('Gradient Boosting Regression Results', model)
        if X_new is not None:
            X_new = X_new if len(self.string_col) == 0 else np.array(self.ct.transform(X_new),dtype=np.float64)
            y_new = model.predict(X_new if not self.feature_scaling else self.scaler_X.transform(X_new))
            results['y_new'] = y_new.tolist()
        return results