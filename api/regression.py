from flask import Blueprint, request, jsonify
from Controllers.RegressionController import RegressionController
import numpy as np

regression_bp = Blueprint('regression', __name__)

def create_regression_response(results):
    response = {
        'model': str(results['model'])  # Simplified model representation
    }

    if 'r2' in results:
        response['r2'] = results['r2']
    
    if 'image' in results:
        response['image'] = results['image']
    
    if 'y_new' in results:
        response['y_new'] = results['y_new']
    if 'mse' in results:
        response['mse'] = results['mse']
    return response

@regression_bp.route('/linear-regression', methods=['POST'])
def linear_regression():
    data = request.get_json()
    if not data or 'X' not in data or 'y' not in data:
        return jsonify({'error': 'Invalid input data'}), 400
    
    try:
        X = np.array(data['X'])
        y = np.array(data['y'])
        X_new = np.array(data['X_new'])
        copy_X = data.get('copy_X', True)
        n_jobs = data.get('n_jobs', None)
        positive = data.get('positive', False)
        fit_intercept = data.get('fit_intercept', True)
        
        reg_controller = RegressionController(X, y,test_size=0.2, random_state=42)
        results = reg_controller.linear_reg(fit_intercept=fit_intercept, X_new=X_new, copy_X=copy_X, n_jobs=n_jobs, positive=positive)
        return jsonify(create_regression_response(results)), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@regression_bp.route('/poly-regression', methods=['POST'])
def poly_regression():
    data = request.get_json()
    if not data or 'X' not in data or 'y' not in data:
        return jsonify({'error': 'Invalid input data'}), 400
    
    try:
        X = np.array(data['X'])
        y = np.array(data['y'])
        X_new = np.array(data['X_new'])
        degree = data.get('degree', 2)
        include_bias = data.get('include_bias', True)
        interaction_only = data.get('interaction_only', False)
        order = data.get('order', 'C')
        copy_X = data.get('copy_X', True)
        n_jobs = data.get('n_jobs', None)
        positive = data.get('positive', False)
        fit_intercept = data.get('fit_intercept', True)
        reg_controller = RegressionController(X, y,test_size=0.2, random_state=42)
        results = reg_controller.poly_reg(X_new=X_new, degree=degree, include_bias=include_bias, interaction_only=interaction_only, order=order, copy_X=copy_X, n_jobs=n_jobs, positive=positive, fit_intercept=fit_intercept)        
        return jsonify(create_regression_response(results)), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@regression_bp.route('/svr-regression', methods=['POST'])
def svr_regression():
    data = request.get_json()
    
    if not data or 'X' not in data or 'y' not in data:
        return jsonify({'error': 'Invalid input data'}), 400
    
    try:
        X = np.array(data['X'])
        y = np.array(data['y'])
        X_new = np.array(data['X_new'])
        kernel = data.get('kernel', 'rbf')
        degree = data.get('degree', 3)
        gamma = data.get('gamma', 'scale')
        coef0 = data.get('coef0', 0.0)
        tol = data.get('tol', 0.001)
        C = data.get('C', 1.0)
        epsilon = data.get('epsilon', 0.1)
        shrinking = data.get('shrinking', True)
        cache_size = data.get('cache_size', 200)
        verbose = data.get('verbose', False)
        max_iter = data.get('max_iter', -1)
        
        reg_controller = RegressionController(X, y,)
        results = reg_controller.svm_reg(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)
        
        return jsonify(create_regression_response(results)), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@regression_bp.route('/knn-regression', methods=['POST'])
def knn_regression():
    data = request.get_json()
    
    if not data or 'X' not in data or 'y' not in data:
        return jsonify({'error': 'Invalid input data'}), 400
    
    try:
        X = np.array(data['X'])
        y = np.array(data['y'])
        n_neighbors = data.get('n_neighbors', 5)
        weights = data.get('weights', 'uniform')
        algorithm = data.get('algorithm', 'auto')
        
        reg_controller = RegressionController(X, y, **data)
        results = reg_controller.knn_reg(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)
        
        return jsonify(create_regression_response(results)), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@regression_bp.route('/decision-tree-regression', methods=['POST'])
def decision_tree_regression():
    data = request.get_json()
    
    if not data or 'X' not in data or 'y' not in data:
        return jsonify({'error': 'Invalid input data'}), 400
    
    try:
        X = np.array(data['X'])
        y = np.array(data['y'])
        criterion = data.get('criterion', 'squared_error')
        splitter = data.get('splitter', 'best')
        max_depth = data.get('max_depth', None)
        
        reg_controller = RegressionController(X, y, **data)
        results = reg_controller.decision_tree_reg(criterion=criterion, splitter=splitter, max_depth=max_depth)
        
        return jsonify(create_regression_response(results)), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@regression_bp.route('/random-forest-regression', methods=['POST'])
def random_forest_regression():
    data = request.get_json()
    
    if not data or 'X' not in data or 'y' not in data:
        return jsonify({'error': 'Invalid input data'}), 400
    
    try:
        X = np.array(data['X'])
        y = np.array(data['y'])
        n_estimators = data.get('n_estimators', 100)
        criterion = data.get('criterion', 'squared_error')
        max_depth = data.get('max_depth', None)
        
        reg_controller = RegressionController(X, y, **data)
        results = reg_controller.random_forest_reg(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth)
        
        return jsonify(create_regression_response(results)), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@regression_bp.route('/gradient-boosting-regression', methods=['POST'])
def gradient_boosting_regression():
    data = request.get_json()
    
    if not data or 'X' not in data or 'y' not in data:
        return jsonify({'error': 'Invalid input data'}), 400
    
    try:
        X = np.array(data['X'])
        y = np.array(data['y'])
        n_estimators = data.get('n_estimators', 100)
        learning_rate = data.get('learning_rate', 0.1)
        max_depth = data.get('max_depth', 3)
        
        reg_controller = RegressionController(X, y, **data)
        results = reg_controller.gradient_boosting_reg(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
        
        return jsonify(create_regression_response(results)), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

