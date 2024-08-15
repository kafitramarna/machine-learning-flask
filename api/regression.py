from flask import Blueprint, request, jsonify
from Controllers.RegressionController import RegressionController
import numpy as np

regression_bp = Blueprint('regression', __name__)

def create_regression_response(results):
    response = {
        'model': str(results['model'])
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

def _is_able_to_encode(X):
    string_col = []
    for col in range(len(X[0])):
        if isinstance(X[0][col], str):
            string_col.append(col)
    return string_col

def get_common_params(data):
    X = data['X']
    string_col = _is_able_to_encode(X)
    X = np.array(X)
    y = np.array(data['y'])
    X_new = data.get('X_new', None)
    feature_scaling_X = data.get('feature_scaling_X', False)
    feature_scaling_y = data.get('feature_scaling_y', False)
    test_size = data.get('test_size', 0.2)
    random_state = data.get('random_state', 42)
    return X, y, X_new, feature_scaling_X, feature_scaling_y, test_size, random_state, string_col

@regression_bp.route('/linear-regression', methods=['POST'])
def linear_regression():
    data = request.get_json()
    if not data or 'X' not in data or 'y' not in data:
        return jsonify({'error': 'Invalid input data'}), 400
    
    try:
        X, y, X_new, feature_scaling_X, feature_scaling_y, test_size, random_state, string_col = get_common_params(data)
        copy_X = data.get('copy_X', True)
        n_jobs = data.get('n_jobs', None)
        positive = data.get('positive', False)
        fit_intercept = data.get('fit_intercept', True)
        
        reg_controller = RegressionController(X, y, feature_scaling_X=feature_scaling_X, feature_scaling_y=feature_scaling_y, test_size=test_size, random_state=random_state, string_col=string_col)
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
        X, y, X_new, feature_scaling_X, feature_scaling_y, test_size, random_state, string_col = get_common_params(data)
        degree = data.get('degree', 2)
        include_bias = data.get('include_bias', True)
        interaction_only = data.get('interaction_only', False)
        order = data.get('order', 'C')
        copy_X = data.get('copy_X', True)
        n_jobs = data.get('n_jobs', None)
        positive = data.get('positive', False)
        fit_intercept = data.get('fit_intercept', True)
        reg_controller = RegressionController(X, y, feature_scaling_X=feature_scaling_X, feature_scaling_y=feature_scaling_y, test_size=test_size, random_state=random_state, string_col=string_col)
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
        X, y, X_new, feature_scaling_X, feature_scaling_y, test_size, random_state, string_col = get_common_params(data)
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
        
        reg_controller = RegressionController(X, y, feature_scaling_X=feature_scaling_X, feature_scaling_y=feature_scaling_y, test_size=test_size, random_state=random_state, string_col=string_col)
        results = reg_controller.svm_reg(X_new=X_new, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, tol=tol, C=C, epsilon=epsilon, shrinking=shrinking, cache_size=cache_size, verbose=verbose, max_iter=max_iter)
        return jsonify(create_regression_response(results)), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@regression_bp.route('/knn-regression', methods=['POST'])
def knn_regression():
    data = request.get_json()
    if not data or 'X' not in data or 'y' not in data:
        return jsonify({'error': 'Invalid input data'}), 400
    
    try:
        X, y, X_new, feature_scaling_X, feature_scaling_y, test_size, random_state, string_col = get_common_params(data)

        n_neighbors = data.get('n_neighbors', 5)
        weights = data.get('weights', 'uniform')
        algorithm = data.get('algorithm', 'auto')
        leaf_size = data.get('leaf_size', 30)
        p = data.get('p', 2)
        metric = data.get('metric', 'minkowski')
        metric_params = data.get('metric_params', None)
        n_jobs = data.get('n_jobs', None)
        
        reg_controller = RegressionController(X, y, feature_scaling_X=feature_scaling_X, feature_scaling_y=feature_scaling_y, test_size=test_size, random_state=random_state, string_col=string_col)
        results = reg_controller.knn_reg(X_new=X_new, n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, leaf_size=leaf_size, p=p, metric=metric, metric_params=metric_params, n_jobs=n_jobs)
        return jsonify(create_regression_response(results)), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@regression_bp.route('/decision-tree-regression', methods=['POST'])
def decision_tree_regression():
    data = request.get_json()
    if not data or 'X' not in data or 'y' not in data:
        return jsonify({'error': 'Invalid input data'}), 400
    
    try:
        X, y, X_new, feature_scaling_X, feature_scaling_y, test_size, random_state, string_col = get_common_params(data)
        criterion = data.get('criterion', 'squared_error')
        splitter = data.get('splitter', 'best')
        max_depth = data.get('max_depth', None)
        min_samples_split = data.get('min_samples_split', 2)
        min_samples_leaf = data.get('min_samples_leaf', 1)
        min_weight_fraction_leaf = data.get('min_weight_fraction_leaf', 0.0)
        max_features = data.get('max_features', None)
        random_state_dec = data.get('random_state_dec', None)
        max_leaf_nodes = data.get('max_leaf_nodes', None)
        min_impurity_decrease = data.get('min_impurity_decrease', 0.0)
        ccp_alpha = data.get('ccp_alpha', 0.0)
        monotonic_cst = data.get('monotonic_cst', None)
        
        reg_controller = RegressionController(X, y, feature_scaling_X=feature_scaling_X, feature_scaling_y=feature_scaling_y, test_size=test_size, random_state=random_state, string_col=string_col)
        results = reg_controller.decision_tree_reg(X_new=X_new, criterion=criterion, splitter=splitter, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, random_state=random_state_dec, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, ccp_alpha=ccp_alpha, monotonic_cst=monotonic_cst)
        return jsonify(create_regression_response(results)), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@regression_bp.route('/random-forest-regression', methods=['POST'])
def random_forest_regression():
    data = request.get_json()
    if not data or 'X' not in data or 'y' not in data:
        return jsonify({'error': 'Invalid input data'}), 400
    
    try:
        X, y, X_new, feature_scaling_X, feature_scaling_y, test_size, random_state, string_col = get_common_params(data)
        n_estimators = data.get('n_estimators', 100)
        criterion = data.get('criterion', 'squared_error')
        max_depth = data.get('max_depth', None)
        min_samples_split = data.get('min_samples_split', 2)
        min_samples_leaf = data.get('min_samples_leaf', 1)
        min_weight_fraction_leaf = data.get('min_weight_fraction_leaf', 0.0)
        max_features = data.get('max_features', 'auto')
        max_leaf_nodes = data.get('max_leaf_nodes', None)
        bootstrap = data.get('bootstrap', True)
        oob_score = data.get('oob_score', False)
        n_jobs = data.get('n_jobs', None)
        random_state_rf = data.get('random_state_rf', None)
        verbose = data.get('verbose', 0)
        warm_start = data.get('warm_start', False)
        ccp_alpha = data.get('ccp_alpha', 0.0)
        max_samples = data.get('max_samples', None)
        monotonic_cst = data.get('monotonic_cst', None)
        
        reg_controller = RegressionController(X, y, feature_scaling_X=feature_scaling_X, feature_scaling_y=feature_scaling_y, test_size=test_size, random_state=random_state, string_col=string_col)
        results = reg_controller.random_forest_reg(X_new=X_new, n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes, bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state_rf, verbose=verbose, warm_start=warm_start, ccp_alpha=ccp_alpha, max_samples=max_samples, monotonic_cst=monotonic_cst)
        return jsonify(create_regression_response(results)), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
