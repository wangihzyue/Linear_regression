import numpy as np
from sklearn.preprocessing import PolynomialFeatures

class BayesianRegression:
    """
    Bayesian regression model

    w ~ N(w|0, alpha^(-1)I)
    y = X @ w
    t ~ N(t|X @ w, beta^(-1))
    """

    def __init__(self, alpha: float = 1., beta: float = 1.):
        self.alpha = alpha
        self.beta = beta
        self.w_mean = None
        self.w_precision = None

    def fit(self, X: np.ndarray, t: np.ndarray):
        """
        bayesian update of parameters given training dataset

        Parameters
        ----------
        X : (N, n_features) np.ndarray
            training data independent variable
        t : (N,) np.ndarray
            training data dependent variable
        """
        a = X.T.dot(X)
        b = self.alpha/self.beta * np.eye(a.shape[0], a.shape[1]) + a
        self.w_mean = self.beta*np.linalg.pinv(b).dot(X.T).dot(t)
        S_inv = self.beta*X.T.dot(X)
        self.w_precision = np.linalg.pinv(self.alpha*np.eye(S_inv.shape[0],S_inv.shape[1])+S_inv)
    def predict(self, X: np.ndarray):
        """
        return mean and standard deviation of predictive distribution

        Parameters
        ----------
        X : (N, n_features) np.ndarray
            independent variable

        Returns
        -------
        y : (N,) np.ndarray
            mean of the predictive distribution
        y_std : (N,) np.ndarray
            standard deviation of the predictive distribution
        """
        y = X.dot(self.w_mean)
        y_std = []
        for k in range(0, len(y)):
            y_std.append(np.sqrt(1/self.beta+X[k].dot(self.w_precision).dot(X[k].T)))
        return y, y_std

def root_mean_square_error(a, b):
    return np.sqrt(np.mean(np.square(a - b)))/2

if __name__ == '__main__':
    n, m = input().split()
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    for _ in range(int(n)):
        x, y = input().split()
        x_train.append([float(x)])
        y_train.append(float(y))
    for _ in range(int(m)):
        x, y = input().split()
        x_test.append([float(x)])
        y_test.append(float(y))
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    poly = PolynomialFeatures(10)
    x_poly = poly.fit_transform(x_train)
    br = BayesianRegression()
    br.fit(x_poly,y_train)
    x2_poly = poly.fit_transform(x_test)
    y_pred, y_std = br.predict(x2_poly)
    print(round(root_mean_square_error(y_pred,y_test),6))
    for k in y_std:
        print(round(k,6))
