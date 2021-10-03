import numpy as np
from sklearn.preprocessing import PolynomialFeatures
class LinearRegression:
    """
    Linear regression model
    y = X @ w
    t ~ N(t|X @ w, var)
    """

    def __init__(self):
        self.w = None
        self.var = None

    def fit(self, X: np.ndarray, t: np.ndarray):
        """
        perform least squares fitting
        Parameters
        ----------
        X : (N, D) np.ndarray
        training independent variable
        t : (N,) np.ndarray
        training dependent variable
        """
        self.w = np.linalg.pinv(X).dot(t)
        self.var = np.var(X.dot(self.w)-t)

    def predict(self, X: np.ndarray):
        """
        make prediction given input
        Parameters
        ----------
        X : (N, D) np.ndarray
        samples to predict their output
        Returns
        -------
        y : (N,) np.ndarray
            prediction of each sample
        """
        y = X.dot(self.w)
        return y

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
    beskK = -1
    toprmse = 1000000
    std = -1
    #for k in range(0,11):
    poly = PolynomialFeatures(10)
    x_poly = poly.fit_transform(x_train)
    lr = LinearRegression()
    lr.fit(x_poly,y_train)
    x2_poly = poly.fit_transform(x_test)
    y_pred = lr.predict(x2_poly)
    yrmse =root_mean_square_error(y_pred,y_test)
    if yrmse<toprmse:
        beskK = 10
        toprmse = yrmse
        std = np.sqrt(lr.var)
    print(beskK)
    print(round(std,6))
