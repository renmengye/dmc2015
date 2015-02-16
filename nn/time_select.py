from stage import *

class TimeSelect(Stage):
    def __init__(self, time, name=None):
        Stage.__init__(self, name=name)
        self.t = time
        self.X = 0
        self.Y = 0
        pass

    def forward(self, X):
        # X(t, n, i)
        Y = X[:, self.t, :]
        self.X = X
        self.Y = Y
        return Y

    def backward(self, dEdY):
        self.dEdW = 0
        dEdX = np.zeros(self.X.shape)
        dEdX[:, self.t,  :] = dEdY
        return dEdX

