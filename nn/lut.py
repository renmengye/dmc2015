from stage import *

class LUT(Stage):
    def __init__(self,
                 inputDim,
                 outputDim,
                 initRange=1.0,
                 initSeed=2,
                 needInit=True,
                 initWeights=0,
                 learningRate=0.0,
                 learningRateAnnealConst=0.0,
                 momentum=0.0,
                 deltaMomentum=0.0,
                 weightClip=0.0,
                 gradientClip=0.0,
                 weightRegConst=0.0,
                 name=None):
        Stage.__init__(self,
                 name=name,
                 learningRate=learningRate,
                 learningRateAnnealConst=learningRateAnnealConst,
                 momentum=momentum,
                 deltaMomentum=deltaMomentum,
                 weightClip=weightClip,
                 gradientClip=gradientClip,
                 weightRegConst=weightRegConst,
                 outputdEdX=False)
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.random = np.random.RandomState(initSeed)

        # Zeroth dimension of the weight matrix is reserved
        # for empty word at the end of a sentence.
        if needInit:
            self.W = np.concatenate(
                (np.zeros((outputDim, 1)),
                 self.random.uniform(
                -initRange/2.0, initRange/2.0,
                (outputDim , inputDim))), axis=1)
        else:
            self.W = np.concatenate(
                (np.zeros((outputDim, 1)), initWeights), axis=1)
        self.X = 0
        self.Y = 0
        pass

    def forward(self, X):
        X = X.reshape(X.size)
        Y = np.zeros((X.shape[0], self.outputDim))
        for n in range(0, X.shape[0]):
            Y[n, :] = self.W[:, X[n]]
        self.X = X
        self.Y = Y
        return Y

    def backward(self, dEdY):
        X = self.X
        self.dEdW = np.zeros(self.W.shape)
        for n in range(0, X.shape[0]):
            self.dEdW[:, X[n]] += dEdY[n, :]
        return None