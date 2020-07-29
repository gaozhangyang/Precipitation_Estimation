from sklearn.metrics import confusion_matrix
import numpy as np


class Indicator:
    def __init__(self):
        self.y_true=[]
        self.y_pred=[]
        
    def indicator_cls(self):
        tn, fp, fn, tp=confusion_matrix(self.y_true,self.y_pred ).ravel()
        POD=tp/(tp+fn)
        FAR=fp/(tp+fp)
        CSI=tp/(tp+fn+fp)
        return {'POD':POD,'FAR':FAR,'CSI':CSI}
    
    def indicator_reg(self):
        conv=np.cov([self.y_true,self.y_pred])
        CC=conv[0,1]/np.sqrt(conv[0,0]*conv[1,1])
        BIAS=np.sum(np.array(self.y_pred)-np.array(self.y_true))/np.sum(self.y_true)
        MSE=np.mean((np.array(self.y_pred)-np.array(self.y_true))**2)
        return {'CC':CC,'BIAS':BIAS,'MSE':MSE}
    
    def reset(self):
        self.y_true=[]
        self.y_pred=[]


if __name__ == "__main__":
    indicator=Indicator()

    y_true=np.array([0,1,0,1])
    y_pred=np.array([0,0,0,1])
    indicator.y_true.extend(y_true.tolist())
    indicator.y_pred.extend(y_pred.tolist())
    print(indicator.indicator_cls())
    indicator.reset()

    y_true=np.random.rand(10)
    y_pred=np.random.rand(10)
    indicator.y_true.extend(y_true.tolist())
    indicator.y_pred.extend(y_pred.tolist())
    print(indicator.indicator_reg())