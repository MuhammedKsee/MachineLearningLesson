import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinReg:
    def __init__(self):
        self.TSS = None
        self.RSS = None
        self.slope = None
        self.intercept = None
        self.numerator = None
        self.denominator = None

    def fit(self,x,y):
        self.numerator = np.sum((x-np.mean(x))*(y-np.mean(y)))
        self.denominator = np.sum((x-np.mean(x))**2)
        self.slope = self.numerator / self.denominator
        self.intercept = np.mean(y) - self.slope * np.mean(x)
    
    def predict(self,x):
        predict = self.intercept + self.slope*x
        return predict
    
    def r2Score(self,y,y_pred):
        self.RSS = np.sum((y-y_pred)**2)
        self.TSS = np.sum((y-np.mean(y))**2)
        r2 = 1-(self.RSS/self.TSS)
        return r2
        