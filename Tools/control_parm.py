import numpy as np

class LinearSchedular():
    def __init__(self,splines):#每一个样条是左闭右开区间,最后追加一次采样，构成完全的闭区间
        self.splines=splines
        self.step=0
        self.stage = [0]
        for sp in splines:
            self.stage.append(self.stage[-1]+sp[2])
        self.stage.remove(0)
        self.spline_idx=0
    
    def Step(self):
        if self.step>=self.stage[-1]:
            return self.splines[-1][1]

        if self.step>=self.stage[self.spline_idx]:
            self.spline_idx+=1
        
        if self.spline_idx==0:
            ratio=(self.step-0)/self.splines[self.spline_idx][2]
        else:
            ratio=(self.step-self.stage[self.spline_idx-1])/self.splines[self.spline_idx][2]

        self.w = self.splines[self.spline_idx][0]+ratio*(self.splines[self.spline_idx][1]-self.splines[self.spline_idx][0])
        self.step+=1
        return self.w

if __name__ == '__main__':
    w=0
    schedule=[(0,1,3),(1,2,4)]
    LS=LinearSchedular(schedule)
    x,y=[],[]
    for i in range(9):
        x.append(i)
        y.append(LS.Step())
    print(y)