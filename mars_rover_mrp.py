from functools import reduce

s=[0 for i in range(0,7)]
s[0]=1
s[6]=10
selfp=[0.2 for i in range(0,7)]
selfp[0]=0.6
selfp[6]=0.6
value=[]
probability=[]
move=list() # only for debug

def dfs(cur,step,v,p=1,prep=1,m=0,horizon=4):
    if(step==horizon):
        if m in move:
            return
        value.append(v)
        probability.append(prep)
        move.append(m)
        return
    else:
        if step==0:
            value.clear()
            probability.clear()
            move.clear()
        v+=0.5**step*s[cur-1]
        m=m*10+cur
        dfs(cur,step+1,v,p*selfp[cur-1],p,m,horizon=horizon)
        if cur==1:
            dfs(cur+1,step+1,v,p*0.4,p,m,horizon=horizon)
        elif cur==7:
            dfs(cur-1,step+1,v,p*0.4,p,m,horizon=horizon)
        else:
            dfs(cur+1,step+1,v,p*0.4,p,m,horizon=horizon)
            dfs(cur-1,step+1,v,p*0.4,p,m,horizon=horizon)
dfs(4,0,0)
v=reduce(lambda x,y:(x[0]*x[1]+y[0]*y[1],1),zip(value,probability),(0,0))[0]
print(v)

import numpy as np

def matrixSolution():
    I=np.identity(7)
    P=np.zeros((7,7))
    for i in range(1,6):
        P[i][i]=0.2
        P[i][i+1]=0.4
        P[i][i-1]=0.4
    P[0][0]=0.6
    P[0][1]=0.4
    P[6][5]=0.4
    P[6][6]=0.6
    print(P)
    gamma=0.5
    R=np.zeros((7,1))
    R[0]=1
    R[6]=10
    V=np.linalg.inv(I-gamma*P)@R
    print(V)

matrixSolution()

def iterSolution():
    V=np.zeros((7,1))
    for k in range(0,1000):
       for i in range(0,7):
            if i==0:
               V[0][0]=1+0.5*(V[1][0]*0.4+V[0][0]*0.6)
            elif i==6:
               V[6][0]=10+0.5*(V[6][0]*0.6+V[5][0]*0.4)
            else:
                V[i][0]=0.5*(V[i-1][0]*0.4+V[i+1][0]*0.4+V[i][0]*0.2)
    print(V)

iterSolution()

