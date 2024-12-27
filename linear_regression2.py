import pandas as pd
import matplotlib.pyplot as plt


#plt.scatter(dataset.Salary, dataset.YearsExperience)
#plt.show()

#Here I am calculating the mean squared error at a particular weight and bias
def mse(m ,b, points): 
    tot_error=0
    for i in range(len(points)):
        x=points.iloc[i].X
        y=points.iloc[i].Y
        tot_error += ((y - (m*x + b))**2)
    return (tot_error/float(len(points)))


def gradient_descent(m_curr,b_curr,L,points):
    m_gradient=0
    b_gradient=0

    n=len(points)

    for i in range(n):
        x=points.iloc[i].X
        y=points.iloc[i].Y

        m_gradient+=-(2/n)*x*(y - (m_curr*x + b_curr))
        b_gradient+=-(2/n)*( y - (m_curr*x + b_curr ))
    
    
    #Calculating the new value of bias and weight
    m_new=m_curr - m_gradient*L
    b_new=b_curr - b_gradient*L

    return (m_new,b_new)





def main():
    dataset= pd.read_csv(r'C:\Users\neelg\Desktop\ML_SELF_PROJECTS\Simple_Linear_Regression\Salary_dataset.csv',index_col=0)
    dk=pd.read_csv(r'C:\Users\neelg\Desktop\ML_SELF_PROJECTS\Simple_Linear_Regression\Salary_dataset.csv',index_col=0)
    dataset=dataset.rename(columns={'Salary':'X','YearsExperience':'Y'})
    print(dataset)
    dk=dk.rename(columns={'Salary':'X','YearsExperience':'Y'})
    from sklearn.preprocessing import MinMaxScaler
    scaler=MinMaxScaler()
    cols=['X','Y']
    dk[cols]=scaler.fit_transform(dk[cols])
    




    L=0.5
    m=0
    b=0

    epochs=1000
    for i in range(epochs+1):
        m,b=gradient_descent(m,b,L,dk)

        if(i%50==0):
            
            print(f"Epochs: {i}   weight: {m}  bias: {b}   MSE: {mse(m,b,dk)}")
    
    mi_X=scaler.data_min_[0]
    mi_Y=scaler.data_min_[1]
    ran_X=scaler.data_range_[0]
    ran_Y=scaler.data_range_[1]

    om=m*(ran_Y/ran_X)
    ob=b*ran_Y + mi_Y  - m*(mi_X * (ran_Y/ran_X)) 

    print('-------------------------------\nCOMPLETED')

    print(f"Final Results [ weight: {om} bias: {ob} MSE: {mse(om,ob,dataset)} ]")

    plt.scatter(dataset.X,dataset.Y, color="pink")
    plt.plot(list(range(35000,150000)), [om*x + ob for x in range(35000,150000)],color="black")
    plt.show()



main()
  
