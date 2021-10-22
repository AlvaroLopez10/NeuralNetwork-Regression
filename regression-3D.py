import numpy as np
import matplotlib.pyplot as plt
import scipy.io

%matplotlib notebook

#Cargar datos
data = np.load('datos/data1.npy')

#Extraer valores
x = data[:, 0:2]
y = data[:, 2]

#Normalizar datos
x[:, 0] = x[:, 0] - np.min(x[:, 0])
x[:, 0] = x[:, 0]/np.max(x[:, 0])
x[:, 1] = x[:, 1] - np.min(x[:, 1])
x[:, 1] = x[:, 1]/np.max(x[:, 1])

#Imprimir tamaño de los datos
print(x.shape)
print(y.shape)

#Imprimir valores
f1 = plt.axes(projection='3d')
f1.scatter(x[:,0], x[:,1], y,marker='o',c='blue')

#Definir funciones auxiliares
def sigmoid(x):
    
    y = 1.0/(1.0 + np.exp(-x))
    
    return y

def emc(ym, yd):
    
    e = 0.5*np.power(ym - yd, 2)
    
    return e


#Crear clase representativa de un perceptron multicapa
class mlp:
    
    #Crear constructor
    def __init__(self, d, ne, no, ns): #(d: entrada, ni: neuronas en capa "i")
        
        #Parametros de la capa de entrada (Entrada: d, Salida: ne)  
        self.w1 = np.random.rand(ne, d) - 0.5 
        self.b1 = np.random.rand(ne, 1) - 0.5
        
        #Parametros de la capa oculta  (Entrada: ne, Salida: no)
        self.w2 = np.random.rand(no, ne) - 0.5
        self.b2 = np.random.rand(no, 1) - 0.5
        
        #Parametros de la capa de salida (Entrada: no, Salida: ns)
        self.w3 = np.random.rand(ns, no) - 0.5
        self.b3 = np.random.rand(ns, 1) - 0.5
        
    #Funcion forward (Paso hacia adelante)
    def forward(self, x):
        
        #Capa de entrada
        h1 = np.dot(self.w1, x) + self.b1  #(ne, 1)
        y1 = sigmoid(h1)  #(ne, 1)
        
        #Capa oculta
        h2 = np.dot(self.w2, y1) + self.b2 #(no, 1)
        y2 = sigmoid(h2)  #(no, 1)
        
        #Capa de salida
        h3 = np.dot(self.w3, y2) + self.b3  #(ns, 1)
        ym = sigmoid(h3)  #(ns, 1)
        
        return ym
    
    #Funcion de entrenamiento
    def train(self, x, y, Lr, epoch):
        
        #Lazo de epocas
        for i in range(epoch):
            
            #Lazo de datos
            for j in range(x.shape[0]):
                
                #Obtener entrada
                x_in = x[j, :]  #(d,)
                x_in= x_in[:, np.newaxis]  #(d,1)
                
                #Capa de entrada
                h1 = np.dot(self.w1, x_in) + self.b1  #(ne, 1)
                y1 = sigmoid(h1)  #(ne, 1)

                #Capa oculta
                h2 = np.dot(self.w2, y1) + self.b2 #(no, 1)
                y2 = sigmoid(h2)  #(no, 1)

                #Capa de salida
                h3 = np.dot(self.w3, y2) + self.b3  #(ns, 1)
                ym = sigmoid(h3)  #(ns, 1)
                
                #Obtener salida deseada
                yd = y[j]
                
                #Calcular error
                e = emc(ym, yd)
                
                #Gradientes de la funcion de error
                de_ym = ym - yd  #(ns, 1)
                
                #Gradientes de la capa de salida
                dym_h3 = ym*(1 - ym)  #(ns, 1) 
                dh3_w3 = y2  #(no, 1)
                dh3_b3 = 1.0  #(1)
                dh3_y2 = self.w3  #(ns, no)
                
                #Gradientes de la capa oculta
                dy2_h2 = y2*(1 - y2)  #(no, 1) 
                dh2_w2 = y1 #(ne, 1)
                dh2_b2 = 1.0 #(1)
                dh2_y1 = self.w2  #(no, ne)
                
                #Gradientes de la capa de entrada
                dy1_h1 = y1*(1 - y1)  #(ne, 1)
                dh1_w1 = x_in  #(d, 1)
                dh1_b1 = 1.0 #(1)
                
                #Obtener gradientes completos de capa de salida
                de_w3 = np.dot(de_ym*dym_h3, np.transpose(dh3_w3)) #(ns, no)
                        #(ns, 1)*(ns, 1)*(1, no)
                    
                de_b3 = de_ym*dym_h3*dh3_b3   #(ns, 1)
                        #(ns, 1)*(ns, 1)*(1)
                
                de_y2 = np.transpose(np.dot(np.transpose(de_ym*dym_h3), dh3_y2))  #(no, 1)
                         #(ns, 1)*(ns, 1)*(ns, no)
                            #(1,ns)x(ns, no) = (1.no)'
                        
                #Obtener gradientes completos de la capa oculta
                de_w2 = np.dot(de_y2*dy2_h2, np.transpose(dh2_w2))  #(no, ne)
                        #(no, 1)*(no, 1)*(ne,1)
                        #(no, 1)x(1, ne)) = (no, ne)
                    
                de_b2 = de_y2*dy2_h2*dh2_b2  #(no, 1)
                        #(no, 1)*(no, 1)*(1)
                        #(no,1)*(1)
                    
                de_y1 = np.transpose(np.dot(np.transpose(de_y2*dy2_h2), dh2_y1)) #(ne, 1)
                        #(no, 1)*(no, 1)
                        #(no, 1)x(no, ne)
                        #(1,no)x(no,ne) = (1, ne)' = (ne, 1) 
                        
                #Obtener gradientes completos de la capa de entrada
                de_w1 = np.dot(de_y1*dy1_h1, np.transpose(dh1_w1)) #(ne, d)
                        #(ne, 1)*(ne, 1)*(d,1)
                        #(ne, 1)*(1,d) = (ne, d)
                
                de_b1 = de_y1*dy1_h1*dh1_b1 #(ne, 1)
                         #(ne, 1)*(ne, 1)*(1)
                        #(ne, 1)*(1)
                        
                #Actualizar parametros (g.descendiente)
                self.w3 = self.w3 - Lr*de_w3
                self.b3 = self.b3 - Lr*de_b3
                
                self.w2 = self.w2 - Lr*de_w2
                self.b2 = self.b2 - Lr*de_b2
                
                self.w1 = self.w1 - Lr*de_w1
                self.b1 = self.b1 - Lr*de_b1


#Crear instancia de red neuronal
#redecita = mlp(2, 10, 3, 1)

#Entrenar red neuronal
redecita.train(x, y, 0.2, 1000)


#Inicializar salida del modelo
ym = np.zeros(y.shape)

#Inicializar error
e = 0.0

#Someter red neuronal a datos de entrada
for i in range(x.shape[0]):
    
    #Obtener entrada
    x_in = x[i, :]  #(d, )
    x_in = x_in[:, np.newaxis]  #(d, 1)
    
    #Someter modelo a entrada
    ym[i] = redecita.forward(x_in)
    
    #Calcular error
    e = e + emc(ym[i], y[i])
    
#Promediar error
e = e/x.shape[0]
print("El error de la red es: " + str(e))

#Graficar resultados
plt.figure(2)
f1 = plt.axes(projection='3d')
f1.scatter(x[:,0], x[:,1], y,marker='o',c='blue')
f1.scatter(x[:,0], x[:,1], ym,marker='o',c='red')


