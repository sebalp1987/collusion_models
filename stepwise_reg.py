import numpy
from sklearn import linear_model
from mpmath import sqrt
import matplotlib.pyplot as plot
import csv
import os

from sklearn.metrics import r2_score
import statsmodels.api as sm
_author_ = 'Sebastian Palacio'

class stepwise_regression:


    def setpwise_reg(xList, target, names):

        def xattrSelect(x,idxSet):  # Toma la matrix X de la regresión y = betaX y la convierte en una lista de listas y retorna un subconjunto que contiene columnass en idxSet
            xOut = []
            for row in x:
                xOut.append([row[i] for i in idxSet])
            return (xOut)


        # dividimos los atributos y el target en training y test
        indices = range(len(xList))
        xListTest = [xList[i] for i in indices if i % 3 == 0]
        xListTrain = [xList[i] for i in indices if i % 3 != 0]
        targetTest = [target[i] for i in indices if i % 3 == 0]
        targetTrain = [target[i] for i in indices if i % 3 != 0]

        # armamos columnas de atributos para ir probando que columnas quedan mejor
        attributeList = []
        index = range(len(xList[1]))  # toma la cantidad de columnas
        indexSet = set(
            index)  # crea una variable ordenada en función de los valores que tenga la variable de la cual se origina
        indexSeq = []
        oosError = []
        aasError = []
        betas = []
        bbs = []
        sse = []
        for i in index:
            attSet = set(attributeList)  # marca las columnas que se van descartando

            attTrySet = indexSet - attSet  # marca las columnas que se van utilizando (el contrario al attSet)

            attTry = [ii for ii in attTrySet]  # paso attTrySet a una lista [0.0, 1.0...31]

            errorList = []
            absError = []
            attTemp = []
            betaList = []
            seList = []
            ####
            for iTry in attTry:
                attTemp = [] + attributeList  # comienza como una columna de [1...31], luego itera nuevamente con dos columnas [31,1..31], luego con tres [31,21,1...31] siempre iterando la ultima columna (esto lo hace usando las columnas que genera attSet)
                attTemp.append(iTry)

                ####
                xTrainTemp = xattrSelect(xListTrain, attTemp)
                xTestTemp = xattrSelect(xListTest, attTemp)
                ####
                xTrain = numpy.array(xTrainTemp)
                yTrain = numpy.array(targetTrain)
                xTest = numpy.array(xTestTemp)
                yTest = numpy.array(targetTest)

                fileQModel = linear_model.LinearRegression()
                fileQModel.fit(xTrain, yTrain)
                ols = sm.OLS(yTrain, xTrain)
                result = ols.fit()

                rmsError = numpy.linalg.norm((yTest - fileQModel.predict(xTest)), 2)*numpy.linalg.norm((yTest - fileQModel.predict(xTest)), 2) / len(yTest)  # este va calculando todo el tiempo el RMS
                absVal = numpy.linalg.norm((yTest - fileQModel.predict(xTest)), 1)/len(yTest)
                betas = fileQModel.coef_
                se = result.HC0_se
                errorList.append(rmsError)  # arma una columna de los RMS
                absError.append(absVal)
                betaList.append(betas)
                seList.append(se)
                attTemp = []

            iBest = numpy.argmin(errorList)  # toma el mínimo RMS
            attributeList.append(attTry[iBest])  # toma la mejor combinación de columnas de donde salió el mínimo RMS
            oosError.append(errorList[iBest])  # va armando una columna de mejores RMS
            aasError.append(absError[iBest])
            bbs.append(betaList[iBest])
            sse.append(seList[iBest])



        print('Out of Sample error versus attribute set Size ', oosError)
        indexBest = oosError.index(min(oosError))
        print('index Best (min oosError = ', indexBest)
        print('Out of Sample absolute error versus attribute set Size ', aasError)
        print('\n Best attribute indices ', attributeList)
        namesList = [names[i] for i in attributeList]
        namesList = namesList[:indexBest+1]
        print('\n Best Attribute Names ', namesList)
        print('betas',bbs[indexBest])
        print('se',sse[indexBest])
        # numpy.savetxt("results\stepwise\oosError.csv", oosError, delimiter=",")
        # numpy.savetxt("results\stepwise\\attributeList.csv", attributeList, delimiter=",")






        '''
        Esto arroja no solo el nombre de los atributos, sino su orden de calidad en términos de predicción. !!!!!
        '''

        # PLOT ERROR VERSUS NUMBER OF ATTRIBUTES#####################################################################################
        x = range(len(oosError))

        plot.plot(x, oosError,
                  'k')  # plot(x,y,keyword arguments) if i put 'k' is black color lines colors=('b', 'g', 'r', 'c', 'm', 'y', 'k', 'w')
        plot.xlabel('Attributes')
        plot.ylabel('MSE')
        plot.title('Stepwise Regression')
        plot.show()
        '''
        Vemos en este ejemplo que el error cae a medida que agregamos nuevos atributos, hasta 25 atributos, luego comienza a subir
        '''

        # PLOT HISTOGRAM OF OUT SAMPLE ERRORS FOR BEST NUMBER OF ATTRIBUTES###########################################################
        '''
        We have to identify index corresponding to min value error, retrain with the corresponding attributes
        and use the resulting model to predict against out sample data.
        Plot errors
        '''

        '''
        attributesBest = attributeList[1:(indexBest + 1)]

        xTrainTemp = xattrSelect(xListTrain, attributesBest)  # Arma una matrix X con los mejores atributos (con el conjunto de 2/3)
        xTestTemp = xattrSelect(xListTest, attributesBest)  # Arma una matrix X con los mejores atributos (con el conjunto de prueba de 1/3)

        xTrain = numpy.array(xTrainTemp)
        xTest = numpy.array(xTestTemp)

        fileQModel = linear_model.LinearRegression()
        fileQModel.fit(xTrain, yTrain)
        errorVector = yTest - fileQModel.predict(xTest)  # hace un vector en función del yTest y de la predicción de Y hecha con el xTest (basado en el modelo lineal con xTrain e yTrain)
        plot.hist(errorVector)
        plot.xlabel('límite del error (bin boundaries)')
        plot.ylabel('Count')
        plot.show()'''
        '''
        Esta figura nos muestra un histograma de la predicción de error mediante forward-stepwise prediction para predecir el viaja-noviaja.
        '''

        # Scatter Plot of actual versus predicted
        r2 = r2_score(yTest, fileQModel.predict(xTest))
        print('r2 Stepwise', r2)
        plot.scatter(fileQModel.predict(xTest), yTest, s=100, alpha=0.1)  # s es el intercept, alpha la pendiente de una línea de ejemplo
        plot.xlabel('Predicted Values')
        plot.ylabel('Real Values')
        plot.title('Stepwise Regression')
        plot.show()

        '''
        Esta figura muestra un scatter plot de los verdaderos valores versus los predichos de la variable Y. Idealmente todos los valores
        deberían caer en una línea de 45º donde los valores predichos son iguales a los reales. Como los valores son enteros, se muestran como una
        línea horizontal. Cuanto más oscuro, implica que hay una acumulación de puntos, y por tanto las predicciones son bastante buenas
        '''

        return namesList
