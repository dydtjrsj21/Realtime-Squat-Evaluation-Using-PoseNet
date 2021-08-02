# 0. 사용할 패키지 불러오기
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.models import load_model
import numpy as np
import os 
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    
    # 1.랜덤시드 고정시키기
    np.random.seed(5)
    
    # 2. 모델 구성하기
    model = Sequential()
    model.add(Dense(30, input_dim=44, activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    # 3. 모델 학습과정 설정하기
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    for root, subdirs, files in os.walk('trainingSet'):
        for file in files:
            # 4. 데이터 준비하기
            dataset = np.loadtxt('trainingSet/'+file, delimiter=",")
            x_train = dataset[:,0:44]
            y_train = dataset[:,44]
            
            # 5. 모델 학습시키기
            model.fit(x_train, y_train, epochs=80, batch_size=64)

    # 6. 모델 저장하기
    model.save('squat_mlp_model.h5')

    # 7. 모델 평가하기
    dataForExcel = []
    for root, subdirs, files in os.walk('testSet'):
        for file in files:
            # 4. 데이터 준비하기
            dataset = np.loadtxt('testSet/'+file, delimiter=",")
            x_test = dataset[:,0:44]
            y_test = dataset[:,44]
            loss_and_metrics = model.evaluate(x_test, y_test, batch_size=64)
            print('## evaluation loss and_metrics ##')
            print(loss_and_metrics)
            predict=model.predict(x_test)
            precision=0
            recall=0
            fscore=0
            graphX=[]
            graphPrecision=[]
            graphRecall=[]
            graphFscore=[]
            std_confirm=0
            for std in [i/100 for i in range(1,100)]:
                tp_count=0
                fp_count=0
                fn_count=0
                for i in range(len(predict)):
                    if predict[i][0]>std and y_test[i]>std:
                        tp_count+=1
                    if predict[i][0]<std and y_test[i]>std:
                        fn_count+=1
                    if predict[i][0]>std and y_test[i]<std:
                        fp_count+=1
                precision=(tp_count)/(tp_count+fp_count)
                recall=(tp_count)/(tp_count+fn_count)
                fscore=2*precision*recall/(precision+recall)
                graphX.append(std)
                graphPrecision.append(precision)
                graphRecall.append(recall)
                graphFscore.append(fscore)
            print(std_confirm)

            fig = plt.figure(figsize=(10,10)) ## 캔버스 생성
            fig.set_facecolor('white') ## 캔버스 색상 설정
            ax = fig.add_subplot() ## 그림 뼈대(프레임) 생성

            ax.plot(graphX,graphPrecision,marker='o',label='Precision') ## 선그래프 생성
            ax.plot(graphX,graphRecall,marker='o',label='Recall') ## 선그래프 생성
            ax.plot(graphX,graphFscore,marker='o',label='Fscore') ## 선그래프 생성
            plt.xlabel('standard value')
            plt.ylabel('value')
            ax.legend() ## 범례
            plt.show()
            dataForExcel.append(pd.DataFrame({'std_val':graphX,'Precision':graphPrecision,'Recall':graphRecall,'Fscore':graphFscore}))
    xlxs_dir='result_Train.xlsx'
    with pd.ExcelWriter(xlxs_dir) as writer:
        dataForExcel[0].to_excel(writer)
    xlxs_dir='result_Test.xlsx'
    with pd.ExcelWriter(xlxs_dir) as writer:
        dataForExcel[1].to_excel(writer)