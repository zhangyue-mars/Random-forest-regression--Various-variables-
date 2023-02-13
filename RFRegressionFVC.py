"""
@author:Zhang Yue
@date  :2023/2/7:14:13
@IDE   :PyCharm
"""
import sys
from osgeo import gdal
import numpy as np
from numpy import inf
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split  # 用来划分训练集和测试集
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
import math
from sklearn.model_selection import GridSearchCV

class GRID:
    # 读图像文件
    def read_img(self, filename):
        dataset = gdal.Open(filename)  # 打开文件
        im_width = dataset.RasterXSize  # 栅格矩阵的列数
        im_height = dataset.RasterYSize  # 栅格矩阵的行数
        im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
        im_proj = dataset.GetProjection()  # 地图投影信息
        im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 将数据写成数组，对应栅格矩阵
        im_data = np.array(im_data)
        del dataset  # 关闭对象，文件dataset
        return im_proj, im_geotrans, im_data, im_width, im_height

    # 写文件，写成tiff
    def write_img(self, filename, im_proj, im_geotrans, im_data):
        # 判断栅格数据的数据类型
        if 'int8' in im_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in im_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32
        # 判读数组维数
        if len(im_data.shape) == 3:
            im_bands, im_height, im_width = im_data.shape
        else:
            im_bands, (im_height, im_width) = 1, im_data.shape
        # 创建文件
        driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，因为要计算需要多大内存空间
        dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
        if im_bands == 1:
            dataset.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据
        else:
            for i in range(im_bands):
                dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
        del dataset

#读取每个tiff图像的属性信息
def Readxy(RasterFile):
    ds = gdal.Open(RasterFile,gdal.GA_ReadOnly)
    if ds is None:
        print ('Cannot open ',RasterFile)
        sys.exit(1)
    cols = ds.RasterXSize
    rows = ds.RasterYSize
    band = ds.GetRasterBand(1)
    # data = band.ReadAsArray(0,0,cols,rows)
    noDataValue = band.GetNoDataValue()
    projection=ds.GetProjection()
    geotransform = ds.GetGeoTransform()
    return rows,cols,geotransform,projection,noDataValue

def root_mean_squared_error(true, pred):
    squared_error = np.square(true - pred)
    sum_squared_error = np.sum(squared_error)
    rmse_loss = np.sqrt(sum_squared_error / len(true))
    return rmse_loss

def mean_error(true, pred):
    me_loss = true - pred
    me_loss = np.sum(me_loss) / len(true)
    return me_loss

def regression_metrics_by_sklearn(y_true, y_pred):
    print('========calculate regression metrics by RF========')
    print('ME: {:.6f}'.format(mean_error(y_true, y_pred)))
    print('MAE: {:.6f}'.format(mean_absolute_error(y_true, y_pred)))
    print('RMSE: {:.6f}'.format(root_mean_squared_error(y_true, y_pred)))
    print('SkLearn-R2: {:.6f}'.format(r2_score(y_true, y_pred)))
    print('R2:', round((R2(y_true, y_pred)[0]), 6))
    #print('MSE: {:.6f}'.format(mean_squared_error(y_true, y_pred)))
    print('MAPE: {:.6f}'.format(mean_absolute_percentage_error(y_true, y_pred)))

def R2(X, Y):
    X = np.array(X)
    Y = np.array(Y)
    xBar = np.mean(X)
    yBar = np.mean(Y)
    SSR = 0
    varX = 0
    varY = 0
    for i in range(0, X.shape[0]):
        diffXXBar = X[i] - xBar
        diffYYBar = Y[i] - yBar
        SSR += (diffXXBar * diffYYBar)
        varX += diffXXBar ** 2
        varY += diffYYBar ** 2
    SST = math.sqrt(varX * varY)
    r2 = SSR / SST
    return r2

if __name__ == "__main__":
    # **********************************程序开始********************************** #
    print("start")
    samplePath = 'E:\\RemoteSensing\\XYKH\\【python】随机森林回归FVC植被覆盖度\\7-数据.xlsx'

    # 读取FVC
    sampleData = pd.read_excel(samplePath)
    listFVC = sampleData[['无人机FVC']].values.tolist()
    arrayFVC = np.array(listFVC)
    # print(arrayFVC)

    # 读取特征
    dfSample = sampleData[['EXG', 'EVI', 'NDGI', 'NDVI', 'RVI', 'SAVI', 'VDVI', '归一化B4B2',
                           '归一化B4B3', 'B1', 'B2', 'B3', 'B4', 'B5']]

    # 影响因子-样本数据
    listSample = [[f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14] for
                  f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14
                  in zip(dfSample['EXG'], dfSample['EVI'], dfSample['NDGI'],
                         dfSample['NDVI'], dfSample['RVI'], dfSample['SAVI'], dfSample['VDVI'],
                         dfSample['归一化B4B2'], dfSample['归一化B4B3'],
                         dfSample['B1'], dfSample['B2'], dfSample['B3'], dfSample['B4'], dfSample['B5'])]

    print('[EXG, EVI, NDGI, NDVI, RVI, SAVI, VDVI, 归一化B4B2, 归一化B4B3, B1, B2, B3, B4, B5]')

    # 数据标准化
    # # 计算均值，方差
    # scaler = preprocessing.StandardScaler().fit(listSample)
    # # 标准化
    # listSampleScale = scaler.transform(listSample)

    # **********************************输入输出路径设置********************************** #
    run = GRID()
    inputDatapath = r'E:\\RemoteSensing\\XYKH\\【python】随机森林回归FVC植被覆盖度\\YD-01\\'
    outputDataPath = r'E:\\RemoteSensing\\XYKH\\【python】随机森林回归FVC植被覆盖度\\outputData\\'

    listTemp = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
    # listTemp = ['01'] # 测试用
    for index in tqdm(listTemp):
        print(index)
        # **********************************读取地理信息********************************** #
        # 这一个没有参与运算，主要为了读取它的行列数、投影信息、坐标系和noData值
        rows, cols, geotransform, projection, noDataValue = Readxy(
            inputDatapath + 'B1' + '\\' + 'YD-' + index + '-450nm.tif')
        print(rows, cols, geotransform, projection, noDataValue)
        nXsize = cols
        nYsize = rows
        # **********************************读取多个变量数据********************************** #
        # 1，读取B1
        _, _, B1, _, _ = run.read_img(inputDatapath + 'B1' + '\\' +'YD-' + index + '-450nm.tif')
        arrayB1 = np.array(B1)
        _, _, B2, _, _ = run.read_img(inputDatapath + 'B2' + '\\' +'YD-' + index + '-555nm.tif')
        arrayB2 = np.array(B2)
        _, _, B3, _, _ = run.read_img(inputDatapath + 'B3' + '\\' +'YD-' + index + '-660nm.tif')
        arrayB3 = np.array(B3)
        _, _, B4, _, _ = run.read_img(inputDatapath + 'B4' + '\\' +'YD-' + index + '-720nm.tif')
        arrayB4 = np.array(B4)
        _, _, B4B2, _, _ = run.read_img(inputDatapath + 'B4B2' + '\\' +'YD-' + index + '-B4B2.tif')
        arrayB4B2 = np.array(B4B2)
        _, _, B4B3, _, _ = run.read_img(inputDatapath + 'B4B3' + '\\' +'YD-' + index + '-B4B3.tif')
        arrayB4B3 = np.array(B4B3)
        _, _, B5, _, _ = run.read_img(inputDatapath + 'B5' + '\\' +'YD-' + index + '-840nm.tif')
        arrayB5 = np.array(B5)
        _, _, EVI, _, _ = run.read_img(inputDatapath + 'EVI' + '\\' +'YD-' + index + '-EVI.tif')
        arrayEVI = np.array(EVI)
        _, _, EXG, _, _ = run.read_img(inputDatapath + 'EXG' + '\\' +'YD-' + index + '-EXG.tif')
        arrayEXG = np.array(EXG)
        _, _, NDGI, _, _ = run.read_img(inputDatapath + 'NDGI' + '\\' +'YD-' + index + '-NDGI.tif')
        arrayNDGI = np.array(NDGI)
        _, _, NDVI, _, _ = run.read_img(inputDatapath + 'NDVI' + '\\' +'YD-' + index + '-NDVI.tif')
        arrayNDVI = np.array(NDVI)
        _, _, RVI, _, _ = run.read_img(inputDatapath + 'RVI' + '\\' +'YD-' + index + '-RVI.tif')
        arrayRVI = np.array(RVI)
        _, _, SAVI, _, _ = run.read_img(inputDatapath + 'SAVI' + '\\' + 'YD-' + index + '-SAVI.tif')
        arraySAVI = np.array(SAVI)
        _, _, VDVI, _, _ = run.read_img(inputDatapath + 'VDVI' + '\\' + 'YD-' + index + '-VDVI.tif')
        arrayVDVI = np.array(VDVI)
        # 判断维度是否相同
        assert (B2.shape == B1.shape)
        assert (B3.shape == B1.shape)
        assert (B4.shape == B1.shape)
        assert (B4B2.shape == B1.shape)
        assert (B4B3.shape == B1.shape)
        assert (B5.shape == B1.shape)
        assert (EVI.shape == B1.shape)
        assert (EXG.shape == B1.shape)
        assert (NDGI.shape == B1.shape)
        assert (NDVI.shape == B1.shape)
        assert (RVI.shape == B1.shape)
        assert (SAVI.shape == B1.shape)
        assert (VDVI.shape == B1.shape)
        # ****************************随机森林模型拟合**************************** #
        # 划分训练集和测试集 X是自变量，Y是因变量
        X_train, X_test, y_train, y_test = train_test_split(np.array(listSample), arrayFVC, test_size=0.2,
                                                            random_state=22)
        print(len(X_train), len(X_test), len(y_train), len(y_test))

        # ****************************最优参数配置**************************** #
        # 随机森林中树的个数
        # n_estimators = [int(x) for x in np.linspace(start=1, stop=2000, num=200)]

        # # 每一节点考虑切分的节点数
        # max_features = ['auto', 'sqrt']

        # # 最大深度
        # max_depth = [int(x) for x in np.linspace(10, 200, num=20)]
        # max_depth.append(None)

        # # 切分一个节点最小数量
        # min_samples_split = [2, 5, 10]

        # # 每一叶子节点最小数量
        # min_samples_leaf = [1, 2, 4]
        # # Method of selecting samples for training each tree
        # bootstrap = [True, False]
        #
        # # 随机种子
        # random_state = [int(x) for x in np.linspace(10, 200, num=15)]
        # random_state.append(None)
        #
        # 创建参数组合网格
        parameters = {
                    'n_estimators': [100, 1000, 2000],
                    # 'random_state': ['None'],
                    # 'max_features': ['auto', 8],
                    # 'max_depth': [3, 5, 7, 10],
                    # 'min_samples_split': [2, 5, 10],
                    # 'min_samples_leaf': [1, 2, 4, 10],
                    # 'bootstrap': [True, False]
                     }

        # 模型构建
        model_RandomForestRegressor = RandomForestRegressor()
        model_RandomForestRegressor = GridSearchCV(model_RandomForestRegressor, parameters)

        # 调用fit训练
        model_RandomForestRegressor.fit(np.array(X_train).reshape(len(X_train), -1),
                                        np.array(y_train).reshape(len(X_train), -1))
        # 预测未知数据
        y_pred = model_RandomForestRegressor.predict(X_test)
        # ****************************计算精度**************************** #
        model_RandomForestRegressor_Importance = RandomForestRegressor(n_estimators=2000, max_features=8,
                                                                       n_jobs=4, min_samples_leaf=10)
        # 调用fit训练
        model_RandomForestRegressor_Importance.fit(np.array(X_train).reshape(len(X_train), -1),
                                                   np.array(y_train).reshape(len(X_train), -1))
        importances = list(model_RandomForestRegressor_Importance.feature_importances_)
        print("\n特征权重值：", importances)
        regression_metrics_by_sklearn(y_test, y_pred)

        # ****************************预测数据整理**************************** #
        listSampleTemp = []  # 用于存储每个像素的特征
        for i in range(0, nYsize):
            row = []
            for j in range(0, nXsize):
                B1Value = B1[i, j]
                B2Value = B2[i, j]
                B3Value = B3[i, j]
                B4Value = B4[i, j]
                B4B2Value = B4B2[i, j]
                B4B3Value = B4B3[i, j]
                B5Value = B5[i, j]
                EVIValue = EVI[i, j]
                EXGValue = EXG[i, j]
                NDGIValue = NDGI[i, j]
                NDVIValue = NDVI[i, j]
                RVIValue = RVI[i, j]
                SAVIValue = SAVI[i, j]
                VDVIValue = VDVI[i, j]
                # features = [px, py, ElevationValue, SlopeValue, AspectValue, PlancValue, ProficValue, TPIValue, SRValue,
                #             TWIValue, NDVIValue, MAPValue, MATValue, EVIValue]  # 每个像素的经纬度
                features = [EXGValue, EVIValue, NDGIValue, NDVIValue, RVIValue, SAVIValue, VDVIValue,
                            B4B2Value, B4B3Value, B1Value, B2Value, B3Value, B4Value, B5Value]  # 每个像素的经纬度

                row.append(features)
                # print(features)
            listSampleTemp.append(row)

        listSampleTemp = np.array(listSampleTemp)
        predictiveFeatures = np.reshape(listSampleTemp, (-1, 14))  # 14

        # predictiveFeatures = np.nan_to_num(predictiveFeatures)
        # predictiveFeatures[predictiveFeatures == inf] = np.finfo(np.float64).max
        # # 均值，方差
        # scaler2 = preprocessing.StandardScaler().fit(predictiveFeatures)
        # # 标准化
        # predictiveFeaturesScale = scaler2.transform(predictiveFeatures)

        # **********************************预测未知数据********************************** #
        resultsLabel = model_RandomForestRegressor.predict(predictiveFeatures)
        arrayRFresult = resultsLabel.reshape(rows, -1)

        # 保存结果为tiff
        outputName = r'RF_regression1_FVC_' + index
        run.write_img(outputDataPath + outputName + '.tif',
                      projection, geotransform, arrayRFresult)
        print("RF Finish!")