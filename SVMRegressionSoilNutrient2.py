"""
@author:Zhang Yue
@date  :2023/1/23:11:27
@IDE   :PyCharm
"""
import sys
import os
from osgeo import gdal
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split  # 用来划分训练集和测试集
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from numpy import inf
from sklearn import svm
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
import xlwt
import xlrd
import math
from sklearn.model_selection import cross_val_score
from osgeo import ogr
from osgeo.gdalconst import *
from sklearn.gaussian_process import GaussianProcessRegressor
import pykrige.kriging_tools as kt
from pykrige.ok import OrdinaryKriging
from numpy import genfromtxt

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
    print('========calculate regression metrics by SVM========')
    print('ME: {:.6f}'.format(mean_error(y_true, y_pred)))
    print('MAE: {:.6f}'.format(mean_absolute_error(y_true, y_pred)))
    print('RMSE: {:.6f}'.format(root_mean_squared_error(y_true, y_pred)))
    # print('R2: {:.6f}'.format(r2_score(y_true, y_pred)))
    print('R2:', round((R2(y_true, y_pred)[0]), 6))
    #print('MSE: {:.6f}'.format(mean_squared_error(y_true, y_pred)))
    print('MAPE: {:.6f}'.format(mean_absolute_percentage_error(y_true, y_pred)))


def saveResidualTable(listXCoor, listYCoor, true, pred, savePathAndName):
    trueD = np.array(true).reshape(len(true), 1).flatten()
    predD = pred.reshape(pred.shape[0], 1).flatten()
    arrayListXCoor = np.array(listXCoor)
    trueV = np.array(listYCoor)
    # print(type(trueD), trueD.shape, trueD[0:10], type(predD), predD.shape, predD[0:10])
    # print(type(arrayListXCoor), arrayListXCoor.shape, arrayListYCoor.shape)
    differenceValue = trueD - predD
    differenceValue = np.array(differenceValue).reshape(differenceValue.shape[0], -1)#.flatten()
    # print('dfghjk', type(differenceValue), differenceValue.shape, differenceValue[0:10])
    saveArray = np.hstack((arrayListXCoor, trueV))
    saveArray = np.hstack((saveArray, predD.reshape(differenceValue.shape[0], -1)))
    # print(saveArray.shape, type(saveArray))
    saveArray1 = np.hstack((saveArray, differenceValue))
    # 第一列X坐标，第二列Y坐标，第三列预测值，第四列预测值，第五列残差
    saveNdarray(saveArray1, savePathAndName)

def saveNdarray(data, path):
  f = xlwt.Workbook() # 创建工作簿
  sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True) # 创建sheet
  [h, l] = data.shape # h为行数，l为列数
  for i in range(h):
    for j in range(l):
      sheet1.write(i, j, data[i, j])
  f.save(path)


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

def extractRasterDataValuesToPoitsShpFunc(inputSHP, InputRasterFolder, ExtractRasterName, saveExcelPathAndName):
    # 设置Excel编码
    file = xlwt.Workbook('encoding = utf-8')
    # 创建sheet工作表
    sheet1 = file.add_sheet('sheet1',cell_overwrite_ok=True)
    #改变工作空间
    #############获取矢量点位的经纬度
    #设置driver
    driver = ogr.GetDriverByName('ESRI Shapefile')
    #打开矢量
    ds = driver.Open(inputSHP, 0)
    #获取图层
    layer = ds.GetLayer()
    #获取要素及要素地理位置
    xValues = []
    yValues = []
    feature = layer.GetNextFeature()
    while feature:
        geometry = feature.GetGeometryRef()
        x = geometry.GetX()
        y = geometry.GetY()
        xValues.append(x)
        yValues.append(y)
        feature = layer.GetNextFeature()
    #获取点位所在像元的栅格值
    #读取栅格
    #获取注册类
    #打开栅格数据
    input_folder_list=os.listdir(InputRasterFolder) #读取文件夹里所有文件
    tif_files=list() #创建一个只装tif格式的列表
    # for filename in input_folder_list:  #遍历
    #     if os.path.splitext(filename)[1] == '.tif':  #不管文件名里面有多少个tif，都只认最后一个tif
    #         tif_files.append(filename)  #将文件夹里的tif文件加入只有tif的列表
    filename = ExtractRasterName
    tif_files.append(filename)
    print(tif_files)
    sheet1.write(0, 0, "x_coor") #excel表的第1列为经度  Lon
    sheet1.write(0, 1, "y_coor") #excel表的第2列为纬度  Lat
    for i in range(0,len(tif_files)): #遍历tif
        sheet1.write(0, i+2, filename) #在表格第一行设置列名
        ds = gdal.Open(InputRasterFolder + '\\' + tif_files[i], GA_ReadOnly)
        #获取行列、波段
        # rows = ds.RasterYSize
        # cols = ds.RasterXSize
        # bands = ds.RasterCount
        #获取放射变换信息
        transform = ds.GetGeoTransform()
        xOrigin = transform[0]
        yOrigin = transform[3]
        pixelWidth = transform[1]
        pixelHeight = transform[5]
        #
        values = []
        for j in range(len(xValues)):       #遍历所有点
            x = xValues[j]
            y = yValues[j]
            #获取点位所在栅格的位置
            xOffset = int((x - xOrigin) / pixelWidth)
            yOffset = int((y - yOrigin) / pixelHeight)
            band = ds.GetRasterBand(1) #读取影像
            data = band.ReadAsArray(xOffset, yOffset, 1, 1) #读出从(xoffset,yoffset)开始，大小为(xsize,ysize)的矩阵
            value = str(data[0, 0])
            #将数据经纬度和对应栅格数值写入excel表
            sheet1.write(j + 1, 0, x)    #第j+1行，第1列
            sheet1.write(j + 1, 1, y)    #第j+1行，第2列
            sheet1.write(j + 1, i+2, value)
    file.save(saveExcelPathAndName) #保存表格

def KrigingInterpolate(InputExcelPathAndName, X_coordinateField, Y_coordinateField, Sample_valueField,
                       tifDataPath, tifName, resultDataSetPath, outputKrigingTifName):

    Data = pd.read_excel(InputExcelPathAndName)
    Points = Data.loc[:, [X_coordinateField, Y_coordinateField]].values
    Values = Data.loc[:, [Sample_valueField]].values
    Points1 = np.array(Points)
    Values1 = np.array(Values)
    # 数据去空值
    Points1 = np.nan_to_num(Points1)
    Points1[Points1 == inf] = np.finfo(np.float16).max
    Values1 = np.nan_to_num(Values1)
    Values1[Values1 == inf] = np.finfo(np.float16).max
    # 读取遥感影像数据
    run = GRID()
    # 这一个没有参与运算，主要为了读取它的行列数、投影信息、坐标系和noData值
    rows, cols, geotransform, projection, noDataValue = Readxy(tifDataPath + tifName)
    # print(rows, cols, geotransform, projection, noDataValue)
    nXsize = cols
    nYsize = rows
    # **********************************//
    dataset = gdal.Open(tifDataPath + tifName)  # 打开tif
    adfGeoTransform = dataset.GetGeoTransform()  # 读取地理信息
    # 左上角地理坐标
    # print(adfGeoTransform[0])
    # print(adfGeoTransform[3])
    # 右下角地理坐标
    px = adfGeoTransform[0] + nXsize * adfGeoTransform[1] + nYsize * adfGeoTransform[2]
    py = adfGeoTransform[3] + nXsize * adfGeoTransform[4] + nYsize * adfGeoTransform[5]
    # px = adfGeoTransform[0] + nYsize * 150 + nXsize * adfGeoTransform[2]
    # py = adfGeoTransform[3] + nYsize * adfGeoTransform[4] - nXsize * 150
    # print(px)
    # print(py)
    OK = OrdinaryKriging(
        Points1[:, 0],
        Points1[:, 1],
        Values1[:, 0],
        variogram_model="linear",
        verbose=False,
        enable_plotting=False,
    )
    gridx = np.arange(adfGeoTransform[0], px, adfGeoTransform[1])
    gridy = np.arange(adfGeoTransform[3], py, adfGeoTransform[5])
    # print(type(gridy), gridx)
    z, ss = OK.execute("grid", gridx, gridy)
    run.write_img(resultDataSetPath + '//' + outputKrigingTifName + '.tif',
                  projection, geotransform, z)
    return z

def removeFile(removeTifFileFolder, removeTifName):
    filePathAndName = removeTifFileFolder + '//' + removeTifName
    os.remove(filePathAndName)

def removeFileByPathAndName(removeTifFileFolderAndName):
    os.remove(removeTifFileFolderAndName)

if __name__ == "__main__":
    # **********************************程序开始********************************** #
    print("start")
    samplePath = 'E:\\RemoteSensing\\XYKH\\随机森林回归土壤水分数据\\2018gz\\'
    sample = pd.read_csv(samplePath + '2018gzyd_ALL.csv', encoding='utf-8')

    dfSample = sample[['x_coor', 'y_coor', 'pH', 'SOM', 'TN', 'AP', 'AK', 'Elevation', 'Slope', 'Aspect', 'Planc',	'Profic',
                       'TPI', 'SR', 'TWI', 'NDVI', 'MAP', 'MAT', 'EVI']]

    # 影响因子-样本数据
    listSample = [[x, y, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12] for x, y, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12
           in zip(dfSample['x_coor'], dfSample['y_coor'], dfSample['Elevation'],
                  dfSample['Slope'], dfSample['Aspect'], dfSample['Planc'],
                  dfSample['Profic'], dfSample['TPI'], dfSample['SR'],
                  dfSample['TWI'], dfSample['NDVI'], dfSample['MAP'],
                  dfSample['MAT'], dfSample['EVI'])]
    # print(type(listSample))
    # print('listSample:', listSample[0:2])

    # **********************************选择计算的变量label********************************** #
    # 标签数据'pH', 'SOM', 'TN', 'AP', 'AK'
    calIndex = 'AK'
    listLabel = sample[[calIndex]].values.tolist()

    # 数据标准化
    # 计算均值，方差
    scaler = preprocessing.StandardScaler().fit(listSample)
    # 标准化
    listSampleScale = scaler.transform(listSample)

    # **********************************输入输出路径设置********************************** #
    run = GRID()
    inputDatapath = r'E:\\RemoteSensing\\XYKH\\随机森林回归土壤水分数据\\2018gz\\DataSet\\'
    outputDataPath = r'E:\\RemoteSensing\\XYKH\\随机森林回归土壤水分数据\\2018gz\\resultTIFF\\SVM_' + calIndex + r'\\'# 每次计算需要修改这里的文件夹名（需要提前新建文件夹），以免覆盖文件
    studyAreaShpPathAndName = r"E:\\RemoteSensing\\XYKH\\随机森林回归土壤水分数据\\2018gz\\hjxbl\\gz_84_UTM.shp"
    samplePointsShpPathAndName = r'E:\\RemoteSensing\\XYKH\\随机森林回归土壤水分数据\\2018gz\\2018gzyd\\gzyd2018_ALL.shp'

    # **********************************读取地理信息********************************** #
    # 这一个没有参与运算，主要为了读取它的行列数、投影信息、坐标系和noData值
    rows, cols, geotransform, projection, noDataValue = Readxy(inputDatapath + 'NDVI.tif')
    print(rows, cols, geotransform, projection, noDataValue)
    nXsize = cols
    nYsize = rows

    dataset = gdal.Open(inputDatapath + 'NDVI.tif')  # 打开tif
    adfGeoTransform = dataset.GetGeoTransform()  # 读取地理信息

    # **********************************读取多个变量数据********************************** #
    # 1，读取高程
    _, _, Elevation, _, _ = run.read_img(inputDatapath + "Elevation.tif")
    Elevation = np.array(Elevation)
    # 2，读取坡度
    _, _, Slope, _, _ = run.read_img(inputDatapath + "Slope.tif")
    Slope = np.array(Slope)
    # 3，读取坡向
    _, _, Aspect, _, _ = run.read_img(inputDatapath + "Aspect.tif")
    Aspect = np.array(Aspect)
    # 4，读取Planc
    _, _, Planc, _, _ = run.read_img(inputDatapath + "Planc.tif")
    Planc = np.array(Planc)
    # 5，读取Profic
    _, _, Profic, _, _ = run.read_img(inputDatapath + "Profic.tif")
    Profic = np.array(Profic)
    # 6，读取TPI
    _, _, TPI, _, _ = run.read_img(inputDatapath + "TPI.tif")
    TPI = np.array(TPI)
    # 7，读取SR
    _, _, SR, _, _ = run.read_img(inputDatapath + "SR.tif")
    SR = np.array(SR)
    # 8，读取TWI
    _, _, TWI, _, _ = run.read_img(inputDatapath + "TWI.tif")
    TWI = np.array(TWI)
    # 9，读取NDVI
    _, _, NDVI, _, _ = run.read_img(inputDatapath + "NDVI.tif")
    NDVI = np.array(NDVI)
    # 10，读取MAP
    _, _, MAP, _, _ = run.read_img(inputDatapath + "MAP.tif")
    MAP = np.array(MAP)
    # 11，读取MAT
    _, _, MAT, _, _ = run.read_img(inputDatapath + "MAT.tif")
    MAT = np.array(MAT)
    # 12，读取EVI
    _, _, EVI, _, _ = run.read_img(inputDatapath + "EVI.tif")
    EVI = np.array(EVI)
    # 判断维度是否相同
    assert(EVI.shape == NDVI.shape)
    assert (MAT.shape == NDVI.shape)
    assert (MAP.shape == NDVI.shape)
    assert (TWI.shape == NDVI.shape)
    assert (SR.shape == NDVI.shape)
    assert (TPI.shape == NDVI.shape)
    assert (Profic.shape == NDVI.shape)
    assert (Planc.shape == NDVI.shape)
    assert (Aspect.shape == NDVI.shape)
    assert (Slope.shape == NDVI.shape)
    assert (Elevation.shape == NDVI.shape)
    # ****************************随机森林模型拟合**************************** #
    # 划分训练集和测试集 X是自变量，Y是因变量
    X_train, X_test, y_train, y_test = train_test_split(np.array(listSampleScale), listLabel, test_size=0.2, random_state=22)
    # print(len(X_train), len(X_test), len(y_train), len(y_test))
    # print(X_train[0:2], X_test[0:2], X_test[ 0:2 ,0:2], y_train[0:2], y_test[0:2])
    listXCoor = X_test[ : ,0:2]# XY坐标
    listXCoor = np.array(listXCoor)
    listYCoor = np.array(y_test)# 测试用的真实值
    # print(listXCoor.shape)
    # print(listYCoor.shape)

    # 创建随机森林回归
    # SVM回归参数参考网页【https://blog.csdn.net/weixin_42279212/article/details/121550052】
    model_SVMRegressor = svm.SVR(kernel="rbf", gamma=0.1, C=100)
    # 调用fit训练
    model_SVMRegressor.fit(np.array(X_train).reshape(len(X_train),-1), np.array(y_train).reshape(len(X_train), -1))
    # 预测未知数据
    y_pred = model_SVMRegressor.predict(X_test)
    # print(results[0])
    # ****************************计算精度**************************** #
    # importances = list(model_SVMRegressor.feature_importances_)
    # print("\n特征权重值：", importances)
    regression_metrics_by_sklearn(y_test, y_pred)
    # savePathAndName = r'E:\\RemoteSensing\\XYKH\\随机森林回归土壤水分数据\\2018gz\\resultDataSet\\test.csv'
    # saveResidualTable(listXCoor, listYCoor, y_test, y_pred, savePathAndName)
    # print('\nCVS_10:', cross_val_score(model_RandomForestRegressor, y_test, y_pred, cv=10, scoring="r2"))

    # ****************************预测数据整理**************************** #
    listSampleTemp = []  # 用于存储每个像素的（X，Y）坐标
    for i in tqdm(range(0, nYsize)):
        row = []
        for j in range(0, nXsize):
            if (NDVI[i, j] == noDataValue):  # 处理图像中的noData
                features = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                row.append(features)
            else:
                px = adfGeoTransform[0] + i * adfGeoTransform[1] + j * adfGeoTransform[2]
                py = adfGeoTransform[3] + i * adfGeoTransform[4] + j * adfGeoTransform[5]
                ElevationValue = Elevation[i, j]
                SlopeValue = Slope[i, j]
                AspectValue = Aspect[i, j]
                PlancValue = Planc[i, j]
                ProficValue = Profic[i, j]
                TPIValue = TPI[i, j]
                SRValue = SR[i, j]
                TWIValue = TWI[i, j]
                NDVIValue = NDVI[i, j]
                MAPValue = MAP[i, j]
                MATValue = MAT[i, j]
                EVIValue = EVI[i, j]

                features = [px, py, ElevationValue, SlopeValue, AspectValue, PlancValue, ProficValue, TPIValue, SRValue,
                            TWIValue, NDVIValue, MAPValue, MATValue, EVIValue]  # 每个像素的经纬度
                row.append(features)
                # print(features)
        listSampleTemp.append(row)

    listSampleTemp = np.array(listSampleTemp)
    # print(listSampleTemp.shape)
    predictiveFeatures = np.reshape(listSampleTemp,(-1,14))

    predictiveFeatures = np.nan_to_num(predictiveFeatures)
    predictiveFeatures[predictiveFeatures == inf] = np.finfo(np.float64).max
    # 均值，方差
    scaler2 = preprocessing.StandardScaler().fit(predictiveFeatures)
    # 标准化
    predictiveFeaturesScale = scaler2.transform(predictiveFeatures)

    # **********************************预测未知数据********************************** #
    resultsLabel = model_SVMRegressor.predict(predictiveFeaturesScale)
    # print(type(resultsLabel), resultsLabel)
    arrayRFresult = resultsLabel.reshape(rows, -1)

    # 保存结果为tiff
    outputName = r'SVM_' + calIndex + r'_regression'
    run.write_img(outputDataPath + outputName + '.tif',
                  projection, geotransform, arrayRFresult)
    print("SVM Finish!")

    # 根据shp裁剪tif
    dst = outputDataPath + outputName + r"_clip.tif"# 裁剪图像保存完整路径（包括文件名）
    src = outputDataPath + outputName + '.tif'# 待裁剪的影像完整路径（包括文件名）
    # shp = r"E:\\RemoteSensing\\XYKH\\随机森林回归土壤水分数据\\2018gz\\hjxbl\\gz_84_UTM.shp"  # 矢量文件的完整路径
    shp = studyAreaShpPathAndName

    ds = gdal.Warp(dst,  # 裁剪图像保存完整路径（包括文件名）
                   src,  # 待裁剪的影像
                   # warpMemoryLimit=500 内存大小M
                   format='GTiff',  # 保存图像的格式
                   cutlineDSName=shp,  # 矢量文件的完整路径
                   cropToCutline=True,
                   copyMetadata=True,
                   creationOptions=['COMPRESS=LZW', "TILED=True"],
                   dstNodata=-9999)

    # print("Clip Finish!")

    # **********************************提取真实值和预测值到Excel********************************** #
    # inputSHP = r'E:\\RemoteSensing\\XYKH\\随机森林回归土壤水分数据\\2018gz\\2018gzyd\\gzyd2018_ALL.shp'  # 点数据文件
    inputSHP = samplePointsShpPathAndName
    # InputRasterFolder = r'E:\\RemoteSensing\\XYKH\\随机森林回归土壤水分数据\\2018gz\\resultDataSet'  # 放栅格数据的文件夹
    InputRasterFolder = outputDataPath
    # ExtractRasterName = r'NDVI_Resample150m.tif'
    ExtractRasterName = outputName + r"_clip.tif"
    saveExcelPathAndName = inputDatapath + r'pointToRaster.xls'
    extractRasterDataValuesToPoitsShpFunc(inputSHP, InputRasterFolder, ExtractRasterName, saveExcelPathAndName)
    # 如果报错AttributeError: 'NoneType' object has no attribute 'GetGeoTransform'， 就是文件路径错误
    # 读取真实值excel和预测值excel

    Data = pd.read_excel(saveExcelPathAndName)
    listXCoor1 = Data[['x_coor']].values.tolist()
    arrayXCoor = np.array(listXCoor1)
    listYCoor1 = Data[['y_coor']].values.tolist()
    arrayYCoor = np.array(listYCoor1)
    dfTolistPred = Data[[ExtractRasterName]].values.tolist()
    arrayPred = np.array(dfTolistPred)
    arrayTrue = np.array(listLabel)
    # print(dfTolistTrue, arrayPred, dfTolistTrue.shape, arrayPred.shape)
    # print(type(arrayTrue), arrayTrue.shape, type(dfTolistPred))
    arrayDiff = arrayTrue - arrayPred
    # print(type(arrayDiff), arrayDiff.shape)

    # 设置Excel编码
    file = xlwt.Workbook('encoding = utf-8')
    # 创建sheet工作表
    sheet1 = file.add_sheet('sheet1', cell_overwrite_ok=True)
    sheet1.write(0, 0, "X_coor")  # excel表的第1列为经度
    sheet1.write(0, 1, "Y_coor")  # excel表的第2列为纬度
    sheet1.write(0, 2, "True")  # excel表的第3列为真实值
    sheet1.write(0, 3, "Pred")  # excel表的第4列为预测值
    sheet1.write(0, 4, "Diff")  # excel表的第5列为差值
    for i in range(0, 5):
        for j in range(0, arrayDiff.shape[0]):
            x = arrayXCoor[:, 0][j]
            y = arrayYCoor[:, 0][j]
            value1 = arrayTrue[:, 0][j]
            value2 = arrayPred[:, 0][j]
            value3 = arrayDiff[:, 0][j]
            # 将数据经纬度和对应栅格数值写入excel表
            sheet1.write(j + 1, 0, x)  # 第j+1行，第1列
            sheet1.write(j + 1, 1, y)  # 第j+1行，第2列
            sheet1.write(j + 1, 2, str(value1))  # 第j+1行，第3列
            sheet1.write(j + 1, 3, value2)  # 第j+1行，第4列
            sheet1.write(j + 1, 4, value3)  # 第j+1行，第5列
    saveExcelPathAndName1 = outputDataPath + r"residualErrorTable.xls"
    file.save(saveExcelPathAndName1) # 保存表格
    print("Residual Extract To Excel Finish!")

    # **********************************克里金插值残差值********************************** #
    InputExcelPathAndName = saveExcelPathAndName1 # "E:\\RemoteSensing\\XYKH\\随机森林回归土壤水分数据\\2018gz\\2018gzyd_ALL.xlsx"
    X_coordinateField = 'X_coor'
    Y_coordinateField = 'Y_coor'
    Sample_valueField = 'Diff'
    # tifDataPath = 'E:\\RemoteSensing\\XYKH\\随机森林回归土壤水分数据\\2018gz\\DataSet\\'
    tifDataPath = inputDatapath
    tifName = 'NDVI_Resample150m.tif'
    # resultDataSetPath = 'E:\\RemoteSensing\\XYKH\\随机森林回归土壤水分数据\\2018gz\\resultDataSet'
    resultDataSetPath = outputDataPath
    outputKrigingTifName = 'KrigingInterpolate_Diff'

    arrayResidualKriging = KrigingInterpolate(InputExcelPathAndName, X_coordinateField, Y_coordinateField,
                           Sample_valueField, tifDataPath, tifName, resultDataSetPath, outputKrigingTifName)
    arrayResidualKrigingReSample = np.repeat(arrayResidualKriging, 5, axis=1).repeat(5, axis=0)
    arrayResidualRF = arrayRFresult[:2225, :2634] + arrayResidualKrigingReSample[:2225, :2634]
    # print(arrayResidualRF.shape)

    # 保存结果为tiff
    outputNameRRF = r'ResidualSVM_' + calIndex + r'_regression'
    run.write_img(outputDataPath + outputNameRRF + '.tif',
                  projection, geotransform, arrayResidualRF)
    print("Residual Kriging Finish!")

    # **********************************根据shp裁剪tif********************************** #
    dstRRF = outputDataPath + outputNameRRF + r"_clip.tif"  # 裁剪图像保存完整路径（包括文件名）
    srcRRF = outputDataPath + outputNameRRF + '.tif'  # 待裁剪的影像完整路径（包括文件名）
    # shpRRF = r"E:\\RemoteSensing\\XYKH\\随机森林回归土壤水分数据\\2018gz\\hjxbl\\gz_84_UTM.shp"  # 矢量文件的完整路径
    shpRRF = studyAreaShpPathAndName

    ds = gdal.Warp(dstRRF,  # 裁剪图像保存完整路径（包括文件名）
                   srcRRF,  # 待裁剪的影像
                   # warpMemoryLimit=500 内存大小M
                   format = 'GTiff',  # 保存图像的格式
                   cutlineDSName = shpRRF,  # 矢量文件的完整路径
                   cropToCutline = True,
                   copyMetadata = True,
                   creationOptions = ['COMPRESS=LZW', "TILED=True"],
                   dstNodata = -9999)

    # print("Clip Finish!")

    # **********************************删除temp数据********************************** #
    removeFileByPathAndName(resultDataSetPath + '\\' +outputKrigingTifName + '.tif') # 删除克里金插值结果
    removeFileByPathAndName(outputDataPath + outputName + '.tif') # 删除随机森林裁剪前影像
    removeFileByPathAndName(outputDataPath + outputNameRRF + '.tif') # 删除随机森林克里金裁剪前影像
    print("Remove Finish!")