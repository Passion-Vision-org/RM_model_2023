#include <io.h>//读取文件和写入文件要用的头文件
#include <string>
#include <iostream>
#include <opencv2\opencv.hpp>
#include <opencv2\ml.hpp>
using namespace cv;
using namespace ml;
using namespace std;


//计算图像梯度
void coumputeHog(const Mat& src, vector<float>& descriptors)
{

	Size win_size = Size(20, 28);// 检测窗口大小。
	Size block_size = Size(16, 16);// 块大小，目前只支持Size(16, 16)
	Size block_stride = Size(2, 2);// 块的滑动步长，大小只支持是单元格cell_size大小的倍数。
	Size cell_size = Size(8, 8);// 单元格的大小，目前只支持Size(8, 8)。
	int nbins = 9;// 直方图bin的数量(投票箱的个数)，目前每个单元格Cell只支持9个。
	//double win_sigma = DEFAULT_WIN_SIGMA;// 高斯滤波窗口的参数。
	double threshold_L2hys = 0.2;// 块内直方图归一化类型L2 - Hys的归一化收缩率
	bool gamma_correction = true;// 是否gamma校正
	//nlevels = DEFAULT_NLEVELS; 检测窗口的最大数量
	HOGDescriptor myHog = HOGDescriptor(win_size, block_size, block_stride, cell_size, 9);
	myHog.compute(src, descriptors);
}


int main()
{
	Mat img_src;//用来承接数据集样本的图片
	Mat training_Data;//存数据集样本图片的hog特征
	Mat label_Data;//存数据集样本图片的hog特征
	const int imagesSum = 23387;//数据集有多少照片	7400 28519
	const int img_information = 756;//一张图片共有多少信息
	const int classSum = 8; //共有几类
	//float (*trainingData)[img_information] = new float[imagesSum][img_information]();
	static float tainingData[imagesSum][img_information] = { {0} };//一行一张图片的数据，多少张数据集图片多少行，一张图片多少信息多少列
	//float(*labels)[classSum] = new float[imagesSum][classSum]();
	static float labels[imagesSum][classSum] = { {0} };//一行一张图片的数据，多少张数据集图片多少行，多少分类多少列

	//==========================读取图片创建训练数据和标签==============================
	int k = 0;//用于往数据集里写数据时换行
	for (int i = 0; i < classSum; i++)
	{
		//目标文件夹路径
		std::string inPath = "/train/";	
		char temp[256];
		sprintf_s(temp, "%d", i);	//因为文件夹是从1开始的，所以i不可以是0
		inPath = inPath + temp + "\\*.jpg";	//string+char,所以后面地址传递时要用.c_str()，强制全转为字符类型
		cout << inPath << endl;
		waitKey(200);

		//创建查找文件的句柄,win10句柄类型要用long long 或者是intptr_t，不能直接用long
		intptr_t handle;
		struct _finddata_t fileinfo;
		//第一次查找，根据handle的值判断文件夹地址有没有传成功
		handle = _findfirst(inPath.c_str(), &fileinfo);
		if (handle == -1)
		{
			cout << "文件夹地址没有传成功" << endl;
			return -1;
			//cout << "1" << endl;
			//waitKey(200000);
		}
		do
		{ 
			//找到的文件的文件名
			string imgname = "/train/";
			imgname = imgname + temp + "\\" + fileinfo.name;
			cout << imgname << endl;
			
			img_src = imread(imgname, 0);
		
			if (img_src.empty())
			{
				std::cout << "读取文件夹图片有错误 \n" << std::endl;
				return -1;
				cout << "1" << endl;
				waitKey(200000);
			}
			//提取图片的hog特征
			vector<float> vecDescriptors;
			resize(img_src, img_src, Size(20, 28), 0, 0, INTER_NEAREST);
			coumputeHog(img_src, vecDescriptors);//提取Hog特征向量
			//cout << "HOG特征数是：" << vecDescriptors.size() <<endl;
			for (int j = 0; j < img_information; j++)
			{
				tainingData[k][j] = (float)vecDescriptors[j];
			}
			// 设置标签数据
			for (int j = 0; j < classSum; j++)
			{
				if (j == i)
					labels[ k][j] = 1;
				else
					labels[ k][j] = 0;
			}
			//k加1，下次存入下一行
			k++;
			//cout << training_Data << endl;
			//cout << label_Data << endl;
		} 
		while (!_findnext(handle, &fileinfo));
		_findclose(handle);
	}

	//==========================训练部分==============================
	 //训练样本数据及对应标签
	Mat trainingDataMat(imagesSum, img_information, CV_32FC1, tainingData);
	Mat labelsMat(imagesSum, classSum, CV_32FC1, labels);
	
	Ptr<ANN_MLP>model = ANN_MLP::create();
	Mat layerSizes = (Mat_<int>(1, 3) << trainingDataMat.cols,20 ,classSum);
	model->setLayerSizes(layerSizes);
	model->setTrainMethod(ANN_MLP::BACKPROP, 0.001, 0.1);
	model->setActivationFunction(ANN_MLP::SIGMOID_SYM, 1.0, 1.0);
	model->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 10000, 0.0001));

	Ptr<TrainData> trainData = TrainData::create(trainingDataMat, ROW_SAMPLE, labelsMat);
	cout << "开始训练中" << endl;
	model->train(trainData);
	//保存训练结果
	model->save("/moddel/MLPModel_13.xml");
	cout << "训练成功" << endl;
	waitKey(200000);
	return 0;
}
