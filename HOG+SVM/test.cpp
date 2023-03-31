#include <io.h>
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

int test_1()
{
	//读取测试图像
	Mat test, dst;
	test = imread("/test/0.jpg", 0);	////C:\Users\zsy\Desktop\新建文件夹\Datas\灰度图\new\6
	//C:\\Users\\zsy\\Desktop\\新建文件夹\\058_MPL测试\\历史数据集\\0705\\1\\(3389).jpg
	if (test.empty())
	{
		std::cout << "can not load image \n" << std::endl;
		return -1;
	}
	//提取图片的hog特征
	vector<float> vecDescriptors;
	resize(test, test, Size(20, 28), (0, 0), (0, 0), INTER_AREA);
	coumputeHog(test, vecDescriptors);//提取Hog特征向量

	//threshold(test, test, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	Mat_<float> testMat = Mat(vecDescriptors).t();//把向量进行转秩
	//cout << testMat << endl;
	//cout << Mat(vecDescriptors) << endl;
	//cout << testMat.cols << endl;
	/*for (int i = 0; i < 20*28; i++)
	{
		testMat.at<float>(0, i) = (float)test.at<uchar>(i / 20, i % 20);
	}*/
	//使用训练好的MLP model预测测试图像
	string Model_path = "/model/MLPModel_12.xml";
	Ptr<ml::ANN_MLP> model = ml::ANN_MLP::create();
	model = cv::Algorithm::load<cv::ml::ANN_MLP>(Model_path);
	cout << testMat.type() << endl;
	cout << testMat.cols << endl;

	model->predict(testMat, dst);
	//std::cout << "testMat: \n" << testMat << "\n" << std::endl;
	std::cout << "dst: \n" << dst << "\n" << std::endl;
	double maxVal = 0;
	Point maxLoc;
	minMaxLoc(dst, NULL, &maxVal, NULL, &maxLoc);
	if (maxVal > 0.9) {
		std::cout << "测试结果：" << maxLoc.x << "置信度:" << maxVal * 100 << "%" << std::endl;
	}
	namedWindow("test", WINDOW_FREERATIO);
	imshow("test", test);
	waitKey(20000000);
	return 0;
}

int test_2()
{
	//使用训练好的MLP model预测测试图像
	string Model_path = "/model/MLPModel_12.xml";
	Ptr<ml::ANN_MLP> model = ml::ANN_MLP::create();
	model = cv::Algorithm::load<cv::ml::ANN_MLP>(Model_path);

	Mat test, dst;
	Mat img_src;//用来承接数据集样本的图片
	const int classSum = 8; //共有几类
	int class_img[classSum + 1] = { 0 };	//图片总数
	int true_img[classSum + 1] = { 0 };	//识别正确的图片数
	float accuracy[classSum + 1] = { 0 };	//准确度

	for (int i = 0; i < classSum; i++)
	{
		//目标文件夹路径
		std::string inPath = "/test/";	
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
		cout << fileinfo.size << endl;
		waitKey(200000);
		if (handle == -1)
		{
			cout << "文件夹地址没有传成功" << endl;
			//waitKey(200000);
			//return -1;

		}
		do
		{
			//找到的文件的文件名
			string imgname = "/test/";
			imgname = imgname + temp + "\\" + fileinfo.name;
			cout << imgname << endl;

			img_src = imread(imgname, 0);

			if (img_src.empty())
			{
				std::cout << "读取文件夹图片有错误 \n" << std::endl;
				//waitKey(200000);
				//return -1;
			}

			//提取图片的hog特征
			vector<float> vecDescriptors;
			resize(img_src, img_src, Size(20, 28), (0, 0), (0, 0), INTER_AREA);
			coumputeHog(img_src, vecDescriptors);//提取Hog特征向量

			//threshold(test, test, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
			Mat_<float> testMat = Mat(vecDescriptors).t();//把向量进行转秩

			model->predict(testMat, dst);
			//std::cout << "testMat: \n" << testMat << "\n" << std::endl;
			cout << "dst: \n" << dst << "\n" << endl;
			double maxVal = 0;
			Point maxLoc;
			minMaxLoc(dst, NULL, &maxVal, NULL, &maxLoc);
			if (maxVal > 0.8) {
				std::cout << "测试结果：" << maxLoc.x << "置信度:" << maxVal * 100 << "%" << std::endl;
				if (maxLoc.x == i)
				{
					true_img[i] = true_img[i] + 1;
					true_img[classSum] = true_img[classSum] + 1;
				}
			}
			class_img[i] = class_img[i] + 1;
			class_img[classSum] = class_img[classSum] + 1;
			//namedWindow("test", WINDOW_FREERATIO);
			//imshow("test", img_src);
			//waitKey(200000);
		} 
		while (!_findnext(handle, &fileinfo));
		_findclose(handle);
		accuracy[i] = true_img[i] * 100 / class_img[i];
	}
	accuracy[classSum] = true_img[classSum] * 100 / class_img[classSum];
	cout << "总测试数：" << class_img[classSum] << endl;
	cout << "总识别正确数：" << true_img[classSum] << endl;
	cout << "总识别正确率：" << true_img[classSum]*100 / class_img[classSum] << "%" << endl;
	for (int i = 0; i < classSum; i++)
	{
		cout << "数字" << i << "\t测试数:" << class_img[i] << "\t识别正确数：" << true_img[i] << "\t识别正确率：" << accuracy[i] << "%"  << endl;
	}
	waitKey(20000000);
	return 0;
}


int main()
{	
	//单图像测试
	test_1();

	//测试集
	//test_2();

	return 0;

}
