#include <io.h>
#include <string>
#include <iostream>
#include <opencv2\opencv.hpp>
#include <opencv2\ml.hpp>
using namespace cv;
using namespace ml;
using namespace std;

//����ͼ���ݶ�
void coumputeHog(const Mat& src, vector<float>& descriptors)
{

	Size win_size = Size(20, 28);// ��ⴰ�ڴ�С��
	Size block_size = Size(16, 16);// ���С��Ŀǰֻ֧��Size(16, 16)
	Size block_stride = Size(2, 2);// ��Ļ�����������Сֻ֧���ǵ�Ԫ��cell_size��С�ı�����
	Size cell_size = Size(8, 8);// ��Ԫ��Ĵ�С��Ŀǰֻ֧��Size(8, 8)��
	int nbins = 9;// ֱ��ͼbin������(ͶƱ��ĸ���)��Ŀǰÿ����Ԫ��Cellֻ֧��9����
	//double win_sigma = DEFAULT_WIN_SIGMA;// ��˹�˲����ڵĲ�����
	double threshold_L2hys = 0.2;// ����ֱ��ͼ��һ������L2 - Hys�Ĺ�һ��������
	bool gamma_correction = true;// �Ƿ�gammaУ��
	//nlevels = DEFAULT_NLEVELS; ��ⴰ�ڵ��������
	HOGDescriptor myHog = HOGDescriptor(win_size, block_size, block_stride, cell_size, 9);
	myHog.compute(src, descriptors);
}

int test_1()
{
	//��ȡ����ͼ��
	Mat test, dst;
	test = imread("/test/0.jpg", 0);	////C:\Users\zsy\Desktop\�½��ļ���\Datas\�Ҷ�ͼ\new\6
	//C:\\Users\\zsy\\Desktop\\�½��ļ���\\058_MPL����\\��ʷ���ݼ�\\0705\\1\\(3389).jpg
	if (test.empty())
	{
		std::cout << "can not load image \n" << std::endl;
		return -1;
	}
	//��ȡͼƬ��hog����
	vector<float> vecDescriptors;
	resize(test, test, Size(20, 28), (0, 0), (0, 0), INTER_AREA);
	coumputeHog(test, vecDescriptors);//��ȡHog��������

	//threshold(test, test, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	Mat_<float> testMat = Mat(vecDescriptors).t();//����������ת��
	//cout << testMat << endl;
	//cout << Mat(vecDescriptors) << endl;
	//cout << testMat.cols << endl;
	/*for (int i = 0; i < 20*28; i++)
	{
		testMat.at<float>(0, i) = (float)test.at<uchar>(i / 20, i % 20);
	}*/
	//ʹ��ѵ���õ�MLP modelԤ�����ͼ��
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
		std::cout << "���Խ����" << maxLoc.x << "���Ŷ�:" << maxVal * 100 << "%" << std::endl;
	}
	namedWindow("test", WINDOW_FREERATIO);
	imshow("test", test);
	waitKey(20000000);
	return 0;
}

int test_2()
{
	//ʹ��ѵ���õ�MLP modelԤ�����ͼ��
	string Model_path = "/model/MLPModel_12.xml";
	Ptr<ml::ANN_MLP> model = ml::ANN_MLP::create();
	model = cv::Algorithm::load<cv::ml::ANN_MLP>(Model_path);

	Mat test, dst;
	Mat img_src;//�����н����ݼ�������ͼƬ
	const int classSum = 8; //���м���
	int class_img[classSum + 1] = { 0 };	//ͼƬ����
	int true_img[classSum + 1] = { 0 };	//ʶ����ȷ��ͼƬ��
	float accuracy[classSum + 1] = { 0 };	//׼ȷ��

	for (int i = 0; i < classSum; i++)
	{
		//Ŀ���ļ���·��
		std::string inPath = "/test/";	
		char temp[256];
		sprintf_s(temp, "%d", i);	//��Ϊ�ļ����Ǵ�1��ʼ�ģ�����i��������0
		inPath = inPath + temp + "\\*.jpg";	//string+char,���Ժ����ַ����ʱҪ��.c_str()��ǿ��ȫתΪ�ַ�����
		cout << inPath << endl;
		waitKey(200);

		//���������ļ��ľ��,win10�������Ҫ��long long ������intptr_t������ֱ����long
		intptr_t handle;
		struct _finddata_t fileinfo;
		//��һ�β��ң�����handle��ֵ�ж��ļ��е�ַ��û�д��ɹ�
		handle = _findfirst(inPath.c_str(), &fileinfo);
		cout << fileinfo.size << endl;
		waitKey(200000);
		if (handle == -1)
		{
			cout << "�ļ��е�ַû�д��ɹ�" << endl;
			//waitKey(200000);
			//return -1;

		}
		do
		{
			//�ҵ����ļ����ļ���
			string imgname = "/test/";
			imgname = imgname + temp + "\\" + fileinfo.name;
			cout << imgname << endl;

			img_src = imread(imgname, 0);

			if (img_src.empty())
			{
				std::cout << "��ȡ�ļ���ͼƬ�д��� \n" << std::endl;
				//waitKey(200000);
				//return -1;
			}

			//��ȡͼƬ��hog����
			vector<float> vecDescriptors;
			resize(img_src, img_src, Size(20, 28), (0, 0), (0, 0), INTER_AREA);
			coumputeHog(img_src, vecDescriptors);//��ȡHog��������

			//threshold(test, test, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
			Mat_<float> testMat = Mat(vecDescriptors).t();//����������ת��

			model->predict(testMat, dst);
			//std::cout << "testMat: \n" << testMat << "\n" << std::endl;
			cout << "dst: \n" << dst << "\n" << endl;
			double maxVal = 0;
			Point maxLoc;
			minMaxLoc(dst, NULL, &maxVal, NULL, &maxLoc);
			if (maxVal > 0.8) {
				std::cout << "���Խ����" << maxLoc.x << "���Ŷ�:" << maxVal * 100 << "%" << std::endl;
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
	cout << "�ܲ�������" << class_img[classSum] << endl;
	cout << "��ʶ����ȷ����" << true_img[classSum] << endl;
	cout << "��ʶ����ȷ�ʣ�" << true_img[classSum]*100 / class_img[classSum] << "%" << endl;
	for (int i = 0; i < classSum; i++)
	{
		cout << "����" << i << "\t������:" << class_img[i] << "\tʶ����ȷ����" << true_img[i] << "\tʶ����ȷ�ʣ�" << accuracy[i] << "%"  << endl;
	}
	waitKey(20000000);
	return 0;
}


int main()
{	
	//��ͼ�����
	test_1();

	//���Լ�
	//test_2();

	return 0;

}
