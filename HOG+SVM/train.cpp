#include <io.h>//��ȡ�ļ���д���ļ�Ҫ�õ�ͷ�ļ�
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


int main()
{
	Mat img_src;//�����н����ݼ�������ͼƬ
	Mat training_Data;//�����ݼ�����ͼƬ��hog����
	Mat label_Data;//�����ݼ�����ͼƬ��hog����
	const int imagesSum = 23387;//���ݼ��ж�����Ƭ	7400 28519
	const int img_information = 756;//һ��ͼƬ���ж�����Ϣ
	const int classSum = 8; //���м���
	//float (*trainingData)[img_information] = new float[imagesSum][img_information]();
	static float tainingData[imagesSum][img_information] = { {0} };//һ��һ��ͼƬ�����ݣ����������ݼ�ͼƬ�����У�һ��ͼƬ������Ϣ������
	//float(*labels)[classSum] = new float[imagesSum][classSum]();
	static float labels[imagesSum][classSum] = { {0} };//һ��һ��ͼƬ�����ݣ����������ݼ�ͼƬ�����У����ٷ��������

	//==========================��ȡͼƬ����ѵ�����ݺͱ�ǩ==============================
	int k = 0;//���������ݼ���д����ʱ����
	for (int i = 0; i < classSum; i++)
	{
		//Ŀ���ļ���·��
		std::string inPath = "/train/";	
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
		if (handle == -1)
		{
			cout << "�ļ��е�ַû�д��ɹ�" << endl;
			return -1;
			//cout << "1" << endl;
			//waitKey(200000);
		}
		do
		{ 
			//�ҵ����ļ����ļ���
			string imgname = "/train/";
			imgname = imgname + temp + "\\" + fileinfo.name;
			cout << imgname << endl;
			
			img_src = imread(imgname, 0);
		
			if (img_src.empty())
			{
				std::cout << "��ȡ�ļ���ͼƬ�д��� \n" << std::endl;
				return -1;
				cout << "1" << endl;
				waitKey(200000);
			}
			//��ȡͼƬ��hog����
			vector<float> vecDescriptors;
			resize(img_src, img_src, Size(20, 28), 0, 0, INTER_NEAREST);
			coumputeHog(img_src, vecDescriptors);//��ȡHog��������
			//cout << "HOG�������ǣ�" << vecDescriptors.size() <<endl;
			for (int j = 0; j < img_information; j++)
			{
				tainingData[k][j] = (float)vecDescriptors[j];
			}
			// ���ñ�ǩ����
			for (int j = 0; j < classSum; j++)
			{
				if (j == i)
					labels[ k][j] = 1;
				else
					labels[ k][j] = 0;
			}
			//k��1���´δ�����һ��
			k++;
			//cout << training_Data << endl;
			//cout << label_Data << endl;
		} 
		while (!_findnext(handle, &fileinfo));
		_findclose(handle);
	}

	//==========================ѵ������==============================
	 //ѵ���������ݼ���Ӧ��ǩ
	Mat trainingDataMat(imagesSum, img_information, CV_32FC1, tainingData);
	Mat labelsMat(imagesSum, classSum, CV_32FC1, labels);
	
	Ptr<ANN_MLP>model = ANN_MLP::create();
	Mat layerSizes = (Mat_<int>(1, 3) << trainingDataMat.cols,20 ,classSum);
	model->setLayerSizes(layerSizes);
	model->setTrainMethod(ANN_MLP::BACKPROP, 0.001, 0.1);
	model->setActivationFunction(ANN_MLP::SIGMOID_SYM, 1.0, 1.0);
	model->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 10000, 0.0001));

	Ptr<TrainData> trainData = TrainData::create(trainingDataMat, ROW_SAMPLE, labelsMat);
	cout << "��ʼѵ����" << endl;
	model->train(trainData);
	//����ѵ�����
	model->save("/moddel/MLPModel_13.xml");
	cout << "ѵ���ɹ�" << endl;
	waitKey(200000);
	return 0;
}
