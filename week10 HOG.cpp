#include<iostream>
#include<opencv2/opencv.hpp>
#include<math.h>
using namespace cv;
using namespace std;


int calcHOG(Mat src, float* hist, int nAngle, int blockSize) {			//����calcHOG��������������ͼ��ķ����ݶ��ͷ�ͼ��HOG��
	int src_nx = src.cols / blockSize;		//�õ�x�����ж��ٸ�cell
	int src_ny = src.rows / blockSize;		//�õ�y�����ж��ٸ�cell	

	Mat src_gx, src_gy, src_angle, src_mag;
	Sobel(src, src_gx, CV_32F, 1, 0, 1);		//��Sobel���õ�Mat��gx��x�����ݶȣ���gy��y�����ݶȣ�
	Sobel(src, src_gy, CV_32F, 0, 1, 1);


	cartToPolar(src_gx, src_gy, src_mag, src_angle, true);	//�õ�mag���ݶ�ǿ�ȣ�,angle���ݶȽǶȷ���


	for (int i = 0; i < src_ny; i++) {				//ͼ��cell����
		for (int j = 0; j < src_nx; j++) {
			Rect roi;
			Mat roiMat;		
			Mat roiMag;
			Mat roiAngle;
			roi.x = j * blockSize;	//����roi��Ա����
			roi.y = i * blockSize;
			roi.height = blockSize;
			roi.width = blockSize;
			//��cell����
			roiMat = src(roi);		//�õ�ԭͼ�и�cell����ֵ��roiMat
			roiMag = src_mag(roi);	//�õ���cell���ݶȴ�С����ֵ��roiMag
			roiAngle = src_angle(roi);	//�õ���cell���ݶȷ��򣬸�ֵ��roiAngle


			int head = (i * src_nx + j) * nAngle;	//��cell���ܵ�HOG�Ŀ�ͷλ��

			for (int n = 0; n < roiMat.rows; n++) {			//������cell
				for (int m = 0; m < roiMat.cols; m++) {
					int an = int(roiAngle.at<float>(n, m) / (360 / nAngle));	//anΪ��cell�е�ǰ�����ݶȽǶ�λ����һ��bin
					hist[head + an] += roiMag.at<float>(n, m);	//�ڵ�ǰbin�ۼӸ����ص��ݶȴ�С
				}
			}



		}
	}
	return 0;


}


int distance(float* src, float* det) {			//����distance���������ڼ�������HOG��ľ���
	float sum = 0;
	for (int i = 0; i < sizeof(src); i++) {			//��������
		sum = +(src[i] - det[i]) * (src[i] - det[i]);	//ŷ����þ���
	}
	float dis;
	dis = sqrt(sum);		//��������
	return int(dis);		//����int�ͽ��
}

int main() {			//������
	Mat ref = imread("img3.jpg",0);		//��ȡ�ο�ͼ�����Ŵ��Ƚ�ͼ��ĻҶ�ͼ
	Mat det1 = imread("img1.jpg",0);
	Mat det2 = imread("img2.jpg",0);

	int nAngle = 8;				//���ýǶȻ�������
	int blockSize = 16;			//����cell��С

	int nx = ref.cols / blockSize;		//nxΪ�ο�ͼ��x�����ϵ�cell����
	int ny = ref.rows / blockSize;		//nyΪ�ο�ͼ��y�����ϵ�cell����
	
	
	int bins=nx*ny*nAngle;				//binsΪ�ο�ͼ��HOG���ܵ�bin��
	float * ref_hist = new float[bins];			//��������ͼ����Ӧ�Ķ�̬���飬���ڴ�����Ӧͼ���HOG�����鳤��Ϊbins
	memset(ref_hist, 0, sizeof(float) * bins);
	float * det1_hist = new float[bins];
	memset(det1_hist, 0, sizeof(float) * bins);
	float * det2_hist = new float[bins];
	memset(det2_hist, 0, sizeof(float) * bins);

	
	int reCode;
	
	reCode = calcHOG(ref, ref_hist, nAngle,blockSize);		//�ֱ��������ͼ���HOG
	reCode = calcHOG(det1, det1_hist, nAngle, blockSize);
	reCode = calcHOG(det2, det2_hist, nAngle, blockSize);

	if (reCode != 0) {			//�����㲻�ɹ����򷵻�-1�����ͷ��ڴ�
		delete[] ref_hist;
		delete[] det1_hist;
		delete[] det2_hist;
		return -1;
	}

	int dis_1 = distance(ref_hist,det1_hist);		//�ֱ�����������Ƚ�ͼ����ο�ͼ���HOG�ľ���
	int dis_2 = distance(ref_hist,det2_hist);

	cout << "ͼ��1��ο�ͼ��HOG�ľ���" << endl  <<dis_1 << endl;		//�ֱ������������
	cout << "ͼ��2��ο�ͼ��HOG�ľ���" << endl << dis_2 << endl;
	(dis_1 < dis_2) ? (cout << "ͼ��det1�ȽϽӽ�") : (cout << "ͼ��det2�ȽϽӽ�");	//��������С��ͼ�񣬼���ͼ����ο�ͼ�������


	delete[] ref_hist;		//�ͷ��ڴ�
	delete[] det1_hist;
	delete[] det2_hist;


	waitKey(0);
}



