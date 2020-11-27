#include<iostream>
#include<opencv2/opencv.hpp>
#include<math.h>
using namespace cv;
using namespace std;


int calcHOG(Mat src, float* hist, int nAngle, int blockSize) {			//定义calcHOG函数，用于生成图像的方向梯度释放图（HOG）
	int src_nx = src.cols / blockSize;		//得到x轴向有多少个cell
	int src_ny = src.rows / blockSize;		//得到y轴向有多少个cell	

	Mat src_gx, src_gy, src_angle, src_mag;
	Sobel(src, src_gx, CV_32F, 1, 0, 1);		//用Sobel，得到Mat类gx（x方向梯度），gy（y方向梯度）
	Sobel(src, src_gy, CV_32F, 0, 1, 1);


	cartToPolar(src_gx, src_gy, src_mag, src_angle, true);	//得到mag（梯度强度）,angle（梯度角度方向）


	for (int i = 0; i < src_ny; i++) {				//图像按cell遍历
		for (int j = 0; j < src_nx; j++) {
			Rect roi;
			Mat roiMat;		
			Mat roiMag;
			Mat roiAngle;
			roi.x = j * blockSize;	//定义roi成员变量
			roi.y = i * blockSize;
			roi.height = blockSize;
			roi.width = blockSize;
			//按cell处理
			roiMat = src(roi);		//得到原图中该cell，赋值给roiMat
			roiMag = src_mag(roi);	//得到该cell中梯度大小，赋值给roiMag
			roiAngle = src_angle(roi);	//得到该cell中梯度方向，赋值给roiAngle


			int head = (i * src_nx + j) * nAngle;	//该cell在总的HOG的开头位置

			for (int n = 0; n < roiMat.rows; n++) {			//遍历该cell
				for (int m = 0; m < roiMat.cols; m++) {
					int an = int(roiAngle.at<float>(n, m) / (360 / nAngle));	//an为该cell中当前像素梯度角度位于哪一个bin
					hist[head + an] += roiMag.at<float>(n, m);	//在当前bin累加该像素的梯度大小
				}
			}



		}
	}
	return 0;


}


int distance(float* src, float* det) {			//定义distance函数，用于计算两个HOG间的距离
	float sum = 0;
	for (int i = 0; i < sizeof(src); i++) {			//遍历数组
		sum = +(src[i] - det[i]) * (src[i] - det[i]);	//欧几里得距离
	}
	float dis;
	dis = sqrt(sum);		//开放运算
	return int(dis);		//返回int型结果
}

int main() {			//主函数
	Mat ref = imread("img3.jpg",0);		//读取参考图和两张待比较图像的灰度图
	Mat det1 = imread("img1.jpg",0);
	Mat det2 = imread("img2.jpg",0);

	int nAngle = 8;				//设置角度划分数量
	int blockSize = 16;			//设置cell大小

	int nx = ref.cols / blockSize;		//nx为参考图像x方向上的cell个数
	int ny = ref.rows / blockSize;		//ny为参考图像y方向上的cell个数
	
	
	int bins=nx*ny*nAngle;				//bins为参考图像HOG的总的bin数
	float * ref_hist = new float[bins];			//建立三个图像相应的动态数组，用于储存相应图像的HOG，数组长度为bins
	memset(ref_hist, 0, sizeof(float) * bins);
	float * det1_hist = new float[bins];
	memset(det1_hist, 0, sizeof(float) * bins);
	float * det2_hist = new float[bins];
	memset(det2_hist, 0, sizeof(float) * bins);

	
	int reCode;
	
	reCode = calcHOG(ref, ref_hist, nAngle,blockSize);		//分别计算三幅图像的HOG
	reCode = calcHOG(det1, det1_hist, nAngle, blockSize);
	reCode = calcHOG(det2, det2_hist, nAngle, blockSize);

	if (reCode != 0) {			//若计算不成功，则返回-1，且释放内存
		delete[] ref_hist;
		delete[] det1_hist;
		delete[] det2_hist;
		return -1;
	}

	int dis_1 = distance(ref_hist,det1_hist);		//分别计算两幅待比较图像与参考图像的HOG的距离
	int dis_2 = distance(ref_hist,det2_hist);

	cout << "图像1与参考图像HOG的距离" << endl  <<dis_1 << endl;		//分别输出两个距离
	cout << "图像2与参考图像HOG的距离" << endl << dis_2 << endl;
	(dis_1 < dis_2) ? (cout << "图像det1比较接近") : (cout << "图像det2比较接近");	//输出距离较小的图像，即该图像与参考图像更相似


	delete[] ref_hist;		//释放内存
	delete[] det1_hist;
	delete[] det2_hist;


	waitKey(0);
}



