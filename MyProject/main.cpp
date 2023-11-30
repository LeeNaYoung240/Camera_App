#include <opencv2/opencv.hpp> //헤더파일 포함(opencv에서 지원하는 모든 기능 포함)
#include <iostream> //헤더파일 포함
#include <fstream> //파일 기록,파일에 저장된 데이터를 읽기 위한 헤더파일
#include<opencv2/photo.hpp>  //사진 처리 및 복원과 관련된 특수 알고리즘 포함
#include<opencv2/highgui.hpp> //윈도우 화면, UI처리 및 마우스 제어 가능
#include <stdlib.h> //난수 생성, 문자열 형식을 다른 형식으로 변환하는 헤더파일
#include <stdio.h> //매크로 정의, 상수, 여러 형의 입출력 함수 포함된 헤더파일
#include "FaceDetectorAndTracker.h" //얼굴 검출 헤더파일
#include "FaceSwapper.h" //얼굴 바꾸기 헤더파일
#include <opencv2/imgcodecs.hpp> //기본 이미지 코덱(압축하거나 푸는 것) 포함
#include <opencv2/core/core.hpp> //Mat class를 포함한 기본 C++ 구조체와 산술 루틴을 포함
#include <opencv2/video/video.hpp> //비디오 추척 및 배경 segmentation과 관련된 루틴 포함
#include <opencv2/imgproc/imgproc.hpp> //Image processing을 위한 기능 포함
#include <opencv2/objdetect/objdetect.hpp> //객체 detection을 위한 기능을 포함
#include <time.h> //시간 관련 함수를 모아놓은 라이브러리 
using namespace std; //이름공간 사용
using namespace cv; //이름공간 사용
using namespace cv::ml; //이름공간 사용
using namespace std; //이름공간 사용
using namespace cv::dnn; //이름공간 사용
void draw_mouse(int event, int x, int y, int flags, void* userdata); //마우스 이벤트
void detectAndDraw(Mat& img, CascadeClassifier& cascade, CascadeClassifier& nestedCascade, double scale, bool tryflip, Mat glasses); //안경 검출
void overlayImage(const Mat& background, const Mat& foreground, Mat& output, Point2i location); //이미지 오버레이
void mosaic(); //모자이크 기능 함수
void cartoon();//만화 질감 처리 기능함수
void sticker(int argc, const char** argv); //스티커 기능 함수
void face_swap(); //얼굴 바꾸기 함수
void end_x(); //종료 함수
VideoCapture cap(0); //0번째 카메라를 사용하기 위한 생성자
double ms, fpsLive; //전역 변수 선언
Mat result, frame; //객체 선언
int main() //메인 함수
{
	vector<String> classNames = { "X","M","C","D","F"}; //클래스 네임

	Mat img = Mat::zeros(400, 400, CV_8UC3); //이미지 객체 생성
	Net net = readNet("frozen_model.pb"); //Net객체 생성

	imshow("img", img); //영상 출력
	setMouseCallback("img", draw_mouse, (void*)&img); //마우스 이벤트 함수

	Point maxLoc; //포인트 선언 
	double maxVal; //변수 선언

	while (true) { //반복문
		int c = waitKey(); //키 입력이 들어올 때까지 대기
		if (c == ' ')  //스페이스 바의 입력이 있을 경우
		 {
			Mat inputBlob = blobFromImage(img, 1/127.5, Size(224, 224),-1.0); //Size(224,244)크기의 블롭 생성
			net.setInput(inputBlob); //inputBlob을 이용하여 네크워크 입력을 설정
			Mat prob = net.forward(); //네트워크를 순방향으로 실행함(추론)
	
			minMaxLoc(prob, NULL, &maxVal, NULL, &maxLoc); //최대값과 그 좌표를 저장
			String str = classNames[maxLoc.x] + format("(%4.2lf%%)", maxVal * 100); //인식률 
			cout << "인식결과 : " << str  << endl; //인식률 값 출력
			
			if (classNames[maxLoc.x] == "M") //마우스 이벤트 M을 그릴 경우
			{
				cout << "모자이크 기능을 수행합니다." << endl; //모자이크 기능을 수행한다는 결과값 출력
				mosaic(); //모자이크 기능 함수 호출
			}
			else if (classNames[maxLoc.x] == "F") //마우스 이벤트 F을 그릴 경우
			{
				cout << "얼굴 바꾸기 기능을 수행합니다." << endl; //얼굴 바꾸기 기능을 수행한다는 결과값 출력
				face_swap(); //얼굴 바꾸기 기능 함수 호출
			}
			else if (classNames[maxLoc.x] == "C") //마우스 이벤트 C를 그릴 경우
			{
				cout << "만화 질감 표현 기능을 수행합니다." << endl; //만화 질감 표현 기능을 수행한다는 결과값 출력
				cartoon();	//만화 질감 처리 기능 함수 호출
			}
			else if (classNames[maxLoc.x] == "D")//마우스 이벤트 D를 그릴 경우
			{
				cout << "스티커 안경 기능을 수행합니다." << endl; //스티커 기능을 수행한다는 결과값 출력
				sticker(0, 0); //스티커 기능 함수 호출
			}
			else if (classNames[maxLoc.x] == "X") //마우스 이벤트 x를 그릴 경우
			{
				end_x(); //종료 기능 함수 호출
			}
		}
		img.setTo(0); //이미지 복사
	}
	return 0; //프로그램 종료
}
void end_x() //종료 기능 함수 정의
{
	cout << "종료합니다." << endl; //종료합니다라는 결과값 출력
	destroyWindow("img"); //img를 파괴
}
Point ptPrev(-1, -1); //포인트 변수
void draw_mouse(int event, int x, int y, int flags, void* userdata) //마우스 콜백 함수
{
	Mat img = *(Mat*)userdata; 
	if (event == EVENT_LBUTTONDOWN) //마우스 왼쪽 버튼을 누른 위치를 
		ptPrev = Point(x, y); //ptprev에 저장
	else if (event == EVENT_LBUTTONUP) //마우스 왼쪽 버튼을 떼면
		ptPrev = Point(-1, -1); //ptprev 좌표를 (-1,-1)로 초기화 
	else if (event == EVENT_MOUSEMOVE && (flags &EVENT_FLAG_LBUTTON))//마우스 왼쪽 버튼을 누른 상태로 마우스가 움직이면
	{
		line(img, ptPrev, Point(x, y), Scalar::all(255), 40, LINE_AA, 0); //선을 그림
		ptPrev = Point(x, y); //ptPrev좌표를 (x,y)로 변경
		imshow("img", img);//이미지 출력
	}
}
void mosaic() //모자이크 기능 함수 정의
{
	Mat img; //img 객체 생성
	VideoCapture cap(0); //0번째 카메라를 사용하기 위한 생성자
	CascadeClassifier face_c; //캐스케이드 분류기(얼굴 검출)
	face_c.load("C:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt.xml"); //사전에 학습된 데이터 불러오기
    frame; //프레임을 받을 Mat객체 생성
	double fps = cap.get(CAP_PROP_FPS); //초당 프레임 수
	int num_frames = 1; //캡쳐할 프레임 수
	clock_t start; //시작 시간 변수
    clock_t end; //끝나는 시간 변수
	while (1) { //오픈에 성공한 경우 sendCommand()를 통해 계속적으로 데이터를 전송한다. 전송에 실패 할 경우 failed 메시지를 출력한다.
		start = clock();//시작 시간
		bool frame_valid = true; //유효한 프레임일 경우 실행
		cap >> frame; //카메라로부터 frame에 프레임을 받음
		try { //예외가 발생할 가능성이 있는 경우
			cap >> frame; //웹캠에서 새 프레임 가져오기
		}
		catch (Exception& e) { //예외가 발생했을 경우
			cout << "예외 발생 " << e.err << endl; //출력 메세지 출력
			frame_valid = false; //false
		}
		if (frame_valid) { //유효한 프레임일 경우
			try { //예외가 발생할 가능성이 있을 경우
				Mat grayframe; //그레이 영상을 만들기 위한 객체 생성
				cvtColor(frame, grayframe, COLOR_BGR2GRAY); //frame영상을 grayframe으로 그레이 영상으로 변환(평활화가 그레이 영상만 받음)
				equalizeHist(grayframe, grayframe); //히스토그램 평활화를 수행(픽셀 분포가 너무 많이 뭉쳐있는 경우 이를 넓게 펼쳐줌)
				
				vector<Rect> faces; //얼굴 위치 저장(검출된 객체의 사각형 좌표 정보)
				face_c.detectMultiScale(grayframe, faces, 1.1, 3, 0, Size(30, 30)); //다양한 크기의 객체 사각형 영역 검출
				Mat mosaic; //모자이크를 수행할 결과를 저장할 객체 생성
				Mat original; //원본 결과를 출력할 객체 생성
				frame.copyTo(original); //이미지 복사(frame->original)
				for (int i = 0; i < faces.size(); i++) {
					Point X(faces[i].x + faces[i].width, faces[i].y + faces[i].height); //w,h(사각형을 그릴 점 만들기)
					Point Y(faces[i].x, faces[i].y); //x,y(사각형을 그릴 점 만들기)
					mosaic = frame(Rect(X, Y));//모자이크에 프레임 좌표 저장
					Mat img_temp; //이미지를 바꿀 객체 생성
					resize(mosaic, img_temp, Size(mosaic.rows / 8, mosaic.cols / 8)); //사이즈를 64배 작게 축소
					resize(img_temp, mosaic, Size(mosaic.rows, mosaic.cols)); //사이즈를 크게 키워 모자이크로 보이게 
			
					end = clock(); //끝나는 시간

					double seconds = (double(end) - double(start)) / double(CLOCKS_PER_SEC); //경과시간 구하기(초)
					cout << "걸린 시간 : " << seconds << " seconds" << endl; //경과시간 결과값 출력
					fpsLive = double(num_frames) / double(seconds); //실시간 fps 구하기
					cout << "fps : " << fpsLive << endl; //실시간 fps 결과값 출력

					putText(frame, "FPS: " + to_string(int(fpsLive)), { 50, 50 }, FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2); //fps값 영상에 문자로 표시
				}
				imshow("mosaic", frame); //모자이크 영상
			}
			catch (Exception& e) { //예외가 발생했을 경우
				cout << "예외 발생 " << e.err << endl; //출력 메세지 출력
			}
		}
		if (waitKey(30) == 27) //키보드 esc키를 눌렀을 경우
		{
			destroyWindow("mosaic"); //mosaic 영상 파괴
			break; //탈출
		}
	}
}
void cartoon() //만화 질감 처리 기능 함수 정의
{
	Mat img; //img 객체 생성
	VideoCapture cap(0); //0번째 카메라를 사용하기 위한 생성자
	CascadeClassifier face_c; //캐스케이드 분류기(얼굴 검출)
	face_c.load("C:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt.xml"); //사전에 학습된 데이터 불러오기
	Mat frame; //프레임을 받을 Mat객체 생성
	double fps = cap.get(CAP_PROP_FPS); //초당 프레임 수
	int num_frames = 1; //캡쳐할 프레임 수
	clock_t start; //시작 시간 변수
	clock_t end; //끝나는 시간 변수
	while (1) { //오픈에 성공한 경우 sendCommand()를 통해 계속적으로 데이터를 전송한다. 전송에 실패 할 경우 failed 메시지를 출력한다.
		start = clock(); //시작 시간
		bool frame_valid = true; //유효한 프레임일 경우 실행
		cap >> frame; //카메라로부터 frame에 프레임을 받음
		try { //예외가 발생할 가능성이 있는 경우
			cap >> frame; //웹캠에서 새 프레임 가져오기
		}
		catch (Exception& e) { //예외가 발생했을 경우
			cout << "예외 발생 " << e.err << endl; //출력 메세지 출력
			frame_valid = false; //false
		}
		if (frame_valid) { //유효한 프레임일 경우
			try { //예외가 발생할 가능성이 있을 경우
				Mat grayframe; //그레이 영상을 만들기 위한 객체 생성
				cvtColor(frame, grayframe, COLOR_BGR2GRAY); //frame영상을 grayframe으로 그레이 영상으로 변환(평활화가 그레이 영상만 받음)
			    equalizeHist(grayframe, grayframe); //히스토그램 평활화를 수행(픽셀 분포가 너무 많이 뭉쳐있는 경우 이를 넓게 펼쳐줌)
				
				vector<Rect> faces; //얼굴 위치 저장(검출된 객체의 사각형 좌표 정보)
				face_c.detectMultiScale(grayframe, faces, 1.1, 3, 0, Size(30, 30)); //객체 사각형 영역 검출			 
				Mat cartoon; //모자이크를 수행할 결과를 저장할 객체 생성
				Mat original; //원본 결과를 출력할 객체 생성
				frame.copyTo(original); //이미지 복사(frame->original)
				for (int i = 0; i < faces.size(); i++) {
					Point X(faces[i].x + faces[i].width, faces[i].y + faces[i].height); //w,h(사각형을 그릴 점 만들기)
					Point Y(faces[i].x, faces[i].y); //x,y(사각형을 그릴 점 만들기)
					GaussianBlur(grayframe, grayframe, Size(3, 3), 0); //가우시안 블러 적용
					//에지 검출
					Mat edgeImage; //에지(픽셀값이 급격히 변하는 지점)를 찾을 객체 생성
					Laplacian(grayframe, edgeImage, -1, 5); //그레이 영상에 라플라시안 필터 적용
					convertScaleAbs(edgeImage, edgeImage); //값을 절대값화 시키고, 정수화 시킴
					//이미지 반전
					edgeImage = 255 - edgeImage; //이미지를 반전 시킴
					//이진화 적용
					threshold(edgeImage, edgeImage, 150, 255, THRESH_BINARY); //반전시킨 이미지에 이진화 적용
					//가장자리 보존 필터를 사용하여 이미지를 크고 흐리게 처리
					Mat edgePreservingImage; //객체 생성
					edgePreservingFilter(frame, edgePreservingImage, 2, 50, 0.4); //프레임 값에 가장자리 보존 필터 적용
					// 출력 행렬 만들기
					Mat output; //output 객체 생성
					output = Scalar::all(0); //모든 원소 0으로 초기화
					// Combine the cartoon and edges
					bitwise_and(edgePreservingImage, edgePreservingImage, output, edgeImage); //이미지 합성
					Mat img2 = frame(Rect(X, Y)); //프레임의 얼굴영역 부분의 객체 저장 (얕은 복사)
					Mat img3 = frame(Rect(X, Y)); //프레임의 얼굴영역 부분의 객체 저장 (얕은 복사)
					img3.copyTo(img2); //이미지 복사
					img2 = Scalar(0, 0, 0); //이미지 색 변경 - black
					bitwise_or(frame, output, frame); // 두 이미지를 합침 
					bitwise_and(frame, original, frame); //공통으로 겹치는 부분을 추출하기 위해 			
					
					end = clock(); //끝나는 시간
		            double seconds = (double(end) - double(start)) / double(CLOCKS_PER_SEC); //경과시간 구하기(초)
					cout << "걸린 시간 : " << seconds << " seconds" << endl; //경과시간 결과값 출력
					fpsLive = double(num_frames) / double(seconds); //실시간 fps 구하기
					cout << "fps : " << fpsLive << endl; //실시간 fps 결과값 출력
					putText(frame, "FPS: " + to_string(int(fpsLive)), { 50, 50 }, FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2); //fps값 영상에 문자로 표시
				}
				imshow("cartoon", frame); //모아지크 영상	
			}
			catch (Exception& e) { //예외가 발생했을 경우
				cout << "예외 발생 " << e.err << endl; //출력 메세지 출력
			}
		}
		if (waitKey(30) == 27)//키보드 esc키를 눌렀을 경우
		{
			destroyWindow("cartoon"); //cartoon 영상 파괴
			break; //탈출
		}
	}
}
void sticker(int argc, const char** argv) //스티커 기능 함수 정으
{
	Mat frame, image, glasses; //프레임, 이미지 안경 객체 생성
	VideoCapture cap(0); //0번째 카메라를 사용하기 위한 생성자
	string glassesImage = "sunglasses3.png"; //안경 이미지 불러오기
	bool tryflip = false; //유효한 경우일 때 실행
	double scale = 1; //변수 선언
	glasses = imread(glassesImage, IMREAD_UNCHANGED);//안경 읽기
	CascadeClassifier face_c; //캐스케이드 분류기(얼굴 검출)
	CascadeClassifier eyes; //캐스케이드 분류기(눈 검출)
	face_c.load("C:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt.xml"); //사전에 학습된 얼굴 데이터 불러오기
	eyes.load("C:\\opencv\\sources\\data\\haarcascades\\haarcascade_eye_tree_eyeglasses.xml"); //사전에 학습된 눈 인식 데이터 불러오기
	double fps = cap.get(CAP_PROP_FPS); //초당 프레임 수
	int num_frames = 1; //캡쳐할 프레임 수
	clock_t start; //시작 시간 변수
	clock_t end; //끝나는 시간 변수
	while (1) { //오픈에 성공한 경우 sendCommand()를 통해 계속적으로 데이터를 전송한다. 전송에 실패 할 경우 failed 메시지를 출력한다.
		start = clock(); //시작 시간
		bool frame_valid = true; //유효한 프레임일 경우 실행
		cap >> frame; //카메라로부터 frame에 프레임을 받음
		try { //예외가 발생할 가능성이 있는 경우
			cap >> frame; //웹캠에서 새 프레임 가져오기
		}
		catch (Exception& e) { //예외가 발생했을 경우
			cout << "예외 발생 " << e.err << endl; //출력 메세지 출력
			frame_valid = false; //false
		}
		if (frame_valid) { //유효한 프레임일 경우
			try { //예외가 발생할 가능성이 있을 경우
				Mat grayframe; //그레이 영상을 만들기 위한 객체 생성

				cvtColor(frame, grayframe, COLOR_BGR2GRAY); //frame영상을 grayframe으로 그레이 영상으로 변환(평활화가 그레이 영상만 받음)
				equalizeHist(grayframe, grayframe); //히스토그램 평활화를 수행(픽셀 분포가 너무 많이 뭉쳐있는 경우 이를 넓게 펼쳐줌)

				vector<Rect> faces; //얼굴 위치 저장(검출된 객체의 사각형 좌표 정보)
				face_c.detectMultiScale(grayframe, faces, 1.1, 3, 0, Size(30, 30)); //객체 사각형 영역 검출

				for (int i = 0; i < faces.size(); i++) {
					Point X(faces[i].x + faces[i].width, faces[i].y + faces[i].height); //w,h(사각형을 그릴 점 만들기)
					Point Y(faces[i].x, faces[i].y); //x,y(사각형을 그릴 점 만들기
					Mat frame1 = frame.clone(); //이미지 복사
					detectAndDraw(frame1, face_c, eyes, scale, tryflip, glasses); //얼굴 검출 후 안경 그리기
					
					end = clock(); //끝나는 시간
					double seconds = (double(end) - double(start)) / double(CLOCKS_PER_SEC); //경과시간 구하기(초)
					cout << "걸린 시간 : " << seconds << " seconds" << endl; //경과시간 결과값 출력
					fpsLive = double(num_frames) / double(seconds); //실시간 fps 구하기
					cout << "fps : " << fpsLive << endl; //실시간 fps 결과값 출력
					putText(frame, "FPS: " + to_string(int(fpsLive)), { 50, 50 }, FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2); //fps값 영상에 문자로 표시
				}
			}
			catch (Exception& e) { //예외가 발생했을 경우
				cout << "예외 발생 " << e.err << endl; //출력 메세지 출력
			}
		}
		if (waitKey(30) == 27) //키보드 esc키를 눌렀을 경우
		{
			destroyWindow("sticker"); //sticker 영상 파괴
			break;//탈출
		}
	}
}
void detectAndDraw(Mat& img, CascadeClassifier& cascade, CascadeClassifier& CCascade, double scale, bool tryflip, Mat glasses) //얼굴 검출 후 안경 그릴 함수
{

	Mat output2;  //결과를 보여줄 경우를 위해
	img.copyTo(output2); //원본 이미지 복사
	double t = 0; //변수 선언
	vector<Rect> faces, faces2; //얼굴 위치 저장(검출된 객체의 사각형 정보)
	Mat gray, smallImg; //객체 선언
	cvtColor(img, gray, COLOR_BGR2GRAY); //이미지 객체를 그레이 형식으로 변환하고 gray에 저장
	double fx = 1 / scale; //사이즈를 줄이기 위해 
	resize(gray, smallImg, Size(), fx, fx, INTER_LINEAR_EXACT); //x,y축 방향으로 크기 변환하고 사이즈 줄이기
	equalizeHist(smallImg, smallImg); //평활화(픽셀 분포가 많이 뭉쳐있는 경우 이를 넓게 펼쳐줌)
	cascade.detectMultiScale(smallImg, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30)); //얼굴 위치 검출
	for (size_t i = 0; i < faces.size(); i++)
	{
		Rect r = faces[i]; //얼굴 좌표 위치 사각형에 저장
		Mat smallImgROI; //작은 이미지ROI 객체 생성
		vector<Rect> Objects; //검출된 객체의 사각형 좌표 정보
		Point center; //중심 좌표 포인트를 저장할 포인트
		int radius;//변수 선언
		smallImgROI = smallImg(r);//객체에 얼굴 좌표 사각형 저장
		CCascade.detectMultiScale(smallImgROI, Objects, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(20, 20));//얼굴 내부 영역에서 눈위치를 검출
		vector<Point> points; //1차원 벡터 선언
		//눈 위치에 그려줌
		for (size_t j = 0; j < Objects.size(); j++)  //검출된 사각형까지
		{
			Rect nr = Objects[j]; //검출된 사각형 크기까지 사각형에 저장
			center.x = cvRound((r.x + nr.x + nr.width * 0.5) * scale); //가로축의 가운데
			center.y = cvRound((r.y + nr.y + nr.height * 0.5) * scale); //세로축의 가운데
			radius = cvRound((nr.width + nr.height) * 0.25 * scale); //반지름
			Point p(center.x, center.y); //중심 x,y를 p에 저장
			points.push_back(p); //points 벡터 끝에 원소를 추가 (중심 x,y)
		}
		if (points.size() == 2) { //눈 위치가 2개가 검출될 경우(눈 영역이 2개일 경우만 선글라스를 씌워주기 위함)
			Point center1 = points[0]; //첫번째 
			Point center2 = points[1]; //두번째
			if (center1.x > center2.x) { //x좌표를 기준으로 정렬
				Point temp; //바꾸기 위한 변수 선언
				temp = center1; //첫번째 눈의 값을 temp변수에 저장
				center1 = center2; //첫번째 눈 변수에 두번째 눈 저장
				center2 = temp; //두번째 눈 변수에 첫번째 눈 값이 저장되어 있는 temp변수 대입
			}
			int width = abs(center2.x - center1.x); //가로 길이
			int height = abs(center2.y - center1.y); //세로 길이
			if (width > height) { //가로 길이가 세로길이보다 클 때
				float imgScale = width / 330.0; //눈 사이 간격과 안경 알 사이 간격 비율 계산
				int w, h;
				w = glasses.cols * imgScale;  //안경 가로 길이 조정
				h = glasses.rows * imgScale;  //안경 세로 길이 조정
				int offsetX = 150 * imgScale;  //안경 위치 조정
				int offsetY = 160 * imgScale;  //안경 위치 조정
				Mat resized_glasses; //안경의 사이즈 조정
				resize(glasses, resized_glasses, Size(w, h), 0, 0); //계산한 비율로 안경의 크기 조정
				overlayImage(output2, resized_glasses, result, Point(center1.x - offsetX, center1.y - offsetY)); //얼굴 이미지에 안경 이미지 오버랩
				output2 = result; //결과를 output2에 저장
				putText(result, "FPS: " + to_string(int(fpsLive)), { 50, 50 }, FONT_HERSHEY_SIMPLEX, 1.5, (255), 2);//fps값 영상에 문자로 표시
				imshow("sticker", result); //스티거 기능 영상 출력
			}
		}
	}
}
void overlayImage(const Mat& background, const Mat& foreground, Mat& output, Point2i location) //이미지 오버랩(얼굴에 안경 합성)
{
	background.copyTo(output); //이미지 복사	
	for (int y = max(location.y, 0); y < background.rows; ++y)//location이 나타내는 행에서 시작하거나 location.y가 음수인 경우 행 0에서 시작
	{
		int fY = y - location.y; //오버레이 이미지의 y좌표
		if (fY >= foreground.rows) { break; } 	//전경 이미지의 모든 열을 처리
		// location이 표시된 열에서 시작하거나 location.x가 음수이면 0열에서 시작
		for (int x = max(location.x, 0); x < background.cols; ++x)
		{
			int fX = x - location.x; // 오버레이 이미지의 x좌표
			if (fX >= foreground.cols) { break; } //전경 이미지의 모든 행을 처리
			// 네 번째(알파) 채널을 사용하여 전경 픽셀의 불투명도를 결정
			double opacity = ((double)foreground.data[fY * foreground.step + fX * foreground.channels() + 3]) / 255.;
			// 불투명도를 사용하여 배경과 전경 픽셀을 결합, 불투명도 > 0 인 경우만
			for (int c = 0; opacity > 0 && c < output.channels(); ++c)
			{
				unsigned char foregroundPx = foreground.data[fY * foreground.step + fX * foreground.channels() + c]; //전경 픽셀
				unsigned char backgroundPx = background.data[y * background.step + x * background.channels() + c]; //배경 픽셀
				output.data[y * output.step + output.channels() * x + c] = backgroundPx * (1. - opacity) + foregroundPx * opacity; //출력
			}
		}
	}
}
void face_swap() //얼굴 바꾸기 함수 정의
{
	try//예외가 발생할 가능성이 있는 경우
	{
		const size_t num_faces = 2; //2명의 얼굴 바꾸기 
		FaceDetectorAndTracker detector("C:/Users/samsung/source/repos/MyProject/MyProject/haarcascade_frontalface_default.xml", 0, num_faces); //학습된 모델-얼굴 검출
		FaceSwapper face_swapper("C:/Users/samsung/source/repos/MyProject/MyProject/shape_predictor_68_face_landmarks.dat"); //학습된 모델 - 랜드마크 추출 알고리즘
		double fps = cap.get(CAP_PROP_FPS); //초당 프레임 수
		int num_frames = 1; //캡쳐할 프레임 수
		clock_t start; //시작 시간 변수
		clock_t end; //끝나는 시간 변수
		while (true) //반복문
		{
			start = clock(); //시작 시간
			Mat frame; //프레임을 받을 Mat 객체 생성
			detector >> frame; //프레임 변수 detector에 저장
			auto cv_faces = detector.faces(); //얼굴 검출을 cv_faces에 저장
			if (cv_faces.size() == num_faces) //얼굴이 2개 검출됐을 경우
			{
				face_swapper.swapFaces(frame, cv_faces[0], cv_faces[1]); //0번째 얼굴과 1번째 얼굴 변환
			}
			end = clock(); //끝나는 시간
			double seconds = (double(end) - double(start)) / double(CLOCKS_PER_SEC); //경과시간 구하기(초)
			cout << "걸린 시간 : " << seconds << " seconds" << endl; //경과시간 결과값 출력
			fpsLive = double(num_frames) / double(seconds); //실시간 fps 구하기
			cout << "fps : " << fpsLive << endl; //실시간 fps 결과값 출력
			putText(frame, "FPS: " + to_string(int(fpsLive)), { 50, 50 }, FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2); //fps값 영상에 문자로 표시
			imshow("Face Swap", frame); //얼굴 바꾸기 기능 영상 출력
			if (waitKey(30) == 27) //키보드 esc키를 눌렀을 경우
			{
				destroyWindow("Face Swap"); //Face swap 영상 파괴
				break;//탈출
			}
		}
	}
	catch (exception& e) //예외가 발생했을 경우
	{
		cout << "예외 발생 "<< endl; //출력 메세지 출력
	}
}
FaceDetectorAndTracker::FaceDetectorAndTracker(const std::string cascadeFilePath, const int cameraIndex, size_t numFaces) //얼굴 검출, 추적
{
	m_camera = make_unique<VideoCapture>(cameraIndex); //실시간 영상을 받음
	m_faceCascade = make_unique<CascadeClassifier>(cascadeFilePath); //하르 캐스케이드 모델을 받음
#if CV_VERSION_MAJOR < 3 //cv version이 3보다 작은 경우
	m_originalFrameSize.width = (int)m_camera->get(cv::CAP_PROP_FRAME_WIDTH); //프레임 폭
	m_originalFrameSize.height = (int)m_camera->get(cv::CAP_PROP_FRAME_HEIGHT);  //프레임 높이
#else //그 외의 경우
	m_originalFrameSize.width = (int)m_camera->get(cv::CAP_PROP_FRAME_WIDTH); //프레임 폭
	m_originalFrameSize.height = (int)m_camera->get(cv::CAP_PROP_FRAME_HEIGHT); //프레임 높이
#endif
	m_downscaledFrameSize.width = m_downscaledFrameWidth; //축소된 프레임 폭 크기
	m_downscaledFrameSize.height = (m_downscaledFrameSize.width * m_originalFrameSize.height) / m_originalFrameSize.width; //축소된 프레임 높이 크기
	m_ratio.x = (float)m_originalFrameSize.width / m_downscaledFrameSize.width; //x 비율
	m_ratio.y = (float)m_originalFrameSize.height / m_downscaledFrameSize.height; //y 비율
	m_numFaces = numFaces; //얼굴 숫자 저장
}
FaceDetectorAndTracker::~FaceDetectorAndTracker() //소멸자
{
}
void FaceDetectorAndTracker::operator>>(cv::Mat& frame) //얼굴_감지_추적
{
	if (m_camera->isOpened() == false) //카메라가 잘못 열렸을 경우
	{
		frame.release(); //오픈한 frame객체를 frame.release 함수를 이용하여 해제
		return; //반환
	}
	*m_camera >> frame; //프레임을 m_camera에 저장
	resize(frame, m_downscaledFrame, m_downscaledFrameSize); //실시간 영상을 축소시킨 폭과 높이로 사이즈 줄이기
	if (!m_tracking) // 2개의 얼굴을 찾을 때까지 전체 frame에서 얼굴을 검색
	{
		detect(); //함수 호출
		return; //반환
	}
	else //그 외의 경우
	{
		track(); //함수 호출
	}
}
vector<Rect> FaceDetectorAndTracker::faces() //얼굴_감지_추적
{
	vector<Rect> faces; //얼굴 위치 저장(검출된 객체의 사각형 좌표 정보)
	for (const auto& face : m_facesRects)
	{
		faces.push_back(Rect(face.x * m_ratio.x, face.y * m_ratio.y, face.width * m_ratio.x, face.height * m_ratio.y)); 
		//vector 끝에 요소 추가(얼굴 x축*x비율, 얼굴 y축*y비율, 얼굴 높이*x비율, 얼굴 세로*y비율)
	}
	return faces; //얼굴 값 반환
}
void FaceDetectorAndTracker::detect() //2개의 얼굴 찾을 경우 얼굴 탐색
{
	m_faceCascade->detectMultiScale(m_downscaledFrame, m_facesRects, 1.1, 3, 0,Size(m_downscaledFrame.rows / 5, m_downscaledFrame.rows / 5), 
		Size(m_downscaledFrame.rows * 2 / 3, m_downscaledFrame.rows * 2 / 3)); //객체 사각형 영역을 검출(얼굴)(최소 얼굴 크기는 화면 높이의 1/5, 최대 얼굴 크기는 화면 높이의 2/3)

	if (m_facesRects.size() < m_numFaces) //사각형의 크기가 얼굴 크기보다 작을 경우
	{
		return; //반환
	}
	else if (m_facesRects.size() >= m_numFaces) //사각형의 크기가 얼굴 크기보다 클 경우
	{
		m_facesRects.resize(m_numFaces); //사각형의 사이즈 줄이기
	}
	m_faceTemplates.clear(); //벡터를 사용한 뒤 메모리 해제함
	for (auto face : m_facesRects) //얼굴 사각형의 중심(좌표)
	{
		face.width /= 2; //폭을 반으로
		face.height /= 2; //높이를 반으로
		face.x += face.width / 2; //x+폭을 반으로
		face.y += face.height / 2; //y+높이를 반으로 
		m_faceTemplates.push_back(m_downscaledFrame(face).clone()); //축소된 프레임 복사후 m_faceTemplates 끝에 요소 추가
	}
	// 얼굴상의 특정 오브젝트, 특이점(ROI) 얻기
	m_faceRois.clear(); //메모리 해제
	for (const auto& face : m_facesRects) 
	{
		m_faceRois.push_back(doubleRectSize(face, m_downscaledFrameSize)); //m_faceRois 끝에 요소 추가
	}
	m_tracking = true; //추적 켜기
}
void FaceDetectorAndTracker::track() //2개의 얼굴이 아닌 경우
{
	for (int i = 0; i < m_faceRois.size(); i++)
	{
		const auto& roi = m_faceRois[i]; // 얼굴상의 특정 오브젝트
		// 이전 검색에서 가장 큰 얼굴의 +/-20% 크기의 얼굴 감지 
		const Mat& faceRoi = m_downscaledFrame(roi); //얼굴영역 저장
		m_faceCascade->detectMultiScale(faceRoi, m_tmpFacesRect, 1.1, 3, 0,Size(roi.width * 4 / 10, roi.height * 4 / 10),
			Size(roi.width * 6 / 10, roi.width * 6 / 10)); //객체 사각형 영역을 검출(얼굴)
		if (m_tmpFacesRect.empty()) // roi에서 얼굴을 찾을 수 없어서 tm으로 대체
		{
			if (m_tmStartTime[i] == 0) // tm이 방금 스톱워치를 시작한 경우
			{
				m_tmStartTime[i] = getCPUTickCount(); //cpu 정밀한 시간 측정_시작
			}
			if (m_faceTemplates[i].cols <= 1 || m_faceTemplates[i].rows <= 1)
			{
				m_facesRects.clear(); //메모리 해제 
				m_tracking = false; //m_tracking= fasle
				return; //반환
			}
			// 템플릿 매칭
			matchTemplate(faceRoi, m_faceTemplates[i], m_matchingResult, TM_SQDIFF_NORMED); //m_faceTemplates 찾고자 하는 이미지 찾기
			normalize(m_matchingResult, m_matchingResult, 0, 1, NORM_MINMAX, -1, cv::Mat()); //정규화
			double min, max; //변수 선언
			Point minLoc, maxLoc;//포인트 선언
			minMaxLoc(m_matchingResult, &min, &max, &minLoc, &maxLoc); //최대, 최소값 구하기
			// 얼굴 위치에 roi 상쇄
			m_facesRects[i].x = minLoc.x + roi.x - m_faceTemplates[i].cols / 2; //얼굴 영역 사각형의 x
			m_facesRects[i].y = minLoc.y + roi.y - m_faceTemplates[i].rows / 2; //얼굴 영역 사각형의 y
			m_facesRects[i].width = m_faceTemplates[i].cols * 2; //얼굴 영역 사각형의 폭
			m_facesRects[i].height = m_faceTemplates[i].rows * 2; //얼굴 영역 사각형의 높이
			m_tmEndTime[i] = getCPUTickCount();//cpu 정밀한 시간 측정_끝
			double duration = (double)(m_tmEndTime[i] - m_tmStartTime[i]) / getTickFrequency(); //지속
			if (duration > m_tmMaxDuration) //지속시간이 최대보다 크다면
			{
				m_facesRects.clear(); //vector 메모리 해제
				m_tracking = false; //m_tracking=false
				return; // 얼굴 검출 반환
			}
		}
		else //다른 경우
		{
			m_tmRunningInRoi[i] = false; //m_tmRunningInRoi=false
			m_tmStartTime[i] = m_tmEndTime[i] = 0; //시작 시간과 끝 시간->0
			m_facesRects[i] = m_tmpFacesRect[0]; //tm얼굴사각형의 0번째 요소를 m_face_Rects에 저장
			m_facesRects[i].x += roi.x; //d얼굴 영역 사각형의 x
			m_facesRects[i].y += roi.y; //얼굴 영역 사각형의 y
		}
	}
	for (int i = 0; i < m_facesRects.size(); i++)
	{
		for (int j = i + 1; j < m_facesRects.size(); j++) 
		{
			if ((m_facesRects[i] & m_facesRects[j]).area() > 0)
			{
				m_facesRects.clear(); //vector 메모리 해제
				m_tracking = false; //m_tracking=false
				return;//얼굴 검출 반환
			}
		}
	}
}
Rect FaceDetectorAndTracker::doubleRectSize(const Rect& inputRect, const Size& frameSize)
{
	Rect outputRect; 
	// 두개의 직사각형 크기
	outputRect.width = inputRect.width * 2; //출력 사각형 폭
	outputRect.height = inputRect.height * 2; //출력 사각형 높이 
	// 원래 중심 주위의 중심 사각형
	outputRect.x = inputRect.x - inputRect.width / 2; //출력 사각형(입력 x-폭을 반으로)
	outputRect.y = inputRect.y - inputRect.height / 2; //출력 사각형(입력 y-높이를 반으로)
	// Handle edge cases(에지 처리)
	if (outputRect.x < 0) { //출력 사각형의 x축이 0보다 작을 경우
		outputRect.width += outputRect.x; //출력 사각형 폭
		outputRect.x = 0; //x값 초기화
	}
	if (outputRect.y < 0) {//출력 사각형의 y축이 0보다 작을 경우
		outputRect.height += outputRect.y; //출력 사각형 높이
		outputRect.y = 0; //y값 초기화
	}
	if (outputRect.x + outputRect.width > frameSize.width) { //가록와 폭의 합이 폭의 사이즈보다 클 경우
		outputRect.width = frameSize.width - outputRect.x; //폭-가로
	}
	if (outputRect.y + outputRect.height > frameSize.height) { //세로와 높이의 합이 높이의 사이즈보타 클 경우
		outputRect.height = frameSize.height - outputRect.y; //높이-세로
	}
	return outputRect; //출력 사각형 반환
}
FaceSwapper::FaceSwapper(const std::string landmarks_path) //얼굴바꾸기_랜드마크 경로
{
	try  //예외 발생에 대한 검사의 범위 지정
	{
		dlib::deserialize(landmarks_path) >> pose_model; //pose_model을 외부 파일의 데이터를 프로그램 내의 object로 읽기
	}
	catch (std::exception& e) //try 블록에서 발생한 예외를 처리
	{
		cout << "Error loading landmarks from " << landmarks_path << endl //랜드마크를 찾을 수 없다는 에러 표시
			<< "You can download the file from http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2" << endl; //얼굴 바꾸는 학습 모델 다운하라는 메세지 출력
		exit(-1); //에러메세지 종료
	}
}
FaceSwapper::~FaceSwapper() //소멸자
{
}
void FaceSwapper::swapFaces(Mat& frame, Rect& rect_ann, Rect& rect_bob) //매개변수가 3개인 생성자
{
	small_frame = getMinFrame(frame, rect_ann, rect_bob); //최소 프레임
	frame_size = Size(small_frame.cols, small_frame.rows); //영상 사이즈
	getFacePoints(small_frame); //얼굴 포인트 얻기
	getTransformationMatrices(); //변환 매트릭스 함수
	mask_ann.create(frame_size, CV_8UC1); //frame_size를 1채널 그레이스케일 영상으로 만들기
	mask_bob.create(frame_size, CV_8UC1); //frame_size를 1채널 그레이스케일 영상으로 만들기
	getMasks(); //볼록 다각형 그리기
	getWarppedMasks(); //어파인 변환
	refined_masks = getRefinedMasks(); //영상 합성, 이미지 복사하여 refined_masks에 저장
	extractFaces(); //얼굴 추출(이미지 복사)
	warpped_faces = getWarppedFaces(); //얼굴 어파인 변환해서 warpped_faces에 저장
	colorCorrectFaces(); //알맞는 얼굴 색상
	auto refined_mask_ann = refined_masks(big_rect_ann);
	auto refined_mask_bob = refined_masks(big_rect_bob);
	featherMask(refined_mask_ann); //노이즈 제거, 영상을 부드럽게 함
	featherMask(refined_mask_bob); //노이즈 제거, 영상을 부드럽게 함
	pasteFacesOnFrame(); //프레임에 얼굴 붙여넣기
}
Mat FaceSwapper::getMinFrame(const Mat& frame, Rect& rect_ann, Rect& rect_bob) //최소 프레임 가져오기
{
	Rect bounding_rect = rect_ann | rect_bob; //비트 포괄 OR 연산자(어느 한쪽 비트가 1이면 결과 1로 설정)
	bounding_rect -= Point(50, 50); 
	bounding_rect += Size(100, 100);
	bounding_rect &= Rect(0, 0, frame.cols, frame.rows);
	this->rect_ann = rect_ann - bounding_rect.tl();
	this->rect_bob = rect_bob - bounding_rect.tl();
	big_rect_ann = ((this->rect_ann - Point(rect_ann.width / 4, rect_ann.height / 4)) + Size(rect_ann.width / 2, rect_ann.height / 2)) & cv::Rect(0, 0, bounding_rect.width, bounding_rect.height);
	big_rect_bob = ((this->rect_bob - Point(rect_bob.width / 4, rect_bob.height / 4)) + Size(rect_bob.width / 2, rect_bob.height / 2)) & cv::Rect(0, 0, bounding_rect.width, bounding_rect.height);
	return frame(bounding_rect);
}

void FaceSwapper::getFacePoints(const Mat& frame) //얼굴 포인트 얻기
{
	using namespace dlib; //dlib 라이브러리 사용
	dlib_rects[0] = dlib::rectangle(rect_ann.x, rect_ann.y, rect_ann.x + rect_ann.width, rect_ann.y + rect_ann.height); //x,y,x+폭,y+높이 사각형
	dlib_rects[1] = dlib::rectangle(rect_bob.x, rect_bob.y, rect_bob.x + rect_bob.width, rect_bob.y + rect_bob.height); //x,y,x+폭,y+높이 사각형
	dlib_frame = frame; //frame영상을 dlib_frame에 저장
	shapes[0] = pose_model(dlib_frame, dlib_rects[0]); //실시간 영상의 (x,y,x+폭,y+높이)를 shape 0번째 요소에 저장
	shapes[1] = pose_model(dlib_frame, dlib_rects[1]);//실시간 영상의 (x,y,x+폭,y+높이)를 shape 1번째 요소에 저장
	auto getPoint = [&](int shape_index, int part_index) -> const cv::Point2i
	{
		const auto& p = shapes[shape_index].part(part_index);
		return cv::Point2i(p.x(), p.y());
	};
	//(얼굴 랜드 마크) 턱
	points_ann[0] = getPoint(0, 0);
	points_ann[1] = getPoint(0, 3);
	points_ann[2] = getPoint(0, 5);
	points_ann[3] = getPoint(0, 8);
	points_ann[4] = getPoint(0, 11);
	points_ann[5] = getPoint(0, 13);
	points_ann[6] = getPoint(0, 16);
	Point2i nose_length = getPoint(0, 27) - getPoint(0, 30); //콧대
	points_ann[7] = getPoint(0, 26) + nose_length; //입슬
	points_ann[8] = getPoint(0, 17) + nose_length; //입술
	//턱
	points_bob[0] = getPoint(1, 0);
	points_bob[1] = getPoint(1, 3);
	points_bob[2] = getPoint(1, 5);
	points_bob[3] = getPoint(1, 8);
	points_bob[4] = getPoint(1, 11);
	points_bob[5] = getPoint(1, 13);
	points_bob[6] = getPoint(1, 16);
	nose_length = getPoint(1, 27) - getPoint(1, 30); //콧대
	points_bob[7] = getPoint(1, 26) + nose_length; //입술
	points_bob[8] = getPoint(1, 17) + nose_length; //입술
	affine_transform_keypoints_ann[0] = points_ann[3]; //턱
	affine_transform_keypoints_ann[1] = getPoint(0, 36); //오른쪽 눈
	affine_transform_keypoints_ann[2] = getPoint(0, 45); //왼쪽 눈
	affine_transform_keypoints_bob[0] = points_bob[3]; //턱
	affine_transform_keypoints_bob[1] = getPoint(1, 36); //오른쪽 눈
	affine_transform_keypoints_bob[2] = getPoint(1, 45); //왼쪽 눈
	feather_amount.width = feather_amount.height = (int)norm(points_ann[0] - points_ann[6]) / 8; //눈썹
}

void FaceSwapper::getTransformationMatrices() //변형 행렬 얻기
{
	trans_ann_to_bob = getAffineTransform(affine_transform_keypoints_ann, affine_transform_keypoints_bob);//affine_transform_keypoints_ann에 저장된 세점을 affine_transform_keypoints_bob으로 옮기는 어파인 변환
	//3쌍의 입력 매칭쌍으로부터 affine변환을 구해줌
	invertAffineTransform(trans_ann_to_bob, trans_bob_to_ann); //역변환 구하기
}
void FaceSwapper::getMasks() //마스크 얻기
{
	mask_ann.setTo(Scalar::all(0)); //mask_ann 행렬의 모든 원소를 0으로 초기화
	mask_bob.setTo(Scalar::all(0)); //mask_bob 행렬의 모든 원소를 0으로 초기화
	fillConvexPoly(mask_ann, points_ann, 9, Scalar(255)); //mask_ann의 영상의 좌표(points_ann,9)를 이용해 255색상으로 볼록 다각형을 그림
	fillConvexPoly(mask_bob, points_bob, 9, Scalar(255)); //mask_bob의 영상의 좌표(points_bob,9)를 이용해 255색상으로 볼록 다각형을 그림
}

void FaceSwapper::getWarppedMasks() //어파인 변환(뒤틀린 얼굴 개선)-기하학적 변환
{
	warpAffine(mask_ann, warpped_mask_ann, trans_ann_to_bob, frame_size, INTER_NEAREST, BORDER_CONSTANT, Scalar(0)); //mask_ann영상을 어파인 변환하여 warpped_mask_ann 영상 생성
	warpAffine(mask_bob, warpped_mask_bob, trans_bob_to_ann, frame_size, INTER_NEAREST, BORDER_CONSTANT, Scalar(0));//mask_bob 영상을 어파인 변환하여 warpped_mask_bob 영상 생성
}
Mat FaceSwapper::getRefinedMasks() //이미지 비트 연산(사진 합성)
{
	bitwise_and(mask_ann, warpped_mask_bob, refined_ann_and_bob_warpped); //mask_ann과 warpped_mask_bob의 영상이 모두 true이면 검정색,결과값 refined_ann_and_bob_warpped
	bitwise_and(mask_bob, warpped_mask_ann, refined_bob_and_ann_warpped); //mask_bob과 warpped_mask_ann의 영상이 모두 true이면 검정색,결과값 refined_bob_and_and_warpped
	Mat refined_masks(frame_size, CV_8UC1, Scalar(0)); //redined_masks 객체 생성
	refined_ann_and_bob_warpped.copyTo(refined_masks, refined_ann_and_bob_warpped); //이미지 복사(refined_masks:복사본이 저장될 행렬)
	refined_bob_and_ann_warpped.copyTo(refined_masks, refined_bob_and_ann_warpped); //이미지 복사(refined_masks:복사본이 저장될 행렬)
	return refined_masks; //함수 결과값 반환
}
void FaceSwapper::extractFaces() //얼굴 추출 (이미지 복사)
{
	small_frame.copyTo(face_ann, mask_ann); //small_frame에서 mask_ann의 부분을 복사하여 face_ann에 저장
	small_frame.copyTo(face_bob, mask_bob); //small_frame에서 mask_bob의 부분을 복사하여 face_bob에 저장
}
cv::Mat FaceSwapper::getWarppedFaces() //어파인 변환_얼굴
{
	Mat warpped_faces(frame_size, CV_8UC3, Scalar::all(0)); //warpped_faces 객체 생성
	warpAffine(face_ann, warpped_face_ann, trans_ann_to_bob, frame_size, INTER_NEAREST, BORDER_CONSTANT, Scalar(0, 0, 0)); //face_ann영상을 어파인 변환하여 warpped_face_ann 영상 생성
	warpAffine(face_bob, warpped_face_bob, trans_bob_to_ann, frame_size, INTER_NEAREST, BORDER_CONSTANT, Scalar(0, 0, 0));  //face_bob영상을 어파인 변환하여 warpped_face_bob 영상 생성
	warpped_face_ann.copyTo(warpped_faces, warpped_mask_ann); //warpped_face_ann에서 warpped_mask_ann의 부분을 복사하여 warpped_faces에 저장
	warpped_face_bob.copyTo(warpped_faces, warpped_mask_bob); //warpped_face_bob에서 warpped_mask_bob의 부분을 복사하여 warpped_faces에 저장
	return warpped_faces; //함수 결과값 반환
}
void FaceSwapper::colorCorrectFaces() //알맞는 얼굴 색상
{
	specifiyHistogram(small_frame(big_rect_ann), warpped_faces(big_rect_ann), warpped_mask_bob(big_rect_ann)); //히스토그램 지정 
	specifiyHistogram(small_frame(big_rect_bob), warpped_faces(big_rect_bob), warpped_mask_ann(big_rect_bob)); //히스토그램 지정
}
void FaceSwapper::featherMask(Mat& refined_masks) //노이즈 제거
{
	erode(refined_masks, refined_masks, getStructuringElement(MORPH_RECT, feather_amount), Point(-1, -1), 1, BORDER_CONSTANT, Scalar(0)); //침식연산(노이즈 제거)
	blur(refined_masks, refined_masks, feather_amount, Point(-1, -1), BORDER_CONSTANT); //영상을 부드럽게 함
}
inline void FaceSwapper::pasteFacesOnFrame() //프레임에 얼굴 붙여넣기
{
	for (size_t i = 0; i < small_frame.rows; i++)
	{
		//auto 키워드를 사용하면 초깃값의 형식에 맞춰 선언하는 인스턴스
		auto frame_pixel = small_frame.row(i).data; //최소 프레임 데이터를 frame_pixel에 저장
		auto faces_pixel = warpped_faces.row(i).data; //어파인 변환한 영상을 faces_pixel에 저장
		auto masks_pixel = refined_masks.row(i).data; //노이즈 제거한 영상을 masks_pixel에 저장
		for (size_t j = 0; j < small_frame.cols; j++)
		{
			if (*masks_pixel != 0) //노이즈 제거한 영상이 0이 아닐 경우
			{
				*frame_pixel = ((255 - *masks_pixel) * (*frame_pixel) + (*masks_pixel) * (*faces_pixel)) >> 8; // 256으로 나누기
				*(frame_pixel + 1) = ((255 - *(masks_pixel + 1)) * (*(frame_pixel + 1)) + (*(masks_pixel + 1)) * (*(faces_pixel + 1))) >> 8; // 256으로 나누기
				*(frame_pixel + 2) = ((255 - *(masks_pixel + 2)) * (*(frame_pixel + 2)) + (*(masks_pixel + 2)) * (*(faces_pixel + 2))) >> 8; // 256으로 나누기
			}
			frame_pixel += 3;//다음 픽셀로 이동
			faces_pixel += 3;//다음 픽셀로 이동
			masks_pixel++;//다음 픽셀로 이동
		}
	}
}
void FaceSwapper::specifiyHistogram(const Mat source_image, Mat target_image, Mat mask) //히스토그램 지정
{
	memset(source_hist_int, 0, sizeof(int) * 3 * 256); //메모리 크기를 0으로 초기화
	memset(target_hist_int, 0, sizeof(int) * 3 * 256); //메모리 크기를 0으로 초기화
	for (size_t i = 0; i < mask.rows; i++)
	{
		auto current_mask_pixel = mask.row(i).data; //현재 마스크 픽셀
		auto current_source_pixel = source_image.row(i).data; //현재 소스 픽셀
		auto current_target_pixel = target_image.row(i).data; //현재 대상 픽셀
		for (size_t j = 0; j < mask.cols; j++)
		{
			if (*current_mask_pixel != 0) { //현재 마스크 픽셀이 0이 아닐 경우
				source_hist_int[0][*current_source_pixel]++; 
				source_hist_int[1][*(current_source_pixel + 1)]++; 
				source_hist_int[2][*(current_source_pixel + 2)]++;

				target_hist_int[0][*current_target_pixel]++;
				target_hist_int[1][*(current_target_pixel + 1)]++;
				target_hist_int[2][*(current_target_pixel + 2)]++;
			}	
			current_source_pixel += 3;// 다음 픽셀로 이동
			current_target_pixel += 3;// 다음 픽셀로 이동
			current_mask_pixel++;// 다음 픽셀로 이동
		}
	}
	// CDF(누적분포함수: 확률변수가 특정값보다 작거나 같은 확률을 나타내는 함수) 계산
	for (size_t i = 1; i < 256; i++)
	{
		source_hist_int[0][i] += source_hist_int[0][i - 1]; //히스토그램 누적 함수 계산
		source_hist_int[1][i] += source_hist_int[1][i - 1];//히스토그램 누적 함수 계산
		source_hist_int[2][i] += source_hist_int[2][i - 1];//히스토그램 누적 함수 계산
		target_hist_int[0][i] += target_hist_int[0][i - 1];//히스토그램 누적 함수 계산
		target_hist_int[1][i] += target_hist_int[1][i - 1];//히스토그램 누적 함수 계산
		target_hist_int[2][i] += target_hist_int[2][i - 1];//히스토그램 누적 함수 계산
	}
	// CDF(누적분포함수) 정규화
	for (size_t i = 0; i < 256; i++)
	{
		source_histogram[0][i] = (source_hist_int[0][255] ? (float)source_hist_int[0][i] / source_hist_int[0][255] : 0);  // 히스토그램 균등화
		source_histogram[1][i] = (source_hist_int[1][255] ? (float)source_hist_int[1][i] / source_hist_int[1][255] : 0);  // 히스토그램 균등화
		source_histogram[2][i] = (source_hist_int[2][255] ? (float)source_hist_int[2][i] / source_hist_int[2][255] : 0);  // 히스토그램 균등화
		target_histogram[0][i] = (target_hist_int[0][255] ? (float)target_hist_int[0][i] / target_hist_int[0][255] : 0);  // 히스토그램 균등화
		target_histogram[1][i] = (target_hist_int[1][255] ? (float)target_hist_int[1][i] / target_hist_int[1][255] : 0);  // 히스토그램 균등화
		target_histogram[2][i] = (target_hist_int[2][255] ? (float)target_hist_int[2][i] / target_hist_int[2][255] : 0);  // 히스토그램 균등화
	}
	// 조회(룩업) 테이블 만들기(주어진 연산에 대해 미리 계산된 결과들의 배열)
	auto binary_search = [&](const float needle, const float haystack[]) -> uint8_t //(0~255)
	{
		uint8_t l = 0, r = 255, m; //변수 선언
		while (l < r) //255보다 작을 경우
		{
			m = (l + r) / 2; 
			if (needle > haystack[m])
				l = m + 1;
			else
				r = m - 1;
		}
		return m; //m값 반환
	};
	for (size_t i = 0; i < 256; i++)
	{
		LUT[0][i] = binary_search(target_histogram[0][i], source_histogram[0]); //이진 탐색(찾는 값이 존재하면 true,아니면 false)
		LUT[1][i] = binary_search(target_histogram[1][i], source_histogram[1]); //이진 탐색
		LUT[2][i] = binary_search(target_histogram[2][i], source_histogram[2]); //이진 탐색
	} 
	//픽셀 다시 칠하기
	for (size_t i = 0; i < mask.rows; i++)
	{
		auto current_mask_pixel = mask.row(i).data; //현재 마스크 픽셀
		auto current_target_pixel = target_image.row(i).data; //현재 타켓 픽셀
		for (size_t j = 0; j < mask.cols; j++)
		{
			if (*current_mask_pixel != 0) //현재 마스크 픽셀이 0이 아닐 경우
			{
				*current_target_pixel = LUT[0][*current_target_pixel]; //현재 타켓 픽셀
				*(current_target_pixel + 1) = LUT[1][*(current_target_pixel + 1)]; //현재 타겟 픽셀
				*(current_target_pixel + 2) = LUT[2][*(current_target_pixel + 2)]; //현재 타겟 픽셀
			}
			current_target_pixel += 3;//다음 픽셀로 이동
			current_mask_pixel++;//다음 픽셀로 이동
		}
	}
}
