
#include <opencv2/highgui/highgui.hpp> //헤더파일 포함 
#include "FaceDetectorAndTracker.h" //헤더파일 포함
#include "FaceSwapper.h" //헤더파일 포함

using namespace std; //이름공간 사용
using namespace cv; //이름공간 사용
//dlib - 안면 랜드마킹 라이브러리(안면의 각 특징부분에 랜드마킹 점을 추출)-svm 머신러닝
int main()
{

    try //예외 발생에 대한 검사의 범위 지정
    {
        const size_t num_faces = 2; //부호 없는 정수 형식(이론적으로 가능한 모든 유형의 객체의 최대 크기 저장 가능)
        FaceDetectorAndTracker detector("C:/Users/samsung/source/repos/MyProject/MyProject/haarcascade_frontalface_default.xml", 0, num_faces); //얼굴 검출기 및 추적기 함수
        FaceSwapper face_swapper("C:/Users/samsung/source/repos/MyProject/MyProject/shape_predictor_68_face_landmarks.dat");  //얼굴을 바꾸는 함수

        while (true) //반복문
        {
               
            Mat frame;  //프레임을 받을 Nat 객체 생성
            detector >> frame; //카메라로부터 frame에 프레임을 받음

            auto cv_faces = detector.faces(); //벡터 안에 있는 자료형의 faces 함수, 지역변수 
            if (cv_faces.size() == num_faces) //faces함수의 크기가 2와 같을 때
            {
                face_swapper.swapFaces(frame, cv_faces[0], cv_faces[1]); //FaceSwapper 클래스의 매개변수에 대입
            }

            imshow("Face Swap", frame);  // 화면에 표시

            if (waitKey(1) == 27) return 0; //1ms을 기다리고 다음 이미지를 display ,esc를 누르면 종료
        }
    }
    catch (exception& e) //try 블록에서 발생한 예외를 처리
    {
        cout << e.what() << endl; //결과 메세지 
    }
}

/*
int main()
{

    try
    {
        const size_t num_faces = 2;
        FaceDetectorAndTracker detector("C:/Users/samsung/source/repos/MyProject/MyProject/haarcascade_frontalface_default.xml", 0, num_faces);
        //얼굴 검출기 및 추적기 함수
        FaceSwapper face_swapper("C:/Users/samsung/source/repos/MyProject/MyProject/shape_predictor_68_face_landmarks.dat");
        //얼굴을 바꾸는 함수

      //  double fps = 0;
        while (true) //반복문
        {
            auto time_start = cv::getTickCount(); //auto는 선언의 초기화 식에서 형식이 추론되는 변수를 선언하는 역할,
            //getTicKCount는 OS부팅할 때부터 지나간 시간을 msec 단위로 돌려주는 함수

            // Grab a frame
            Mat frame;  //Mat 클래스
            detector >> frame; //비트 연산자(시프트 연산자) 비트를 왼쪽으로 이동

            auto cv_faces = detector.faces(); //벡터 안에 있는 자료형의 faces 함수
            if (cv_faces.size() == num_faces) //faces함수의 크기가 2와 같을 때
            {
                face_swapper.swapFaces(frame, cv_faces[0], cv_faces[1]); //FaceSwapper 클래스의 매개변수에 대입
            }

           // auto time_end = cv::getTickCount();   //getTicKCount는 OS부팅할 때부터 지나간 시간을 msec 단위로 돌려주는 함수 <끝나는 시간>
           // auto time_per_frame = (time_end - time_start) / cv::getTickFrequency(); //연산시간을 측정

          //  fps = (15 * fps + (1 / time_per_frame)) / 16; //초당 프레임 수

         //    printf("Total time: %3.5f | FPS: %3.2f\n", time_per_frame, fps); //결과 메세지


            imshow("Face Swap", frame);  // Display it all on the screen

            if (waitKey(1) == 27) return 0; //1ms을 기다리고 다음 이미지를 display ,esc를 누르면 종료
        }
    }
    catch (exception& e) //exception class
    {
        cout << e.what() << endl; //결과 메세지
    }
}
*/