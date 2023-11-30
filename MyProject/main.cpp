
#include <opencv2/highgui/highgui.hpp> //������� ���� 
#include "FaceDetectorAndTracker.h" //������� ����
#include "FaceSwapper.h" //������� ����

using namespace std; //�̸����� ���
using namespace cv; //�̸����� ���
//dlib - �ȸ� ���帶ŷ ���̺귯��(�ȸ��� �� Ư¡�κп� ���帶ŷ ���� ����)-svm �ӽŷ���
int main()
{

    try //���� �߻��� ���� �˻��� ���� ����
    {
        const size_t num_faces = 2; //��ȣ ���� ���� ����(�̷������� ������ ��� ������ ��ü�� �ִ� ũ�� ���� ����)
        FaceDetectorAndTracker detector("C:/Users/samsung/source/repos/MyProject/MyProject/haarcascade_frontalface_default.xml", 0, num_faces); //�� ����� �� ������ �Լ�
        FaceSwapper face_swapper("C:/Users/samsung/source/repos/MyProject/MyProject/shape_predictor_68_face_landmarks.dat");  //���� �ٲٴ� �Լ�

        while (true) //�ݺ���
        {
               
            Mat frame;  //�������� ���� Nat ��ü ����
            detector >> frame; //ī�޶�κ��� frame�� �������� ����

            auto cv_faces = detector.faces(); //���� �ȿ� �ִ� �ڷ����� faces �Լ�, �������� 
            if (cv_faces.size() == num_faces) //faces�Լ��� ũ�Ⱑ 2�� ���� ��
            {
                face_swapper.swapFaces(frame, cv_faces[0], cv_faces[1]); //FaceSwapper Ŭ������ �Ű������� ����
            }

            imshow("Face Swap", frame);  // ȭ�鿡 ǥ��

            if (waitKey(1) == 27) return 0; //1ms�� ��ٸ��� ���� �̹����� display ,esc�� ������ ����
        }
    }
    catch (exception& e) //try ��Ͽ��� �߻��� ���ܸ� ó��
    {
        cout << e.what() << endl; //��� �޼��� 
    }
}

/*
int main()
{

    try
    {
        const size_t num_faces = 2;
        FaceDetectorAndTracker detector("C:/Users/samsung/source/repos/MyProject/MyProject/haarcascade_frontalface_default.xml", 0, num_faces);
        //�� ����� �� ������ �Լ�
        FaceSwapper face_swapper("C:/Users/samsung/source/repos/MyProject/MyProject/shape_predictor_68_face_landmarks.dat");
        //���� �ٲٴ� �Լ�

      //  double fps = 0;
        while (true) //�ݺ���
        {
            auto time_start = cv::getTickCount(); //auto�� ������ �ʱ�ȭ �Ŀ��� ������ �߷еǴ� ������ �����ϴ� ����,
            //getTicKCount�� OS������ ������ ������ �ð��� msec ������ �����ִ� �Լ�

            // Grab a frame
            Mat frame;  //Mat Ŭ����
            detector >> frame; //��Ʈ ������(����Ʈ ������) ��Ʈ�� �������� �̵�

            auto cv_faces = detector.faces(); //���� �ȿ� �ִ� �ڷ����� faces �Լ�
            if (cv_faces.size() == num_faces) //faces�Լ��� ũ�Ⱑ 2�� ���� ��
            {
                face_swapper.swapFaces(frame, cv_faces[0], cv_faces[1]); //FaceSwapper Ŭ������ �Ű������� ����
            }

           // auto time_end = cv::getTickCount();   //getTicKCount�� OS������ ������ ������ �ð��� msec ������ �����ִ� �Լ� <������ �ð�>
           // auto time_per_frame = (time_end - time_start) / cv::getTickFrequency(); //����ð��� ����

          //  fps = (15 * fps + (1 / time_per_frame)) / 16; //�ʴ� ������ ��

         //    printf("Total time: %3.5f | FPS: %3.2f\n", time_per_frame, fps); //��� �޼���


            imshow("Face Swap", frame);  // Display it all on the screen

            if (waitKey(1) == 27) return 0; //1ms�� ��ٸ��� ���� �̹����� display ,esc�� ������ ����
        }
    }
    catch (exception& e) //exception class
    {
        cout << e.what() << endl; //��� �޼���
    }
}
*/