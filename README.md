# 블록도
![image](https://github.com/LeeNaYoung240/Camera_App/assets/107848521/c06a6b80-e748-4d94-ba52-97dd751e3b94)

# 설명
readNet함수를 통해 학습된 데이터(Teachable Machine 이용)를 불러오고 마우스 이벤트를 통해 기능을 수행하게 합니다. M, F, C, D, X를 그릴 경우에 따라 다른 함수를 호출합니다.

1) M일 경우, mosic() 함수 호출  -> 모자이크 기능 수행
2) F일 경우, face_swap() 함수 호출 -> 얼굴 바꾸기 기능 수행
3) C일 경우, cartoon() 함수 호출  -> 만화 질감 처리 기능 수행
4) D일 경우, sticker() 함수 호출 -> 스티커 기능 수행
5) X일 경우, end_x()함수 호출 -> 마우스 이벤트 창 종료(전체 종료)

![image](https://github.com/LeeNaYoung240/Camera_App/assets/107848521/f62a551e-52b8-4fd4-8621-a24e690d580a)

TM(Teachable Machine)을 활용하여 필기체를 학습합니다. M, F, C, D, X를 총 100개씩 500개의 학습 데이터를 만든 뒤, OpenCV용 데이터 포맷으로 변환합니다. 다음 학습 완룍된 모델 파일을 통해 마우스 이벤트의 동작에 따라 필기체를 인식하게 됩니다.

esc키만 누를 경우 계속 기능을 무한루프로 돌게 되고 x를 그려야 최종적으로 종료가 되고 기능인 마우스 이벤트를 그릴 때 마다 인식률을 출력하게 되고 걸린 시간과 fps를 실시간으로 받게 됩니다.

# 결과 영상
https://www.youtube.com/watch?v=gR2mlCJo6M8
