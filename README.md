# Tensorflow object detection

- ไฟล์แบบที่ 1. 01_TFOD_installation.ipynb เอาไว้ลองรันเทรนเองตั้งแต่เริ่ม ให้รันบน colab นะ
- ไฟล์แบบที่ 2.
  - Protoc : https://github.com/protocolbuffers/protobuf/releases/download/v3.18.0/protoc-3.18.0-win64.zip
  - ติดตั้ง protoc ไว้ในไดรฟ์ C:\\AdditionalPackage (สร้างโฟล์เดอร์นี้มา)
  - สร้างโฟลเดอร์ขึ้นมา 1 โฟล์เดอร์แล้วโหลดทั้งหมดนี้ไว้
  - โหลดไฟล์ 02_TFOD_webcam.ipynb 
  - โหลดไฟล์ Anno&Dataset.rar
  - โหลดไฟล์ TensorFlow_ws -> https://drive.google.com/file/d/1-16TUTMyswL9J1ej7WnAuNEt2tmqOfi9/view?usp=sharing
  - โหลดไฟล์ model ที่เทรนแล้ว -> https://drive.google.com/file/d/1uhHiyPaAHTHe79VAZy30pxn3Je-CoY4H/view?usp=sharing
  - แตกไฟล์ทั้งหมด Anno&dataset จะมี 2 โฟลเดอร์ให้ทำตามนี้ 
    1. เอาโฟล์เดอร์ annotations ย้ายไปใน TensorFlow_ws/workspace/training_demo/
    2. เอาโฟล์เดอร์ images ย้ายไปใน TensorFlow_ws/workspace/training_demo/
    3. เอา model ที่เทรนแล้วทั้งโฟลเดอร์ ย้ายไปใน TensorFlow_ws/workspace/training_demo/exported-models
    4. รันไฟล์ 02_TFOD_webcam.ipynb
   หน้าตาโฟลเดอร์ประมาณนี้ แล้วใช้ vscode รันไฟล์ 02_TFOD_webcam ได้เลย
![image](https://user-images.githubusercontent.com/42464592/135790931-305976dd-f82b-4edd-af5a-21d9d32d6d7e.png)
![image](https://user-images.githubusercontent.com/42464592/135790991-e4874be6-762c-4ed2-91b8-914807c4e546.png)

