# Online Exam Cheating Detection 🎓🤖

Hệ thống phát hiện gian lận trong thi trực tuyến sử dụng **Trí tuệ nhân tạo (AI)** và **Computer Vision**.  
Dự án tập trung vào việc kết hợp nhiều mô hình thị giác máy tính nhằm giám sát hành vi thí sinh trong môi trường thi online.

---

## 📌 Mục tiêu
- Phát hiện các hành vi gian lận phổ biến trong thi trực tuyến
- Giám sát hướng nhìn và tư thế đầu của thí sinh
- Phát hiện các đối tượng khả nghi (điện thoại, người thứ hai, tài liệu, …)
- Hỗ trợ giám thị trong việc đánh giá và phát hiện bất thường

---

## 🧠 Các mô hình sử dụng
- **YOLO (YOLOv8 / YOLOv11)**: phát hiện đối tượng gian lận
- **L2CS-Net**: ước lượng hướng nhìn (Gaze Estimation)
- **SixDRepNet**: ước lượng tư thế đầu (Head Pose Estimation)
- **MediaPipe FaceMesh**: trích xuất landmark khuôn mặt
- **Rule-based Engine**: phân tích hành vi gian lận dựa trên luật

---

## ⚙️ Pipeline hệ thống
1. Nhận video hoặc webcam từ thí sinh
2. Phát hiện đối tượng bằng YOLO
3. Căn chỉnh và crop khuôn mặt
4. Ước lượng:
   - Hướng nhìn (L2CS-Net)
   - Tư thế đầu (SixDRepNet)
5. Phân tích hành vi bằng luật (rules)
6. Hiển thị kết quả và cảnh báo gian lận

---

## 📊 Dataset (Bộ dữ liệu tự xây dựng)

Bộ dữ liệu được **tự xây dựng** nhằm phục vụ bài toán phát hiện gian lận trong thi trực tuyến.  
Dữ liệu được thu thập từ video webcam và video mô phỏng môi trường thi online, sau đó trích xuất thành các khung hình (frames).

Quá trình gán nhãn được thực hiện **thủ công** thông qua nền tảng **Roboflow**, bộ dữ liệu được quản lý và phiên bản hóa tại:

🔗 **Roboflow Dataset – Version 9**  
https://app.roboflow.com/nhn-dng-vt-th/online-exam-proctoring-wjh05/9

Bộ dữ liệu bao gồm các lớp đối tượng và hành vi liên quan đến gian lận trong thi trực tuyến, ví dụ:
- `phone`
- `book`
- `extra_person`
- `absence`
- …

Dữ liệu được thu thập trong nhiều điều kiện khác nhau về **ánh sáng, góc quay và bối cảnh** nhằm tăng tính đa dạng và khả năng tổng quát hóa của mô hình.  
Sau khi gán nhãn, dữ liệu được chia thành các tập **train / validation / test** để huấn luyện và đánh giá mô hình YOLO.

⚠️ Do liên quan đến **quyền riêng tư** và **dung lượng lớn**, bộ dữ liệu **không được công bố trực tiếp trên GitHub**, chỉ được quản lý trên Roboflow và cung cấp theo yêu cầu cho mục đích học tập và nghiên cứu.

---

## 📦 Model Weights
Do GitHub giới hạn dung lượng file, các trọng số mô hình **không được đính kèm trong repository**.

Tải model tại Google Drive:  
🔗 https://drive.google.com/drive/folders/1HjF7Wc_q62KblFQCDwmEPPb_-fxBCsLX?hl=vi

Bao gồm:
- YOLO weights (`.pt`)
- L2CS-Net gaze model (`.onnx`)
- SixDRepNet head pose model (`.onnx`)

---
