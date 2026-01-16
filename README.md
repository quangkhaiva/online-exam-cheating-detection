# Online Exam Cheating Detection 🎓🤖

Hệ thống phát hiện gian lận trong thi trực tuyến sử dụng Trí tuệ nhân tạo và Computer Vision.  
Đề tài tập trung vào việc kết hợp nhiều mô hình thị giác máy tính để giám sát hành vi thí sinh trong môi trường thi online.

---

## 📌 Mục tiêu
- Phát hiện các hành vi gian lận phổ biến trong thi trực tuyến
- Giám sát hướng nhìn, tư thế đầu và các đối tượng khả nghi
- Hỗ trợ giám thị trong việc đánh giá và phát hiện bất thường

---

## 🧠 Các mô hình sử dụng
- **YOLO (YOLOv8 / YOLOv11)**: phát hiện đối tượng (điện thoại, người khác, tài liệu, ...)
- **L2CS-Net**: ước lượng hướng nhìn (Gaze Estimation)
- **SixDRepNet**: ước lượng tư thế đầu (Head Pose Estimation)
- **MediaPipe FaceMesh**: trích xuất landmark khuôn mặt
- **Rule-based Engine**: phân tích hành vi gian lận

---

## ⚙️ Pipeline hệ thống
1. Nhận video/webcam từ thí sinh
2. Phát hiện đối tượng bằng YOLO
3. Căn chỉnh và crop khuôn mặt
4. Ước lượng:
   - Hướng nhìn (L2CS-Net)
   - Tư thế đầu (SixDRepNet)
5. Phân tích hành vi bằng luật (rules)
6. Hiển thị kết quả và cảnh báo

---

## 📂 Cấu trúc thư mục
