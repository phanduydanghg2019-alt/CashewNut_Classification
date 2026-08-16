# 🥜 Phân loại chất lượng Hạt Điều bằng mạng Custom CNN (MATLAB)

Dự án này xây dựng một mô hình Mạng nơ-ron tích chập (CNN) từ đầu bằng ngôn ngữ MATLAB để tự động phân loại chất lượng hạt điều dựa trên hình ảnh. Dự án ứng dụng các hàm huấn luyện học sâu hiện đại nhất của MATLAB như `trainnet` và `minibatchpredict`.

## 📑 Mục lục
- [Giới thiệu](#-giới-thiệu)
- [Bộ dữ liệu](#-bộ-dữ-liệu)
- [Kiến trúc mô hình](#-kiến-trúc-mô-hình)
- [Kết quả huấn luyện](#-kết-quả-huấn-luyện)
- [Yêu cầu hệ thống](#-yêu-cầu-hệ-thống)
- [Hướng dẫn sử dụng](#-hướng-dẫn-sử-dụng)
- [Tác giả](#-tác-giả)

---

## 🌟 Giới thiệu
Phân loại hạt điều thủ công đòi hỏi nhiều nhân công và dễ xảy ra sai sót. Dự án này cung cấp một giải pháp thị giác máy tính tự động hóa, nhẹ gọn, có khả năng phân biệt 4 loại hạt điều thương mại cơ bản với độ chính xác cao mà không cần phụ thuộc vào các mạng Transfer Learning cồng kềnh.

## 📊 Bộ dữ liệu
Bộ dữ liệu gồm khoảng **3.000 hình ảnh** hạt điều thực tế, được chia thành 4 phân lớp chất lượng:
*   **Wholes:** Hạt nguyên
*   **Butts:** Hạt vỡ dọc
*   **Split:** Hạt vỡ đôi
*   **Pieces:** Mảnh vỡ

**Tiền xử lý & Phân chia:**
*   Dữ liệu được chia theo tỷ lệ: **70% Train - 20% Validation - 10% Test**.
*   Ảnh đầu vào được chuẩn hóa về kích thước `224x224x3`.
*   Tích hợp kỹ thuật tăng cường dữ liệu (Data Augmentation): Xoay ngẫu nhiên (±15 độ) và lật ảnh (trục X, Y) để chống quá khớp (overfitting).

## 🧠 Kiến trúc mô hình
Mô hình Custom CNN bao gồm 4 khối trích xuất đặc trưng nối tiếp nhau:
1.  **Lớp Input:** `224x224x3`, chuẩn hóa `zerocenter`.
2.  **4 Khối Convolutional:** Mỗi khối gồm `Conv2D` (16, 32, 64, 128 filters) + `BatchNorm` + `ReLU` + `MaxPooling2D`.
3.  **Lớp Output:** `FullyConnected` + `Softmax` tính xác suất chéo (`crossentropy`).

*Thuật toán tối ưu:* **Adam** (Learning Rate: 0.001, Batch Size: 16, Epochs: 25).

## 📈 Kết quả huấn luyện
*   **Độ chính xác (Accuracy):** Mô hình đạt độ chính xác tổng thể **91%** trên tập kiểm tra.
*   **Điểm mạnh:** Nhận diện cực kỳ xuất sắc các loại hạt hình dáng rõ rệt như **Butts (96,4%)** và **Wholes (96,1%)**.
*   Ma trận nhầm lẫn (Confusion Matrix) chi tiết được lưu tại thư mục `Ket_Qua_Do_An_Hat_Dieu3` sau khi chạy code.

## 💻 Yêu cầu hệ thống
Để chạy được mã nguồn này, máy tính của bạn cần cài đặt:
*   **MATLAB** (Phiên bản R2024a hoặc mới hơn được khuyến nghị để hỗ trợ hàm `trainnet`).
*   Add-on: **Deep Learning Toolbox**.
*   Phần cứng: Khuyến nghị sử dụng GPU có ít nhất 4GB VRAM (ví dụ: NVIDIA GTX 1650) để tăng tốc độ huấn luyện.

## 🚀 Hướng dẫn sử dụng

**Bước 1: Tải mã nguồn về máy (Clone Repository)**  
Mở Terminal (hoặc Command Prompt/Git Bash) và chạy dòng lệnh sau để tải toàn bộ dự án về máy:

```bash
git clone https://github.com/phanduydanghg2019-alt/CashewNut_Classification.git


**Bước 2: Chuẩn bị và tổ chức bộ dữ liệu**
Mặc định trong code, biến dataRoot đang trỏ tới thư mục D:\Tai_Xuong\Dataset_V1. Bạn có thể tạo đúng đường dẫn này trên máy, hoặc chủ động đổi lại đường dẫn trong file code để phù hợp với vị trí lưu data của bạn.

Hãy đảm bảo dữ liệu hình ảnh được sắp xếp theo đúng cấu trúc cây thư mục sau để hàm imageDatastore đọc đúng nhãn:

Dataset_V1/
├── train/
│   ├── Butts/
│   ├── Pieces/
│   ├── Split/
│   └── Wholes/
└── valid/
    ├── Butts/
    ├── Pieces/
    ├── Split/
    └── Wholes/
**Bước 3: Khởi chạy quy trình huấn luyện**

Mở phần mềm MATLAB.

Mở file script huấn luyện chính (file .m) của dự án.

Nhấn nút Run (hoặc phím F5) trên thanh công cụ để bắt đầu quá trình học.

Hệ thống sẽ tự động chạy, xuất các thông số ra Command Window. Khi kết thúc, mô hình tốt nhất (file .mat) và ảnh chụp Ma trận nhầm lẫn (Confusion Matrix) sẽ tự động được lưu vào thư mục xuất kết quả (exportFolder).
