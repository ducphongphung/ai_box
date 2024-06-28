# Ứng dụng Camera giám sát gia đình

## 1. Động lực phát triển dự án

Tại Việt Nam, tình trạng cháy nổ đang diễn ra ngày càng phổ biến, gây ra những thiệt hại nghiêm trọng về người và tài sản. Bên cạnh đó, các vấn đề như người lạ đột nhập hay tai nạn ngã trong gia đình cũng đang được quan tâm nhiều. Nhằm giải quyết những thách thức này, chúng tôi đã phát triển một ứng dụng có khả năng chạy trực tiếp trên thiết bị biên kết nối với camera. Ứng dụng này có thể phát hiện các trường hợp đặc biệt như khói cháy, người lạ và người ngã, từ đó kịp thời cảnh báo cho các thành viên trong gia đình.

## 2. Yêu cầu phần cứng

- Camera HIKVISION
- Aibox6490

## 3. Nguyên lý hoạt động

Chúng tôi đã tiến hành đào tạo lại mô hình phát hiện đối tượng YOLO - một trong những mô hình phổ biến hiện nay, với các bộ dữ liệu về khói/cháy và người ngã. Đồng thời, chúng tôi cũng sử dụng các mô hình OpenCV-ResNet và GhostFaceNet trong việc phát hiện người lạ. Tất cả các mô hình này đều được chuyển đổi sang định dạng DLC để đảm bảo tính tương thích với thiết bị phần cứng.

## 4. Tính năng chính

Ứng dụng của chúng tôi bao gồm các chức năng cơ bản sau:
- Đăng nhập và đăng ký tài khoản
- Bật/tắt các chức năng theo dõi
- Tùy chỉnh vùng quét theo nhu cầu

## 5. Hạn chế và hướng phát triển

Hiện tại, độ chính xác của các mô hình vẫn chưa đạt được như mong đợi do những yêu cầu về dung lượng. Mặc dù đã thử nghiệm thành công 3 chức năng trên môi trường server, trên thiết bị biên chúng tôi mới chỉ hoàn thiện 2 chức năng là phát hiện khói cháy và người ngã.

Trong tương lai, chúng tôi sẽ tập trung vào:
- Cải thiện độ chính xác của các mô hình
- Triển khai cả 3 chức năng trên thiết bị biên

## 6. Hướng dẫn cài đặt và chạy ứng dụng

```bash
git clone https://github.com/ducphongphung/ai_box.git
cd ai_box
pip install -r requirements.txt
python3 src/app_core/infer_service_fall.py
python3 src/app_core/main.py