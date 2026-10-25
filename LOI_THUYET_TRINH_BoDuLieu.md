# Lời thuyết trình — Bộ dữ liệu OmniFall (13 slide)

Tổng thời gian ước tính: **12:05** (nói chậm rãi, không đọc nguyên văn). Cùng nội dung này đã gắn vào phần Ghi chú (Notes) của từng slide trong file .pptx.

## Slide 1 — Bìa  (0:20, cộng dồn 0:20)

Xin chào mọi người. Hôm nay mình trình bày về bộ dữ liệu của dự án phát hiện té ngã: gồm những bộ nào, mình đã xử lý ra sao, và kết quả thu được là gì. Mục tiêu của dự án là phân biệt hai trạng thái nhìn rất giống nhau: người vừa ngã cần cấp cứu, gọi là fallen, và người đang nằm nghỉ bình thường, gọi là lying.

## Slide 2 — OmniFall là gì  (1:00, cộng dồn 1:20)

Dữ liệu nền của dự án là OmniFall. Các bộ dữ liệu té ngã trước đây thường quay trong môi trường đơn giản, và mỗi bộ gán nhãn một kiểu riêng nên không so sánh được với nhau. OmniFall gộp chúng lại: khoảng 80 giờ video, khoảng 15 nghìn clip, và một bộ 16 lớp hành động thống nhất. Dữ liệu chia làm ba miền. Thứ nhất là Staged, gồm 8 bộ công khai quay có dàn dựng. Thứ hai là Synthetic, 12 nghìn video tổng hợp, thay đổi tuổi, thể hình, môi trường và góc camera. Thứ ba là In-the-Wild, chỉ dùng để kiểm thử, lấy từ video tai nạn có thật. Bài báo của nhóm Schneider và cộng sự trên arXiv, và trang của tác giả ghi bài được nhận ở ECCV 2026 dưới dạng poster.

## Slide 3 — Taxonomy 16 lớp, dùng 10 lớp  (0:50, cộng dồn 2:10)

Mỗi đoạn video, mình gọi là một segment, có một nhãn hành động. Dự án dùng 10 lớp lõi: walk, fall, fallen, sit_down, sitting, lie_down, lying, stand_up, standing và other. Hai màu vàng là hành động chuyển tiếp: fall là đang ngã, lie_down là đang nằm xuống. Fallen và lying là trạng thái sau đó. Sáu lớp mở rộng như kneeling, squatting, crawl, jump được gộp vào other, vì các bộ staged chỉ có 10 lớp lõi. Làm vậy để mọi bộ dùng chung một không gian nhãn và so sánh công bằng với baseline chính thức.

## Slide 4 — 5 bộ dữ liệu đã dùng  (1:30, cộng dồn 3:40)

Dự án dùng 5 bộ. Ba bộ staged là GMDCSA24 với 160 video, Le2i với 190 video và CAUCAFall với 100 video. Một bộ tổng hợp là OF-Synthetic với 12 nghìn video. Và một bộ ngoài đời thật là OOPS, thuộc OF-ItW, với 818 video. Bảng này đếm trên toàn bộ từng bộ, kể cả phần không dùng. Có một điểm cần chú ý: lying và lie_down rất hiếm ở các bộ thật. Ở OOPS chỉ có 20 segment lying và 16 segment lie_down, ở các bộ staged còn ít hơn. Gần như toàn bộ nhãn lying đến từ dữ liệu tổng hợp. Điều này ảnh hưởng đến cách diễn giải kết quả ở phần sau.

## Slide 5 — Chia train, val, test  (1:00, cộng dồn 4:40)

Mình dùng cách chia cross-subject của OmniFall. Tập train có 9.877 video, tương ứng 16.403 segment, gồm OF-Synthetic và ba bộ staged. Tập validation có 1.276 video, thêm 81 video OOPS để chọn checkpoint tốt nhất. Tập test chỉ gồm OF-ItW, 572 video và 2.804 segment, tách riêng hoàn toàn, mô hình chưa từng thấy lúc huấn luyện. Đây là đúng toàn bộ tập test của OF-ItW, mình không lọc bớt video nào.

## Slide 6 — Bộ dữ liệu không dùng  (0:50, cộng dồn 5:30)

Có 5 bộ mình đã thử nhưng không đưa vào kết quả. MCFD tải được và đã tích hợp, nhưng mình loại vì làm kết quả tệ đi, mình sẽ nói ở slide 10. CMDFall cần xin quyền tác giả qua email và nặng khoảng 350 gigabyte, không khả thi. UP-Fall bị chặn tải tự động, cần đăng nhập thủ công. EDF và OCCU mình không tìm được nguồn tải dùng được. Kết quả là chỉ có 3 trong 8 bộ staged được dùng để huấn luyện.

## Slide 7 — Pipeline xử lý  (1:00, cộng dồn 6:30)

Quy trình xử lý gồm 6 bước. Một, ghép đường dẫn video với file nhãn. Hai, lấy mẫu lại video xuống 7,5 khung hình mỗi giây. Ba, trượt một cửa sổ 16 khung hình, bước trượt khoảng một khung. Bốn, đưa mỗi cửa sổ qua VideoMAE-K400 để ra một vector 768 chiều. Năm, lưu mỗi video thành một file h5. Sáu, đưa 18 vector liên tiếp vào mô hình baseline FeatureTransformer. Mình giữ đúng cấu hình của paper gốc, Table 12, và dùng baseline chính thức của repo omnifall-experiments thay vì tự viết lại. Logits sau đó được lưu lại để bước tiếp theo dùng mà không phải chạy lại video.

## Slide 8 — Xử lý đặc thù từng bộ  (1:00, cộng dồn 7:30)

Mỗi bộ có một kiểu đặt tên và dung lượng riêng nên cần xử lý riêng. Tên video và nhãn không khớp: GMDCSA24 có dấu cách trong tên thư mục còn nhãn dùng gạch dưới, Le2i đặt tên có ngoặc đơn, CAUCAFall thì phải khớp theo tên file. OF-Synthetic nằm trong một file tar khoảng 9,7 gigabyte, mình đọc trực tiếp từng video từ tar để không phải giải nén hết, vì ổ đĩa không đủ chỗ. OOPS có archive gốc khoảng 45 gigabyte, mình tải dạng stream và chỉ giữ 818 video cần dùng, khoảng 2,6 gigabyte. Mình chỉ xoá video gốc sau khi đếm đủ số file đặc trưng, ví dụ 12.000 trên 12.000.

## Slide 9 — Phân bố tập test  (1:00, cộng dồn 8:30)

Đây là phân bố nhãn trong tập test. Lớp other nhiều nhất với 902 segment, tiếp theo là fall với 636 và fallen với 318. Nhưng lying chỉ có 17 và lie_down chỉ có 11, tổng cộng 28 segment. Vì vậy kết luận của dự án chủ yếu dựa trên cặp fall và fallen. Việc phân biệt fallen với lying, vốn là động lực ban đầu, thì chưa kiểm chứng được trên tập test này. Mình muốn nói rõ điều đó ngay từ đây.

## Slide 10 — Thêm dữ liệu không phải lúc nào cũng tốt  (1:00, cộng dồn 9:30)

Đây là một kết quả về dữ liệu mà mình thấy đáng chú ý. Khi huấn luyện baseline trên ba bộ staged cộng OF-Synthetic, Fall∪Fallen F1 là 55,5 phần trăm. Khi mình thêm MCFD, cả ba chỉ số đều giảm, Fall∪Fallen còn 52,8. Dùng thêm mô hình học có ngữ cảnh thời gian trên bản có MCFD cũng không cứu được, chỉ đạt 51,7. Mình đoán nguyên nhân là MCFD chỉ có một người thực hiện và quay bằng camera Kinect đời cũ, nên mô hình học đặc điểm riêng của người và camera đó. Đây mới là giả thuyết, mình chưa kiểm chứng riêng. Nhưng nó gợi ý rằng đa dạng người thực hiện quan trọng hơn số lượng video.

## Slide 11 — Kết quả mô hình trên tập test  (1:15, cộng dồn 10:45)

Với dữ liệu như vậy, đây là kết quả trên tập test. Baseline đạt Fallen-F1 là 26,6 phần trăm. Rule-based, một bước hậu xử lý theo thời gian, đạt 30,0. Hybrid v8, bản học được, đạt 34,3, trung bình qua ba seed. Tức là Fallen-F1 tăng khoảng 7,7 điểm so với baseline. Chỉ số gộp Fall∪Fallen tăng nhỏ hơn, khoảng 1,6 điểm, và Fall-F1 giảm nhẹ 0,7 điểm. Con số tuyệt đối vẫn thấp, vì dữ liệu ngoài đời rất khó. Bên phải là ma trận nhầm lẫn bốn lớp cho thấy lỗi nằm ở đâu. Về thống kê, cải thiện có ý nghĩa ở hai trong ba seed khi đo bằng bootstrap trên F1.

## Slide 12 — Hạn chế của dữ liệu  (0:50, cộng dồn 11:35)

Cuối cùng là những hạn chế. Thứ nhất, lying quá ít, nên chưa đánh giá được phân biệt fallen và lying. Thứ hai, 97 phần trăm video train là dữ liệu tổng hợp, chỉ 277 video staged thật với ít người thực hiện và ít góc camera. Thứ ba, tập test lấy từ video tai nạn trên mạng, chưa thử với camera giám sát trong nhà thật. Thứ tư, chỉ có 3 trong 8 bộ staged, nên chưa biết kết quả đổi thế nào nếu có đủ dữ liệu. Mình nêu những điều này để người nghe đánh giá đúng mức độ tin cậy của kết quả.

## Slide 13 — Liên kết nguồn  (0:30, cộng dồn 12:05)

Đây là các liên kết nguồn: bài báo OmniFall trên arXiv, trang dự án, dataset trên Hugging Face, mã baseline chính thức, và nguồn từng bộ dữ liệu. Mình xin kết thúc phần trình bày ở đây. Cảm ơn mọi người đã lắng nghe, mình sẵn sàng trả lời câu hỏi.

## Câu hỏi có thể gặp và cách trả lời

**Vì sao chỉ dùng 3 trong 8 bộ staged?** CMDFall cần xin tác giả và nặng khoảng 350GB, UP-Fall bị chặn tải tự động, EDF và OCCU không tìm được nguồn dùng được. MCFD tải được nhưng bị loại vì làm kết quả tệ đi. Đây là hạn chế đã nêu ở slide 12.

**Vì sao gộp 16 lớp về 10 lớp?** Các bộ staged chỉ có 10 lớp lõi, nên gộp 6 lớp mở rộng vào other để mọi bộ dùng chung một không gian nhãn và so sánh công bằng với baseline chính thức.

**Tập test có bị lộ khi huấn luyện không?** Tập test chỉ gồm 572 video OF-ItW theo split chính thức, tách riêng khỏi train. Validation có thêm 81 video OOPS khác (không thuộc test) để chọn checkpoint.

**Vì sao thêm MCFD lại làm kết quả tệ đi?** Mình chưa kiểm chứng. Giả thuyết là MCFD chỉ có 1 người thực hiện và camera Kinect đời cũ, nên mô hình học đặc điểm riêng của người và camera đó. Muốn kiểm chứng cần thử loại bớt từng yếu tố.

**Kết quả vẫn thấp (Fallen-F1 khoảng 34%), có dùng được không?** Đây là dữ liệu ngoài đời rất khó, và con số tuyệt đối thấp là thật. Ý nghĩa của kết quả là xu hướng cải thiện so với baseline, không phải mức dùng được cho chăm sóc người già thật.

**Phân biệt fallen và lying đã được chứng minh chưa?** Chưa. Tập test chỉ có 17 lying và 11 lie_down. Cải thiện chủ yếu nằm ở cặp fall và fallen.

**Dữ liệu tổng hợp có làm mô hình lệch không?** Có khả năng: 97% video train là OF-Synthetic. Mình chưa có thí nghiệm riêng để tách tác động này.

**Bài báo OmniFall được nhận ở đâu?** Trang của tác giả ghi ECCV 2026 (poster). Trang arXiv chưa ghi tên hội nghị, nên nên kiểm tra thêm nếu cần trích dẫn chính thức.
