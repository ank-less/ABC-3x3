# TrackID3x3 Challenge (ABC 2027): code + paper

Bài làm cho **ABC2027 TrackID3x3 Challenge**: với mỗi frame của video bóng rổ 3x3, cho biết
**vị trí trên sân (cm)** và **danh tính** (đội A/B + số áo) của 6 cầu thủ. Chấm bằng TI-HOTA.

Trên đề thi thử (video Training IMG_0108 + IMG_0109, không dùng để train, nén lại cho giống video
Challenge), phương pháp đạt **TI-HOTA 0.968**.

## Trong thư mục có gì

```
TrackID3x3-ABC2027/
├── DOC_TRUOC.md          ← file này
├── code/                 ← bộ code hoàn chỉnh (đúng bản đã tạo file nộp)
│   ├── README.md         hướng dẫn chi tiết (tiếng Anh)
│   ├── run_challenge.sh  1 lệnh: video Challenge → 2 file nộp + tự kiểm tra định dạng
│   ├── run_val.sh        chấm điểm trên đề thi thử 0108/0109
│   ├── requirements.txt  các thư viện Python (Python 3.13)
│   ├── tools/            toàn bộ mã nguồn (tracking, nhận diện áo, chiếu xuống sân, chấm điểm, train)
│   ├── weights/          2 model đã train + SHA256SUMS để kiểm tra file
│   ├── predictions/      2 file dự đoán nộp chính thức (IMG_0114.txt, IMG_0115.txt)
│   └── data/challenge/   roster + homography của BTC; chỗ để bỏ 2 video Challenge
└── paper/
    ├── main.pdf          bản thảo paper (tiếng Anh, 10 trang)
    ├── main.tex          mã LaTeX của paper
    ├── figures/          3 hình đã tạo sẵn (hình sơ đồ quy trình vẽ thẳng trong main.tex)
    └── make_figures.py   script tạo hình (chỉ chạy được trong thư mục dự án gốc)
```

## Các bước xử lý (tóm tắt)

1. **YOLO11l** (fine-tune 3 vòng) tìm người, **BoT-SORT** nối khung qua các frame.
2. **ResNet18** đọc màu áo + số áo của từng khung (13 lớp). Mỗi đội luôn mặc một màu áo cố định,
   nên "đội + số áo" gộp thành một bài phân loại.
3. Trung bình xác suất 31 frame dọc mỗi track, rồi ghép Hungarian từng frame
   (mỗi cầu thủ tối đa 1 khung).
4. Chiếu **điểm giữa cạnh dưới khung** (điểm chân) xuống sân bằng homography của BTC.
5. Lọc điểm sai (lệch xa trung vị, đoạn ngắn phải "chạy > 12 m/s" mới tới), nội suy chỗ trống.
   Kết quả luôn đủ 6 dòng mỗi frame.

## Chạy lại

```bash
cd code
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
# chép IMG_0114.MP4 và IMG_0115.MP4 (video Challenge của BTC) vào data/challenge/videos/
./run_challenge.sh
```

+ Mất khoảng 25 phút trên GPU RTX 4050 Laptop. Kết quả ghi vào `code/predictions/`.
+ Trên máy gốc, kết quả **giống hệt từng byte** với file đã nộp. GPU khác có thể làm lệch rất nhẹ vài khung.
+ Muốn chấm trên đề thi thử: cần video Training của BTC, xem mục 3 trong `code/README.md`.
+ Kiểm tra model không bị hỏng khi tải về: `cd code/weights && sha256sum -c SHA256SUMS`.

Dịch lại paper ra PDF (cần [tectonic](https://tectonic-typesetting.github.io)):

```bash
cd paper
tectonic main.tex
```

## Lưu ý

+ **Luật BTC:** chỉ train trên tập Training. Video Challenge chỉ được chạy dự đoán: không train,
  không tự gán nhãn, không sửa tay từng frame.
+ **Đóng băng phiên bản:** code, model và cấu hình nộp cùng paper phải **đúng bản** đã tạo
  file dự đoán. Nếu muốn thử sửa gì, hãy làm trên một bản sao khác, đừng sửa trực tiếp thư mục này.
+ Paper chưa điền tên nhóm và tác giả, và chưa chuyển sang mẫu IJABC chính thức của hội thảo.
