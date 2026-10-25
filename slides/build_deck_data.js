const pptxgen = require("pptxgenjs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");
const FIG_CONFUSION = path.join(ROOT, "figures", "confusion_matrix_4class.png");

const NAVY = "1B2A4A", NAVY2 = "24365C", ICE = "E8EDF7", WHITE = "FFFFFF";
const TEAL = "2E86AB", RED = "C8213B", GREEN = "2F9E44", AMBER = "D98E04";
const GRAYTEXT = "44506B", LIGHTGRAYTEXT = "AEB9D4";
const FONT_HEAD = "Cambria", FONT_BODY = "Calibri";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
const SCRIPT = require("./script_data.js");
const _slides = [];
const _origAddSlide = pres.addSlide.bind(pres);
pres.addSlide = (...a) => { const sl = _origAddSlide(...a); _slides.push(sl); return sl; };
const PAGE_W = 13.33, PAGE_H = 7.5;

const darkSlide = () => { const s = pres.addSlide(); s.background = { color: NAVY }; return s; };
const lightSlide = () => { const s = pres.addSlide(); s.background = { color: WHITE }; return s; };
function kicker(s, t) {
  s.addText(t.toUpperCase(), { x: 0.6, y: 0.42, w: 10, h: 0.35, fontFace: FONT_BODY, fontSize: 13, bold: true, color: TEAL, charSpacing: 2, isTextBox: true, margin: 0 });
}
function title(s, t, onDark = false, size = 30) {
  s.addText(t, { x: 0.6, y: 0.72, w: 12.1, h: 0.9, fontFace: FONT_HEAD, fontSize: size, bold: true, color: onDark ? WHITE : NAVY, isTextBox: true, margin: 0 });
}
function pageNum(s, n, onDark = false) {
  s.addText(String(n), { x: PAGE_W - 0.7, y: PAGE_H - 0.45, w: 0.4, h: 0.3, fontFace: FONT_BODY, fontSize: 10, color: onDark ? LIGHTGRAYTEXT : "A0A8BD", align: "right", isTextBox: true, margin: 0 });
}
function bullets(s, items, x, y, w, h, size = 13.5, color = GRAYTEXT, gap = 8) {
  s.addText(items.map((t, i) => ({ text: t, options: { bullet: { code: "2022" }, breakLine: i < items.length - 1, fontSize: size, color } })),
    { x, y, w, h, fontFace: FONT_BODY, paraSpaceAfter: gap, isTextBox: true, margin: 0 });
}

// ===== 1. Bìa =====
{
  const s = darkSlide();
  s.addShape("ellipse", { x: 9.6, y: -2.2, w: 6.5, h: 6.5, fill: { color: NAVY2 }, line: { type: "none" } });
  s.addShape("ellipse", { x: 11.3, y: 4.6, w: 3.6, h: 3.6, fill: { color: NAVY2 }, line: { type: "none" } });
  s.addText("BỘ DỮ LIỆU CỦA DỰ ÁN", { x: 0.8, y: 1.6, w: 10, h: 0.4, fontFace: FONT_BODY, fontSize: 14, bold: true, color: TEAL, charSpacing: 2, isTextBox: true, margin: 0 });
  s.addText("OmniFall: gồm những bộ nào,\nxử lý ra sao, cho kết quả gì", { x: 0.75, y: 2.1, w: 11.5, h: 1.9, fontFace: FONT_HEAD, fontSize: 40, bold: true, color: WHITE, isTextBox: true, margin: 0 });
  s.addText("Phát hiện té ngã: phân biệt “fallen” (cần cấp cứu) và “lying” (nằm nghỉ)", { x: 0.8, y: 4.2, w: 11, h: 0.5, fontFace: FONT_BODY, fontSize: 18, color: "CADCFC", isTextBox: true, margin: 0 });
  s.addText("Dữ liệu → Xử lý → Kết quả → Hạn chế → Liên kết nguồn", { x: 0.8, y: 6.55, w: 11, h: 0.4, fontFace: FONT_BODY, fontSize: 12, italic: true, color: LIGHTGRAYTEXT, isTextBox: true, margin: 0 });
}

// ===== 2. OmniFall là gì =====
{
  const s = lightSlide();
  kicker(s, "Tổng quan");
  title(s, "OmniFall: benchmark té ngã gộp từ nhiều nguồn");
  s.addText("Mục tiêu của OmniFall: các bộ dữ liệu té ngã cũ quay trong môi trường đơn giản, nhãn mỗi bộ một kiểu nên không so sánh được với nhau. OmniFall gộp chúng lại, thống nhất nhãn và bổ sung dữ liệu tổng hợp lẫn video thật.", {
    x: 0.6, y: 1.65, w: 12.1, h: 0.9, fontFace: FONT_BODY, fontSize: 14.5, color: GRAYTEXT, isTextBox: true, margin: 0 });

  const stats = [["~80 giờ", "video, khoảng 15.000 clip"], ["16 lớp", "taxonomy hành động thống nhất"], ["3 miền", "Staged · Synthetic · In-the-Wild"]];
  stats.forEach(([big, small], i) => {
    const x = 0.6 + i * 4.1;
    s.addShape("roundRect", { x, y: 2.8, w: 3.8, h: 1.55, rectRadius: 0.1, fill: { color: ICE }, line: { type: "none" } });
    s.addText(big, { x: x + 0.3, y: 2.95, w: 3.2, h: 0.7, fontFace: FONT_HEAD, bold: true, fontSize: 30, color: TEAL, isTextBox: true, margin: 0 });
    s.addText(small, { x: x + 0.3, y: 3.65, w: 3.3, h: 0.55, fontFace: FONT_BODY, fontSize: 12.5, color: GRAYTEXT, isTextBox: true, margin: 0 });
  });

  const dom = [
    ["OF-Staged", "8 bộ dữ liệu công khai, quay có dàn dựng, chuẩn hoá chia train/val/test", TEAL],
    ["OF-Synthetic", "12.000 video tổng hợp (~17 giờ), thay đổi tuổi, thể hình, môi trường, góc camera", AMBER],
    ["OF-In-the-Wild", "bộ chỉ để test, lấy từ video tai nạn có thật (nguồn OOPS)", RED],
  ];
  dom.forEach(([h, b, c], i) => {
    const y = 4.65 + i * 0.72;
    s.addShape("ellipse", { x: 0.65, y: y + 0.12, w: 0.22, h: 0.22, fill: { color: c }, line: { type: "none" } });
    s.addText(h, { x: 1.05, y, w: 2.4, h: 0.45, fontFace: FONT_HEAD, bold: true, fontSize: 14, color: NAVY, isTextBox: true, margin: 0 });
    s.addText(b, { x: 3.5, y, w: 9.2, h: 0.5, fontFace: FONT_BODY, fontSize: 12.5, color: GRAYTEXT, isTextBox: true, margin: 0 });
  });
  s.addText("Bài báo: Schneider et al., arXiv:2505.19889. Trang tác giả ghi bài được nhận tại ECCV 2026 (poster).", {
    x: 0.6, y: 6.85, w: 12, h: 0.35, italic: true, fontFace: FONT_BODY, fontSize: 11, color: "8791A8", isTextBox: true, margin: 0 });
  pageNum(s, 2);
}

// ===== 3. Taxonomy =====
{
  const s = lightSlide();
  kicker(s, "Nhãn");
  title(s, "Taxonomy 16 lớp, dự án dùng 10 lớp lõi");
  s.addText("Mỗi đoạn video (segment) có 1 nhãn hành động. 10 lớp lõi dùng cho cả huấn luyện và đánh giá:", { x: 0.6, y: 1.65, w: 12, h: 0.45, fontFace: FONT_BODY, fontSize: 14, color: GRAYTEXT, isTextBox: true, margin: 0 });

  const core = [["walk", ICE, NAVY], ["fall", "FFF1D6", AMBER], ["fallen", "FCEEF0", RED], ["sit_down", ICE, NAVY], ["sitting", ICE, NAVY], ["lie_down", "FFF1D6", AMBER], ["lying", "E9F7EC", GREEN], ["stand_up", ICE, NAVY], ["standing", ICE, NAVY], ["other", ICE, NAVY]];
  core.forEach(([n, bg, fg], i) => {
    const x = 0.6 + (i % 5) * 2.45, y = 2.3 + Math.floor(i / 5) * 0.95;
    s.addShape("roundRect", { x, y, w: 2.25, h: 0.75, rectRadius: 0.1, fill: { color: bg }, line: { color: fg, width: 1 } });
    s.addText(n, { x, y, w: 2.25, h: 0.75, align: "center", valign: "middle", fontFace: FONT_BODY, bold: true, fontSize: 15, color: fg, isTextBox: true, margin: 0 });
  });
  s.addText("Hành động chuyển tiếp (fall, lie_down) màu vàng; trạng thái cần chú ý (fallen, lying) màu đỏ/xanh.", { x: 0.6, y: 4.25, w: 12, h: 0.4, italic: true, fontFace: FONT_BODY, fontSize: 11.5, color: "8791A8", isTextBox: true, margin: 0 });

  s.addShape("roundRect", { x: 0.6, y: 4.95, w: 12.1, h: 1.7, rectRadius: 0.1, fill: { color: NAVY }, line: { type: "none" } });
  s.addText([
    { text: "6 lớp mở rộng gộp vào “other”:  ", options: { bold: true, color: TEAL } },
    { text: "kneel_down, kneeling, squat_down, squatting, crawl, jump. Các bộ staged chỉ có 10 lớp lõi, nên gộp để mọi bộ dùng chung một không gian nhãn và so sánh công bằng với baseline chính thức.", options: { color: "E4E9F5" } },
  ], { x: 0.95, y: 4.95, w: 11.4, h: 1.7, valign: "middle", fontFace: FONT_BODY, fontSize: 14, isTextBox: true, margin: 0 });
  pageNum(s, 3);
}

// ===== 4. Các bộ đã dùng =====
{
  const s = lightSlide();
  kicker(s, "Dữ liệu đã dùng");
  title(s, "5 bộ dữ liệu: 3 staged, 1 tổng hợp, 1 ngoài đời thật");
  const H = (t, al = "center") => ({ text: t, options: { bold: true, color: WHITE, fill: { color: NAVY }, align: al } });
  const C = (t, fill, al = "center", bold = false) => ({ text: t, options: { color: GRAYTEXT, fill: { color: fill }, align: al, bold } });
  const rows = [
    [H("Bộ dữ liệu", "left"), H("Loại"), H("Video"), H("Segment"), H("fall"), H("fallen"), H("lie_down"), H("lying"), H("Vai trò", "left")],
    [C("OF-Synthetic", WHITE, "left", true), C("Tổng hợp", WHITE), C("12.000", WHITE), C("19.228", WHITE), C("2.148", WHITE), C("1.449", WHITE), C("1.259", WHITE), C("2.344", WHITE), C("Train + val", WHITE, "left")],
    [C("GMDCSA24", ICE, "left", true), C("Staged", ICE), C("160", ICE), C("458", ICE), C("78", ICE), C("77", ICE), C("13", ICE), C("14", ICE), C("Train + val", ICE, "left")],
    [C("Le2i", WHITE, "left", true), C("Staged", WHITE), C("190", WHITE), C("967", WHITE), C("130", WHITE), C("123", WHITE), C("1", WHITE), C("0", WHITE), C("Train + val", WHITE, "left")],
    [C("CAUCAFall", ICE, "left", true), C("Staged", ICE), C("100", ICE), C("258", ICE), C("50", ICE), C("48", ICE), C("1", ICE), C("1", ICE), C("Train + val", ICE, "left")],
    [C("OOPS (OF-ItW)", "FCEEF0", "left", true), C("Ngoài đời", "FCEEF0"), C("818", "FCEEF0"), C("4.022", "FCEEF0"), C("914", "FCEEF0"), C("464", "FCEEF0"), C("16", "FCEEF0"), C("20", "FCEEF0"), C("Val + TEST", "FCEEF0", "left", true)],
  ];
  s.addTable(rows, { x: 0.6, y: 1.8, w: 12.1, h: 3.3, fontFace: FONT_BODY, fontSize: 13, border: { type: "none" }, autoPage: false,
    colW: [2.2, 1.35, 1.0, 1.15, 0.95, 1.0, 1.1, 0.95, 2.4], rowH: 0.55, valign: "middle" });
  s.addText("Số liệu đếm trên toàn bộ từng bộ (file nhãn), gồm cả phần không dùng. Segment = đoạn video có 1 nhãn hành động.", { x: 0.6, y: 5.2, w: 12, h: 0.35, italic: true, fontFace: FONT_BODY, fontSize: 11, color: "8791A8", isTextBox: true, margin: 0 });

  s.addShape("roundRect", { x: 0.6, y: 5.75, w: 12.1, h: 1.0, rectRadius: 0.1, fill: { color: ICE }, line: { type: "none" } });
  s.addText([
    { text: "Nhận xét:  ", options: { bold: true, color: TEAL } },
    { text: "lying và lie_down rất hiếm ở các bộ thật (staged và OOPS chỉ có vài chục segment). Phần lớn nhãn lying/lie_down đến từ dữ liệu tổng hợp OF-Synthetic.", options: { color: GRAYTEXT } },
  ], { x: 0.9, y: 5.75, w: 11.5, h: 1.0, valign: "middle", fontFace: FONT_BODY, fontSize: 13, isTextBox: true, margin: 0 });
  pageNum(s, 4);
}

// ===== 5. Chia tập =====
{
  const s = lightSlide();
  kicker(s, "Chia tập");
  title(s, "Huấn luyện trên staged + tổng hợp, kiểm thử trên video thật");
  const cards = [
    ["TRAIN", "9.877 video\n16.403 segment", "OF-Synthetic (9.600 video) + GMDCSA24, Le2i, CAUCAFall (277 video)", TEAL],
    ["VAL", "1.276 video\n2.227 segment", "OF-Synthetic (1.200) + staged (76). Thêm 81 video OOPS (409 segment) để chọn checkpoint", AMBER],
    ["TEST", "572 video\n2.804 segment", "Chỉ OF-ItW (OOPS), tách riêng hoàn toàn, mô hình chưa từng thấy", RED],
  ];
  cards.forEach(([h, big, body, c], i) => {
    const x = 0.6 + i * 4.1;
    s.addShape("roundRect", { x, y: 1.85, w: 3.85, h: 3.9, rectRadius: 0.12, fill: { color: ICE }, line: { color: c, width: 1.5 } });
    s.addText(h, { x: x + 0.3, y: 2.05, w: 3.2, h: 0.4, fontFace: FONT_BODY, bold: true, fontSize: 13, color: c, charSpacing: 2, isTextBox: true, margin: 0 });
    s.addText(big, { x: x + 0.3, y: 2.5, w: 3.3, h: 1.2, fontFace: FONT_HEAD, bold: true, fontSize: 26, color: NAVY, isTextBox: true, margin: 0 });
    s.addText(body, { x: x + 0.3, y: 3.85, w: 3.3, h: 1.7, fontFace: FONT_BODY, fontSize: 12.5, color: GRAYTEXT, isTextBox: true, margin: 0 });
  });
  s.addText("Dùng split “cs” (cross-subject) của OmniFall. Test là đúng toàn bộ tập OF-ItW test (572 video, 2.804 segment), không lọc bớt.", { x: 0.6, y: 6.0, w: 12.1, h: 0.7, fontFace: FONT_BODY, fontSize: 13, color: GRAYTEXT, isTextBox: true, margin: 0 });
  pageNum(s, 5);
}

// ===== 6. Bộ không dùng được =====
{
  const s = lightSlide();
  kicker(s, "Dữ liệu không dùng");
  title(s, "5 bộ đã thử nhưng không đưa vào kết quả");
  const items = [
    ["MCFD", "Tải được và đã tích hợp (192 video), nhưng loại bỏ vì làm kết quả tệ đi trên mọi chỉ số (xem slide 10).", RED],
    ["CMDFall", "Cần xin quyền từ tác giả qua email và nặng khoảng 350GB. Không khả thi.", AMBER],
    ["UP-Fall", "IEEE DataPort chặn tải tự động, cần đăng nhập thủ công. Chưa lấy được.", AMBER],
    ["EDF, OCCU", "Không tìm được nguồn tải dùng được trong thời gian làm dự án.", AMBER],
  ];
  let y = 1.85;
  items.forEach(([h, body, c]) => {
    s.addShape("rect", { x: 0.6, y, w: 0.08, h: 1.0, fill: { color: c }, line: { type: "none" } });
    s.addText(h, { x: 0.95, y, w: 2.6, h: 1.0, valign: "top", fontFace: FONT_HEAD, bold: true, fontSize: 16, color: NAVY, isTextBox: true, margin: 0 });
    s.addText(body, { x: 3.7, y, w: 9, h: 1.0, valign: "top", fontFace: FONT_BODY, fontSize: 13.5, color: GRAYTEXT, isTextBox: true, margin: 0 });
    y += 1.2;
  });
  s.addText("Hệ quả: chỉ có 3 trong 8 bộ staged của OF-Staged được dùng để huấn luyện.", { x: 0.6, y: 6.65, w: 12, h: 0.4, italic: true, fontFace: FONT_BODY, fontSize: 13, color: "8791A8", isTextBox: true, margin: 0 });
  pageNum(s, 6);
}

// ===== 7. Pipeline xử lý =====
{
  const s = lightSlide();
  kicker(s, "Xử lý dữ liệu");
  title(s, "Từ video thô đến logits dự đoán");
  const steps = [
    ["1", "Tải & ghép nhãn", "ghép đường dẫn video với file nhãn CSV"],
    ["2", "Lấy mẫu 7,5 fps", "giảm từ 25-30 fps xuống 7,5 fps"],
    ["3", "Cửa sổ 16 khung", "trượt mỗi 2/15 giây (~1 khung)"],
    ["4", "VideoMAE-K400", "mỗi cửa sổ thành vector 768 chiều"],
    ["5", "Lưu file .h5", "mỗi video 1 file, kích thước T×768"],
    ["6", "Baseline", "18 vector liên tiếp vào FeatureTransformer"],
  ];
  const w = 1.85, gap = 0.2;
  steps.forEach(([n, h, b], i) => {
    const x = 0.6 + i * (w + gap);
    s.addShape("roundRect", { x, y: 2.0, w, h: 2.4, rectRadius: 0.1, fill: { color: i === 3 ? TEAL : ICE }, line: { type: "none" } });
    s.addText(n, { x: x + 0.15, y: 2.1, w: 0.6, h: 0.5, fontFace: FONT_HEAD, bold: true, fontSize: 22, color: i === 3 ? WHITE : TEAL, isTextBox: true, margin: 0 });
    s.addText(h, { x: x + 0.15, y: 2.7, w: w - 0.3, h: 0.65, fontFace: FONT_HEAD, bold: true, fontSize: 13.5, color: i === 3 ? WHITE : NAVY, isTextBox: true, margin: 0 });
    s.addText(b, { x: x + 0.15, y: 3.4, w: w - 0.3, h: 0.9, fontFace: FONT_BODY, fontSize: 11.5, color: i === 3 ? WHITE : GRAYTEXT, isTextBox: true, margin: 0 });
  });
  s.addShape("roundRect", { x: 0.6, y: 4.85, w: 12.1, h: 1.7, rectRadius: 0.1, fill: { color: NAVY }, line: { type: "none" } });
  s.addText([
    { text: "Giữ đúng cấu hình của paper:  ", options: { bold: true, color: TEAL } },
    { text: "7,5 fps, 16 khung hình mỗi cửa sổ, bước trượt 2/15 giây (Table 12). Baseline là FeatureTransformer 2 lớp từ repo chính thức omnifall-experiments, huấn luyện tới hội tụ. Sau đó logits được lưu lại để bước conditioning dùng mà không cần chạy lại video.", options: { color: "E4E9F5" } },
  ], { x: 0.95, y: 4.85, w: 11.4, h: 1.7, valign: "middle", fontFace: FONT_BODY, fontSize: 13.5, isTextBox: true, margin: 0 });
  pageNum(s, 7);
}

// ===== 8. Xử lý đặc thù =====
{
  const s = lightSlide();
  kicker(s, "Xử lý dữ liệu");
  title(s, "Mỗi bộ có một kiểu đặt tên và dung lượng riêng");
  const rows = [
    ["Khớp tên video ↔ nhãn", "GMDCSA24: “Subject 1/ADL/01.mp4” ↔ “Subject_1/ADL/01”. Le2i: “video (N).avi” ↔ “video_N”. CAUCAFall: khớp theo tên file vì cấu trúc thư mục khác nhãn."],
    ["OF-Synthetic", "12.000 video nằm trong 1 file tar ~9,7GB. Đọc trực tiếp từng video từ tar rồi trích đặc trưng, không giải nén toàn bộ (ổ đĩa không đủ chỗ)."],
    ["OOPS", "Archive gốc ~45GB. Tải dạng stream, chỉ giữ 818 video thuộc OF-ItW (~2,6GB), đổi tên theo quy ước của OmniFall."],
    ["Nhãn", "Gộp 16 lớp về 10 lớp lõi. Nhãn tính theo giây, không theo chỉ số khung hình."],
    ["Dọn dẹp", "Chỉ xoá video gốc sau khi đếm đủ số file .h5 (vd 12.000/12.000, 818/818) để không mất dữ liệu."],
  ];
  let y = 1.8;
  rows.forEach(([h, b]) => {
    s.addText(h, { x: 0.6, y, w: 2.9, h: 0.9, valign: "top", fontFace: FONT_HEAD, bold: true, fontSize: 14, color: NAVY, isTextBox: true, margin: 0 });
    s.addText(b, { x: 3.6, y, w: 9.1, h: 0.9, valign: "top", fontFace: FONT_BODY, fontSize: 12.5, color: GRAYTEXT, isTextBox: true, margin: 0 });
    y += 0.98;
  });
  pageNum(s, 8);
}

// ===== 9. Phân bố test (native chart) =====
{
  const s = lightSlide();
  kicker(s, "Kết quả: dữ liệu test");
  title(s, "Tập test lệch mạnh: lying và lie_down gần như không có");
  const labels = ["other", "fall", "walk", "fallen", "standing", "stand_up", "sit_down + sitting", "lying", "lie_down"];
  const values = [902, 636, 324, 318, 264, 253, 79, 17, 11];
  s.addChart(pres.charts.BAR, [{ name: "Số segment", labels, values }], {
    x: 0.6, y: 1.75, w: 7.6, h: 4.9, barDir: "bar", chartColors: [TEAL],
    showTitle: true, title: "Số segment mỗi lớp trong OF-ItW test (n = 2.804)", titleFontSize: 13, titleColor: NAVY,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 11, dataLabelColor: GRAYTEXT,
    catAxisLabelColor: GRAYTEXT, catAxisLabelFontSize: 11, catAxisOrientation: "maxMin",
    valAxisLabelColor: GRAYTEXT, valAxisLabelFontSize: 10, valGridLine: { color: "E2E6EE", size: 0.5 }, catGridLine: { style: "none" },
    showLegend: false,
  });
  const notes = [
    ["318", "segment fallen", RED],
    ["28", "segment lying + lie_down", AMBER],
  ];
  notes.forEach(([big, small, c], i) => {
    const y = 1.9 + i * 1.6;
    s.addShape("roundRect", { x: 8.6, y, w: 4.1, h: 1.4, rectRadius: 0.1, fill: { color: ICE }, line: { type: "none" } });
    s.addText(big, { x: 8.85, y: y + 0.1, w: 3.6, h: 0.7, fontFace: FONT_HEAD, bold: true, fontSize: 32, color: c, isTextBox: true, margin: 0 });
    s.addText(small, { x: 8.85, y: y + 0.85, w: 3.6, h: 0.4, fontFace: FONT_BODY, fontSize: 13, color: GRAYTEXT, isTextBox: true, margin: 0 });
  });
  s.addText("Vì vậy kết luận chủ yếu dựa trên cặp fall ↔ fallen. Vế fallen ↔ lying chưa kiểm chứng được trên tập test này.", {
    x: 8.6, y: 5.2, w: 4.1, h: 1.4, fontFace: FONT_BODY, fontSize: 12.5, color: GRAYTEXT, isTextBox: true, margin: 0 });
  pageNum(s, 9);
}

// ===== 10. Ablation MCFD =====
{
  const s = lightSlide();
  kicker(s, "Kết quả: ảnh hưởng của dữ liệu");
  title(s, "Thêm dữ liệu không phải lúc nào cũng tốt hơn");
  const H = (t, al = "center") => ({ text: t, options: { bold: true, color: WHITE, fill: { color: NAVY }, align: al } });
  const C = (t, fill, al = "center", color = GRAYTEXT, bold = false) => ({ text: t, options: { color, fill: { color: fill }, align: al, bold } });
  const rows = [
    [H("Baseline huấn luyện trên", "left"), H("Fallen-F1 (%)"), H("Fall-F1 (%)"), H("Fall∪Fallen F1 (%)")],
    [C("3 bộ staged + OF-Syn (khuyến nghị)", WHITE, "left", NAVY, true), C("26,6", WHITE), C("45,3", WHITE), C("55,5", WHITE)],
    [C("Thêm MCFD (4 bộ staged + OF-Syn)", "FCEEF0", "left", RED, true), C("21,8", "FCEEF0", "center", RED), C("41,3", "FCEEF0", "center", RED), C("52,8", "FCEEF0", "center", RED)],
  ];
  s.addTable(rows, { x: 0.6, y: 1.85, w: 12.1, h: 1.95, fontFace: FONT_BODY, fontSize: 15, border: { type: "none" }, autoPage: false, colW: [5.2, 2.3, 2.3, 2.3], rowH: 0.65, valign: "middle" });
  s.addText("Cả ba chỉ số đều giảm. Thêm bộ conditioning learnable (Hybrid) trên bản có MCFD cũng không cứu được: Fall∪Fallen chỉ 51,7%.", { x: 0.6, y: 4.0, w: 12.1, h: 0.6, fontFace: FONT_BODY, fontSize: 13.5, color: GRAYTEXT, isTextBox: true, margin: 0 });
  s.addShape("roundRect", { x: 0.6, y: 4.85, w: 12.1, h: 1.8, rectRadius: 0.1, fill: { color: NAVY }, line: { type: "none" } });
  s.addText([
    { text: "Giả thuyết (chưa kiểm chứng riêng):  ", options: { bold: true, color: TEAL } },
    { text: "MCFD chỉ có 1 người thực hiện, quay bằng 8 camera Kinect đời cũ. Mô hình có thể học đặc điểm riêng của người đó và loại camera đó, làm giảm khả năng áp dụng sang video ngoài đời. Đa dạng người thực hiện quan trọng hơn số lượng video.", options: { color: "E4E9F5" } },
  ], { x: 0.95, y: 4.85, w: 11.4, h: 1.8, valign: "middle", fontFace: FONT_BODY, fontSize: 13.5, isTextBox: true, margin: 0 });
  pageNum(s, 10);
}

// ===== 11. Kết quả mô hình =====
{
  const s = lightSlide();
  kicker(s, "Kết quả: mô hình trên tập test");
  title(s, "Với dữ liệu như trên, các phương pháp cho kết quả này");
  const H = (t, al = "center") => ({ text: t, options: { bold: true, color: WHITE, fill: { color: NAVY }, align: al } });
  const C = (t, fill, al = "center", color = GRAYTEXT, bold = false) => ({ text: t, options: { color, fill: { color: fill }, align: al, bold } });
  const rows = [
    [H("Phương pháp", "left"), H("Fallen-F1"), H("Fall-F1"), H("Fall∪Fallen")],
    [C("Baseline", WHITE, "left"), C("26,6", WHITE), C("45,3", WHITE), C("55,5", WHITE)],
    [C("Rule-based", ICE, "left"), C("30,0", ICE), C("45,3", ICE), C("55,6", ICE)],
    [C("Hybrid v8 (3 seed)", RED, "left", WHITE, true), C("34,3 ± 0,6", RED, "center", WHITE, true), C("44,6 ± 0,9", RED, "center", WHITE, true), C("57,1 ± 0,8", RED, "center", WHITE, true)],
  ];
  s.addTable(rows, { x: 0.6, y: 1.85, w: 6.4, h: 2.4, fontFace: FONT_BODY, fontSize: 14, border: { type: "none" }, autoPage: false, colW: [2.2, 1.4, 1.4, 1.4], rowH: 0.6, valign: "middle" });
  s.addText("F1 tính theo %, trên 572 video / 2.804 segment. Hybrid v8: trung bình ± độ lệch chuẩn qua 3 seed.", { x: 0.6, y: 4.35, w: 6.4, h: 0.6, italic: true, fontFace: FONT_BODY, fontSize: 11, color: "8791A8", isTextBox: true, margin: 0 });
  bullets(s, [
    "Fallen-F1 tăng 7,7 điểm so với baseline",
    "Fall∪Fallen tăng 1,6 điểm; Fall-F1 giảm 0,7 điểm",
    "Con số tuyệt đối còn thấp: dữ liệu ngoài đời rất khó",
  ], 0.6, 5.1, 6.4, 1.6, 13, GRAYTEXT, 8);
  const cmW = 5.3, cmH = cmW * (1577 / 2761);
  s.addImage({ path: FIG_CONFUSION, x: 7.4, y: 2.2, w: cmW, h: cmH });
  s.addText("Confusion matrix 4 lớp, Baseline và Hybrid v8", { x: 7.4, y: 2.2 + cmH + 0.1, w: cmW, h: 0.4, align: "center", italic: true, fontFace: FONT_BODY, fontSize: 11, color: "8791A8", isTextBox: true, margin: 0 });
  pageNum(s, 11);
}

// ===== 12. Hạn chế =====
{
  const s = lightSlide();
  kicker(s, "Hạn chế của dữ liệu");
  title(s, "Những điều bộ dữ liệu này chưa cho phép kết luận");
  const items = [
    ["Lying quá ít", "Test chỉ có 17 lying và 11 lie_down. Chưa đánh giá được việc phân biệt fallen với lying, dù đó là động lực ban đầu của đề tài.", RED],
    ["Train lệch về dữ liệu tổng hợp", "9.600 trên 9.877 video train (97%) là OF-Synthetic. Chỉ 277 video staged thật, ít người thực hiện, ít góc camera.", AMBER],
    ["Test chỉ một kiểu ngoài đời", "OF-ItW lấy từ video tai nạn trên mạng. Chưa thử với camera giám sát trong nhà thật.", AMBER],
    ["Thiếu 5 bộ staged", "Chỉ 3 trong 8 bộ staged dùng được. Chưa biết kết quả đổi thế nào nếu có đủ CMDFall, UP-Fall, EDF, OCCU.", AMBER],
  ];
  let y = 1.8;
  items.forEach(([h, body, c]) => {
    s.addShape("rect", { x: 0.6, y, w: 0.08, h: 1.0, fill: { color: c }, line: { type: "none" } });
    s.addText(h, { x: 0.95, y, w: 3.4, h: 1.0, valign: "top", fontFace: FONT_HEAD, bold: true, fontSize: 15, color: NAVY, isTextBox: true, margin: 0 });
    s.addText(body, { x: 4.5, y, w: 8.2, h: 1.0, valign: "top", fontFace: FONT_BODY, fontSize: 13, color: GRAYTEXT, isTextBox: true, margin: 0 });
    y += 1.2;
  });
  pageNum(s, 12);
}

// ===== 13. Liên kết =====
{
  const s = darkSlide();
  kicker(s, "Liên kết nguồn");
  title(s, "Nơi lấy dữ liệu và mã nguồn", true);
  const links = [
    ["Bài báo OmniFall (arXiv)", "https://arxiv.org/abs/2505.19889"],
    ["Trang dự án OmniFall", "https://dsch.ai/omnifall/"],
    ["Dataset trên Hugging Face", "https://huggingface.co/datasets/simplexsigil2/omnifall"],
    ["Mã baseline chính thức", "https://github.com/simplexsigil/omnifall-experiments"],
    ["OOPS (nguồn OF-ItW)", "https://oops.cs.columbia.edu"],
    ["GMDCSA24", "https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos"],
    ["Le2i (bản trên Kaggle)", "https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia"],
    ["CAUCAFall (Kaggle)", "https://www.kaggle.com/datasets/tuyenldvn/caucafall"],
    ["VideoMAE-K400 (backbone)", "https://huggingface.co/MCG-NJU/videomae-base-finetuned-kinetics"],
  ];
  let y = 1.75;
  links.forEach(([h, u]) => {
    s.addText(h, { x: 0.7, y, w: 3.8, h: 0.5, fontFace: FONT_BODY, bold: true, fontSize: 13.5, color: WHITE, isTextBox: true, margin: 0 });
    s.addText([{ text: u, options: { hyperlink: { url: u }, color: "8EC9E8", underline: true } }], { x: 4.6, y, w: 8.2, h: 0.5, fontFace: FONT_BODY, fontSize: 12.5, isTextBox: true, margin: 0 });
    y += 0.56;
  });
  s.addText("Cảm ơn đã lắng nghe", { x: 0.7, y: 6.85, w: 8, h: 0.4, fontFace: FONT_HEAD, bold: true, fontSize: 16, color: LIGHTGRAYTEXT, isTextBox: true, margin: 0 });
  pageNum(s, 13, true);
}

if (_slides.length !== SCRIPT.length) throw new Error("Số slide " + _slides.length + " khác số đoạn script " + SCRIPT.length);
_slides.forEach((sl, i) => sl.addNotes(SCRIPT[i].text));
pres.writeFile({ fileName: path.join(__dirname, "OmniFall_BoDuLieu_TiengViet.pptx") }).then(() => console.log("Deck (data) written."));
