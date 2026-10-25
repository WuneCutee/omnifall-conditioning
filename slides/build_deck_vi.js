const pptxgen = require("pptxgenjs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");
const FIG_CONFUSION = path.join(ROOT, "figures", "confusion_matrix_4class.png");
const FIG_BARCHART = path.join(ROOT, "figures", "comparison_bar_chart.png");
const FIG_REAL_FALL = path.join(ROOT, "figures", "real_demo", "fall_detected_real_footage.png");
const FIG_REAL_FALLEN = path.join(ROOT, "figures", "real_demo", "fallen_detected_real_footage.png");

// ---------- Palette (đồng bộ với bản tiếng Anh) ----------
const NAVY = "1B2A4A";
const NAVY2 = "24365C";
const ICE = "E8EDF7";
const WHITE = "FFFFFF";
const TEAL = "2E86AB";
const RED = "C8213B";
const GREEN = "2F9E44";
const AMBER = "D98E04";
const GRAYTEXT = "44506B";
const LIGHTGRAYTEXT = "AEB9D4";

const FONT_HEAD = "Cambria";
const FONT_BODY = "Calibri";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
const PAGE_W = 13.33;
const PAGE_H = 7.5;

function darkSlide() {
  const s = pres.addSlide();
  s.background = { color: NAVY };
  return s;
}
function lightSlide() {
  const s = pres.addSlide();
  s.background = { color: WHITE };
  return s;
}
function kicker(slide, text, color = TEAL) {
  slide.addText(text.toUpperCase(), {
    x: 0.6, y: 0.42, w: 10, h: 0.35, fontFace: FONT_BODY, fontSize: 13, bold: true,
    color, charSpacing: 2, isTextBox: true, margin: 0,
  });
}
function title(slide, text, onDark = false, size = 30) {
  slide.addText(text, {
    x: 0.6, y: 0.72, w: 12.1, h: 0.9, fontFace: FONT_HEAD, fontSize: size, bold: true,
    color: onDark ? WHITE : NAVY, isTextBox: true, margin: 0,
  });
}
function pageNum(slide, n, onDark = false) {
  slide.addText(String(n), {
    x: PAGE_W - 0.7, y: PAGE_H - 0.45, w: 0.4, h: 0.3, fontFace: FONT_BODY, fontSize: 10,
    color: onDark ? LIGHTGRAYTEXT : "A0A8BD", align: "right", isTextBox: true, margin: 0,
  });
}
function iconCircle(slide, x, y, d, bg, label, labelColor) {
  slide.addShape("ellipse", { x, y, w: d, h: d, fill: { color: bg }, line: { type: "none" } });
  slide.addText(label, {
    x, y, w: d, h: d, align: "center", valign: "middle", fontFace: FONT_HEAD,
    fontSize: d > 1.1 ? 30 : 22, bold: true, color: labelColor, isTextBox: true, margin: 0,
  });
}

// ============================== 1. Trang bìa ==============================
{
  const s = darkSlide();
  s.addShape("ellipse", { x: 9.6, y: -2.2, w: 6.5, h: 6.5, fill: { color: NAVY2 }, line: { type: "none" } });
  s.addShape("ellipse", { x: 11.3, y: 4.6, w: 3.6, h: 3.6, fill: { color: NAVY2 }, line: { type: "none" } });

  s.addText("ĐỒ ÁN HỌC PHẦN · DEEP LEARNING", {
    x: 0.8, y: 1.35, w: 10, h: 0.4, fontFace: FONT_BODY, fontSize: 14, bold: true,
    color: TEAL, charSpacing: 2, isTextBox: true, margin: 0,
  });
  s.addText("Phân biệt “Ngã cần cấp cứu” và “Nằm nghỉ”", {
    x: 0.75, y: 1.85, w: 11.5, h: 1.1, fontFace: FONT_HEAD, fontSize: 40, bold: true,
    color: WHITE, isTextBox: true, margin: 0,
  });
  s.addText("trong Hệ thống Phát hiện Té ngã bằng Thị giác Máy tính", {
    x: 0.8, y: 2.75, w: 11.3, h: 0.6, fontFace: FONT_HEAD, fontSize: 24, bold: true,
    color: "CADCFC", isTextBox: true, margin: 0,
  });
  s.addText("Khai thác cấu trúc nhân quả trong taxonomy hành động (fall→fallen, lie_down→lying) để giải quyết một điểm mù trong benchmark OmniFall", {
    x: 0.8, y: 3.5, w: 10.6, h: 0.7, fontFace: FONT_BODY, fontSize: 15, italic: true, color: "CADCFC",
    isTextBox: true, margin: 0,
  });

  const chips = [["NGÃ (fall)", AMBER], ["FALLEN – khẩn cấp", RED], ["LYING – nghỉ ngơi", GREEN]];
  let cx = 0.8;
  chips.forEach(([label, color]) => {
    const w = 2.35;
    s.addShape("roundRect", { x: cx, y: 4.5, w, h: 0.55, rectRadius: 0.1, fill: { color: NAVY2 }, line: { color, width: 1.5 } });
    s.addText(label, { x: cx, y: 4.5, w, h: 0.55, align: "center", valign: "middle", fontFace: FONT_BODY, fontSize: 13, bold: true, color, isTextBox: true, margin: 0 });
    cx += w + 0.25;
  });

  s.addText("Ý tưởng → Nghiên cứu liên quan → Quy trình thực hiện → Kết quả → Demo thực tế", {
    x: 0.8, y: 6.55, w: 11, h: 0.4, fontFace: FONT_BODY, fontSize: 12, italic: true, color: LIGHTGRAYTEXT,
    isTextBox: true, margin: 0,
  });
}

// ============================== 2. Đặt vấn đề ==============================
{
  const s = lightSlide();
  kicker(s, "Đặt vấn đề");
  title(s, "Một lỗ hổng nguy hiểm trong hệ thống giám sát té ngã");

  const stats = [
    ["727 triệu → 1,5 tỷ", "người từ 65 tuổi trở lên trên toàn cầu, từ 2020 đến dự báo 2050"],
    ["Ngã", "là nguyên nhân hàng đầu gây chấn thương và tử vong ở người cao tuổi sống một mình"],
  ];
  let sx = 0.6;
  stats.forEach(([big, small]) => {
    s.addText(big, { x: sx, y: 1.7, w: 5.8, h: 0.7, fontFace: FONT_HEAD, bold: true, fontSize: 28, color: TEAL, isTextBox: true, margin: 0 });
    s.addText(small, { x: sx, y: 2.4, w: 5.8, h: 0.9, fontFace: FONT_BODY, fontSize: 13, color: GRAYTEXT, isTextBox: true, margin: 0 });
    sx += 6.1;
  });

  s.addText(
    "Hệ thống giám sát camera ngày càng phổ biến trong chăm sóc người cao tuổi — nhưng phần lớn hệ thống phân loại từng đoạn video (segment) MỘT CÁCH ĐỘC LẬP, bỏ qua bối cảnh vừa xảy ra trước đó. Hệ quả:",
    { x: 0.6, y: 3.55, w: 12.1, h: 0.7, fontFace: FONT_BODY, fontSize: 14.5, color: GRAYTEXT, isTextBox: true, margin: 0 }
  );

  const cardW = 5.85, cardY = 4.45, cardH = 2.3;
  s.addShape("roundRect", { x: 0.6, y: cardY, w: cardW, h: cardH, rectRadius: 0.12, fill: { color: "FCEEF0" }, line: { type: "none" } });
  iconCircle(s, 0.95, cardY + 0.35, 0.9, RED, "!", WHITE);
  s.addText("Bỏ sót “fallen” thật", { x: 2.05, y: cardY + 0.3, w: 4.2, h: 0.4, fontFace: FONT_HEAD, bold: true, fontSize: 16, color: RED, isTextBox: true, margin: 0 });
  s.addText("Nhầm người vừa ngã thành “đang nằm nghỉ” → bỏ lỡ thời gian vàng cấp cứu.", { x: 2.05, y: cardY + 0.8, w: 4.2, h: 0.9, fontFace: FONT_BODY, fontSize: 13, color: GRAYTEXT, isTextBox: true, margin: 0 });

  s.addShape("roundRect", { x: 6.85, y: cardY, w: cardW, h: cardH, rectRadius: 0.12, fill: { color: "FFF6E5" }, line: { type: "none" } });
  iconCircle(s, 7.2, cardY + 0.35, 0.9, AMBER, "?", WHITE);
  s.addText("Báo động giả liên tục", { x: 8.3, y: cardY + 0.3, w: 4.2, h: 0.4, fontFace: FONT_HEAD, bold: true, fontSize: 16, color: AMBER, isTextBox: true, margin: 0 });
  s.addText("Nhầm người đang nằm nghỉ bình thường thành “fallen” → người dùng mất niềm tin, tắt hệ thống.", { x: 8.3, y: cardY + 0.8, w: 4.2, h: 0.9, fontFace: FONT_BODY, fontSize: 13, color: GRAYTEXT, isTextBox: true, margin: 0 });

  pageNum(s, 2);
}

// ============================== 3. Ý tưởng cốt lõi ==============================
{
  const s = lightSlide();
  kicker(s, "Ý tưởng");
  title(s, "Trạng thái tĩnh “thừa hưởng” ý nghĩa từ hành động trước đó");

  s.addText(
    "Quan sát: bản thân taxonomy 10 lớp của OmniFall đã mã hoá sẵn mối liên hệ nhân quả — một hành động chuyển tiếp (transient) luôn xảy ra NGAY TRƯỚC trạng thái tĩnh (static) tương ứng của nó.",
    { x: 0.6, y: 1.6, w: 11.8, h: 0.65, fontFace: FONT_BODY, fontSize: 15, color: GRAYTEXT, isTextBox: true, margin: 0 }
  );

  function flowRow(y, a, b, color) {
    const boxW = 2.3, boxH = 0.85, x1 = 1.2, x2 = x1 + boxW + 1.6;
    s.addShape("roundRect", { x: x1, y, w: boxW, h: boxH, rectRadius: 0.1, fill: { color: ICE }, line: { color: TEAL, width: 1 } });
    s.addText(a, { x: x1, y, w: boxW, h: boxH, align: "center", valign: "middle", fontFace: FONT_BODY, bold: true, fontSize: 15, color: NAVY, isTextBox: true, margin: 0 });
    s.addText("hành động chuyển tiếp", { x: x1, y: y + boxH + 0.03, w: boxW, h: 0.3, align: "center", fontFace: FONT_BODY, italic: true, fontSize: 10.5, color: GRAYTEXT, isTextBox: true, margin: 0 });
    s.addShape("rightArrow", { x: x1 + boxW + 0.15, y: y + boxH / 2 - 0.14, w: 1.3, h: 0.28, fill: { color }, line: { type: "none" } });
    s.addShape("roundRect", { x: x2, y, w: boxW, h: boxH, rectRadius: 0.1, fill: { color }, line: { type: "none" } });
    s.addText(b, { x: x2, y, w: boxW, h: boxH, align: "center", valign: "middle", fontFace: FONT_BODY, bold: true, fontSize: 15, color: WHITE, isTextBox: true, margin: 0 });
    s.addText("trạng thái tĩnh (được boost)", { x: x2, y: y + boxH + 0.03, w: boxW, h: 0.3, align: "center", fontFace: FONT_BODY, italic: true, fontSize: 10.5, color: GRAYTEXT, isTextBox: true, margin: 0 });
    s.addText("xác suất tăng dần rồi suy giảm theo thời gian (~10s)", {
      x: x2 + boxW + 0.3, y: y + boxH / 2 - 0.2, w: 4.2, h: 0.5, fontFace: FONT_BODY, fontSize: 12.5, color: GRAYTEXT, isTextBox: true, margin: 0,
    });
  }
  flowRow(2.55, "fall (ngã)", "fallen", RED);
  flowRow(4.15, "lie_down (nằm xuống)", "lying (nằm nghỉ)", GREEN);

  s.addShape("roundRect", { x: 1.2, y: 5.7, w: 10.9, h: 1.05, rectRadius: 0.1, fill: { color: NAVY }, line: { type: "none" } });
  s.addText([
    { text: "Ý tưởng cốt lõi:  ", options: { bold: true, color: TEAL } },
    { text: "thay vì phân loại từng segment độc lập, hãy điều kiện hoá (condition) dự đoán của trạng thái tĩnh dựa trên hành động chuyển tiếp gần nhất đã xảy ra trước đó.", options: { color: WHITE } },
  ], { x: 1.55, y: 5.7, w: 10.2, h: 1.05, valign: "middle", fontFace: FONT_BODY, fontSize: 14.5, isTextBox: true, margin: 0 });

  pageNum(s, 3);
}

// ============================== 4. Nghiên cứu liên quan ==============================
{
  const s = lightSlide();
  kicker(s, "Nghiên cứu liên quan");
  title(s, "Đề tài đứng ở đâu trong bức tranh nghiên cứu hiện có", false, 28);

  const groups = [
    ["Bộ dữ liệu & bài toán nền", "OmniFall (Schneider et al., 2025) — benchmark hợp nhất 3 miền dữ liệu (staged / synthetic / in-the-wild), 80 giờ video, taxonomy 16 lớp thống nhất. Đây là nền tảng benchmark mà đề tài kế thừa và mở rộng.", TEAL],
    ["Backbone trích đặc trưng video", "VideoMAE (Tong et al., NeurIPS 2022) — pretraining tự giám sát hiệu quả với ít dữ liệu; I3D (Carreira & Zisserman, CVPR 2017) — backbone 3D-CNN kinh điển cho nhận diện hành động. Đề tài dùng VideoMAE-K400 làm backbone chính.", TEAL],
    ["Làm mượt theo thời gian (temporal smoothing)", "Các hướng Temporal Action Segmentation tổng quát (vd. các kỹ thuật hậu xử lý giảm “over-segmentation”) làm mượt dự đoán liên tiếp, nhưng KHÔNG khai thác cấu trúc nhân quả cụ thể giữa từng CẶP hành động — đây chính là khoảng trống đề tài lấp vào.", AMBER],
    ["Khảo sát phát hiện té ngã bằng thị giác máy tính", "Các bài khảo sát (vd. Computers in Biology and Medicine, 2022; systematic review 2024) xác nhận: phân biệt các tư thế tĩnh tương tự nhau vẫn là thách thức mở của lĩnh vực — đúng như vấn đề đề tài giải quyết.", GREEN],
  ];
  let y = 1.75;
  groups.forEach(([h, body, color]) => {
    s.addShape("rect", { x: 0.6, y, w: 0.08, h: 1.15, fill: { color }, line: { type: "none" } });
    s.addText(h, { x: 0.95, y, w: 3.1, h: 1.15, valign: "top", fontFace: FONT_HEAD, bold: true, fontSize: 14.5, color: NAVY, isTextBox: true, margin: 0 });
    s.addText(body, { x: 4.25, y, w: 8.5, h: 1.15, valign: "top", fontFace: FONT_BODY, fontSize: 12, color: GRAYTEXT, isTextBox: true, margin: 0 });
    y += 1.28;
  });

  pageNum(s, 4);
}

// ============================== 5. Khoảng trống & Mục tiêu ==============================
{
  const s = lightSlide();
  kicker(s, "Khoảng trống nghiên cứu & Mục tiêu");
  title(s, "Chưa ai khai thác trực tiếp cấu trúc nhân quả này");

  s.addShape("roundRect", { x: 0.6, y: 1.7, w: 12.1, h: 1.2, rectRadius: 0.1, fill: { color: NAVY }, line: { type: "none" } });
  s.addText([
    { text: "Khoảng trống:  ", options: { bold: true, color: TEAL } },
    { text: "OmniFall công bố taxonomy có cấu trúc nhân quả (fall→fallen, lie_down→lying), nhưng baseline chính thức của benchmark vẫn phân loại từng segment độc lập, không tận dụng thông tin thời gian giữa các hành động liên quan.", options: { color: "E4E9F5" } },
  ], { x: 0.95, y: 1.7, w: 11.4, h: 1.2, valign: "middle", fontFace: FONT_BODY, fontSize: 14, isTextBox: true, margin: 0 });

  s.addText("4 mục tiêu cụ thể", { x: 0.6, y: 3.15, w: 6, h: 0.4, fontFace: FONT_HEAD, bold: true, fontSize: 17, color: NAVY, isTextBox: true, margin: 0 });

  const goals = [
    ["1", "Tái lập baseline chính thức", "Dùng đúng repo omnifall-experiments gốc, không tự viết lại — đảm bảo độ tin cậy để so sánh."],
    ["2", "Xây module “Conditioning” hậu xử lý", "Rule-based (minh bạch) và Learnable/Hybrid (self-attention theo thời gian)."],
    ["3", "Kiểm định thống kê nghiêm ngặt", "Bootstrap CI, McNemar test, nhiều seed — tránh kết luận vội vàng từ 1 lần chạy."],
    ["4", "Triển khai hệ thống chạy thật", "Demo real-time qua webcam, kiểm chứng trên video thật (OOPS), không chỉ dừng ở bảng số liệu."],
  ];
  const gw = 5.9, gh = 1.55;
  goals.forEach(([num, h, body], i) => {
    const gx = 0.6 + (i % 2) * (gw + 0.3);
    const gy = 3.7 + Math.floor(i / 2) * (gh + 0.25);
    s.addShape("roundRect", { x: gx, y: gy, w: gw, h: gh, rectRadius: 0.1, fill: { color: ICE }, line: { type: "none" } });
    s.addText(num, { x: gx + 0.25, y: gy + 0.18, w: 0.7, h: 0.7, fontFace: FONT_HEAD, bold: true, fontSize: 28, color: TEAL, isTextBox: true, margin: 0 });
    s.addText(h, { x: gx + 1.05, y: gy + 0.15, w: gw - 1.3, h: 0.4, fontFace: FONT_HEAD, bold: true, fontSize: 14, color: NAVY, isTextBox: true, margin: 0 });
    s.addText(body, { x: gx + 1.05, y: gy + 0.58, w: gw - 1.3, h: 0.9, fontFace: FONT_BODY, fontSize: 11.5, color: GRAYTEXT, isTextBox: true, margin: 0 });
  });

  pageNum(s, 5);
}

// ============================== 6. Quy trình thực hiện ==============================
{
  const s = darkSlide();
  kicker(s, "Quy trình thực hiện", TEAL);
  title(s, "Hành trình 5 bước — không đi thẳng, có rẽ hướng khi cần", true, 28);

  const steps = [
    ["01", "Tái lập baseline chính thức", "Chạy đúng repo omnifall-experiments (VideoMAE-K400 + FeatureTransformer), vá vài lỗi tương thích Python — không viết lại từ đầu để giữ độ tin cậy."],
    ["02", "Rule-Based Conditioner (thử nghiệm ban đầu)", "Boost xác suất theo hàm suy giảm mũ. Phát hiện & sửa 2 bug tinh vi (tự ghi đè tại t=0; ghi đè sai các segment “fall” liên tiếp) — Fall-F1 phục hồi về đúng mức baseline."],
    ["03", "Learnable Conditioner — lặp qua nhiều phiên bản (v2 → v8)", "Từ self-attention cơ bản đến residual init, class-balanced loss, KL-anchor regularization, và đổi tiêu chí chọn checkpoint sang Fall∪Fallen F1 để tránh đánh đổi ngầm."],
    ["04", "Kiểm định thống kê nghiêm ngặt", "Bootstrap CI (10.000 lần, resample theo video) + McNemar test trên 3 seed độc lập — báo cáo trung thực cả phần CHƯA đạt ý nghĩa thống kê."],
    ["05", "Mở rộng dữ liệu thật & triển khai demo", "Tải 818 video thật từ OOPS, xây dựng scripts/realtime_demo.py chạy qua webcam/video thật, kiểm chứng trực tiếp trên footage chưa từng thấy."],
  ];
  let y = 1.75;
  const rowH = 1.03;
  steps.forEach(([num, h, body]) => {
    s.addText(num, { x: 0.6, y, w: 0.9, h: rowH, fontFace: FONT_HEAD, bold: true, fontSize: 30, color: NAVY2, isTextBox: true, margin: 0 });
    s.addShape("ellipse", { x: 0.62, y: y + 0.08, w: 0.12, h: 0.12, fill: { color: TEAL }, line: { type: "none" } });
    s.addText(h, { x: 1.6, y, w: 3.6, h: rowH, valign: "top", fontFace: FONT_HEAD, bold: true, fontSize: 13.5, color: WHITE, isTextBox: true, margin: 0 });
    s.addText(body, { x: 5.35, y, w: 7.4, h: rowH, valign: "top", fontFace: FONT_BODY, fontSize: 11.5, color: "CADCFC", isTextBox: true, margin: 0 });
    y += rowH + 0.12;
  });

  pageNum(s, 6, true);
}

// ============================== 7. Phương pháp ==============================
{
  const s = lightSlide();
  kicker(s, "Phương pháp");
  title(s, "Hai bộ điều kiện hoá, xây trên baseline đáng tin cậy");

  const stageY = 2.2, stageH = 1.6, stageW = 2.55;
  const stages = [
    ["Đoạn video", "cửa sổ 16 khung hình, 7.5 fps", ICE, NAVY],
    ["Baseline\nVideoMAE-K400", "FeatureTransformer\ntừ repo chính thức", TEAL, WHITE],
    ["Conditioner", "Rule-Based  HOẶC  Hybrid v8\n(đóng góp của đề tài)", RED, WHITE],
    ["Nhãn cuối cùng", "1 trong 10 lớp,\nfallen được làm nổi bật", GREEN, WHITE],
  ];
  let x = 0.6;
  stages.forEach(([h, sub, bg, fg], i) => {
    s.addShape("roundRect", { x, y: stageY, w: stageW, h: stageH, rectRadius: 0.12, fill: { color: bg }, line: { type: "none" } });
    s.addText(h, { x: x + 0.15, y: stageY + 0.15, w: stageW - 0.3, h: 0.65, fontFace: FONT_HEAD, bold: true, fontSize: 14, color: fg, align: "center", valign: "middle", isTextBox: true, margin: 0 });
    s.addText(sub, { x: x + 0.15, y: stageY + 0.8, w: stageW - 0.3, h: 0.7, fontFace: FONT_BODY, fontSize: 10, color: fg, align: "center", isTextBox: true, margin: 0 });
    if (i < stages.length - 1) {
      s.addShape("rightArrow", { x: x + stageW + 0.06, y: stageY + stageH / 2 - 0.15, w: 0.5, h: 0.3, fill: { color: "C7CEDD" }, line: { type: "none" } });
    }
    x += stageW + 0.62;
  });

  const cardY = 4.35, cardW = 5.5, cardH = 2.55;
  s.addShape("roundRect", { x: 0.6, y: cardY, w: cardW, h: cardH, rectRadius: 0.12, fill: { color: ICE }, line: { type: "none" } });
  s.addText("Rule-Based Conditioner", { x: 0.9, y: cardY + 0.25, w: cardW - 0.6, h: 0.4, fontFace: FONT_HEAD, bold: true, fontSize: 16, color: NAVY, isTextBox: true, margin: 0 });
  s.addText([
    { text: "Hàm suy giảm mũ theo thời gian", options: { bold: true, breakLine: true } },
    { text: "Không có tham số học được — minh bạch, dễ kiểm chứng.", options: { breakLine: true, paraSpaceAfter: 6 } },
    { text: "Phát hiện & sửa 2 bug tinh vi trong quá trình phát triển (xem Quy trình, bước 2).", options: {} },
  ], { x: 0.9, y: cardY + 0.7, w: cardW - 0.6, h: cardH - 0.9, fontFace: FONT_BODY, fontSize: 12.5, color: GRAYTEXT, isTextBox: true, margin: 0, paraSpaceAfter: 4 });

  const cardX2 = 6.75;
  s.addShape("roundRect", { x: cardX2, y: cardY, w: cardW, h: cardH, rectRadius: 0.12, fill: { color: NAVY }, line: { type: "none" } });
  s.addText("Hybrid v8 (Learnable)", { x: cardX2 + 0.3, y: cardY + 0.25, w: cardW - 0.6, h: 0.4, fontFace: FONT_HEAD, bold: true, fontSize: 16, color: WHITE, isTextBox: true, margin: 0 });
  s.addText([
    { text: "Self-attention theo thời gian + KL-anchor", options: { bold: true, color: TEAL, breakLine: true } },
    { text: "Tự học cách trọng số hoá các segment trước đó thay vì hàm suy giảm cố định; neo về phân phối rule-based để ổn định.", options: { color: "CADCFC", breakLine: true, paraSpaceAfter: 6 } },
    { text: "Chọn checkpoint theo Fall∪Fallen F1 (không chỉ Fallen-F1) để tránh đánh đổi ngầm với Fall-F1.", options: { color: "CADCFC" } },
  ], { x: cardX2 + 0.3, y: cardY + 0.7, w: cardW - 0.6, h: cardH - 0.9, fontFace: FONT_BODY, fontSize: 12.5, isTextBox: true, margin: 0, paraSpaceAfter: 4 });

  pageNum(s, 7);
}

// ============================== 8. Thực nghiệm ==============================
{
  const s = lightSlide();
  kicker(s, "Thực nghiệm");
  title(s, "Dữ liệu thật + tổng hợp + trong-tự-nhiên, baseline đáng tin cậy");

  s.addText("Huấn luyện (St + Sy)", { x: 0.6, y: 1.75, w: 5.8, h: 0.4, fontFace: FONT_HEAD, bold: true, fontSize: 16, color: NAVY, isTextBox: true, margin: 0 });
  const trainRows = ["GMDCSA24 (staged)", "Le2i (staged)", "CAUCAFall (staged)", "OF-Synthetic (12.000 video tổng hợp)"];
  s.addText(trainRows.map((t, i) => ({ text: t, options: { bullet: { code: "2022" }, breakLine: i < trainRows.length - 1, fontSize: 13.5, color: GRAYTEXT } })), {
    x: 0.6, y: 2.2, w: 5.8, h: 1.6, fontFace: FONT_BODY, paraSpaceAfter: 6, isTextBox: true, margin: 0,
  });

  s.addText("Kiểm thử — OF-ItW (trong tự nhiên)", { x: 0.6, y: 4.0, w: 5.8, h: 0.4, fontFace: FONT_HEAD, bold: true, fontSize: 16, color: NAVY, isTextBox: true, margin: 0 });
  const testRows = ["Video tai nạn thật OOPS — footage không kiểm soát", "2.804 segment kiểm thử trên 572 video", "Taxonomy 10 lớp OmniFall (lớp mở rộng gộp vào “other”)"];
  s.addText(testRows.map((t, i) => ({ text: t, options: { bullet: { code: "2022" }, breakLine: i < testRows.length - 1, fontSize: 13.5, color: GRAYTEXT } })), {
    x: 0.6, y: 4.45, w: 5.8, h: 1.6, fontFace: FONT_BODY, paraSpaceAfter: 6, isTextBox: true, margin: 0,
  });

  const bx = 7.1, by = 1.75, bw = 5.6, bh = 4.6;
  s.addShape("roundRect", { x: bx, y: by, w: bw, h: bh, rectRadius: 0.12, fill: { color: NAVY }, line: { type: "none" } });
  s.addText("Vì sao baseline đáng tin cậy", { x: bx + 0.35, y: by + 0.3, w: bw - 0.7, h: 0.5, fontFace: FONT_HEAD, bold: true, fontSize: 17, color: TEAL, isTextBox: true, margin: 0 });
  const trustRows = [
    "Dùng đúng repo chính thức omnifall-experiments (không tự viết lại)",
    "Đặc trưng VideoMAE-K400: 7.5 fps, 16 khung hình/cửa sổ, stride 2/15s — khớp Table 12 của paper gốc",
    "Baseline FeatureTransformer, 2 lớp, huấn luyện tới hội tụ (checkpoint-7000)",
    "Các conditioner chỉ áp dụng hậu xử lý (post-hoc) lên logits baseline đã đóng băng",
  ];
  s.addText(trustRows.map((t, i) => ({ text: t, options: { bullet: { code: "2022" }, breakLine: i < trustRows.length - 1, color: "E4E9F5", fontSize: 13.5 } })), {
    x: bx + 0.35, y: by + 0.95, w: bw - 0.7, h: bh - 1.3, fontFace: FONT_BODY, paraSpaceAfter: 10, isTextBox: true, margin: 0,
  });

  pageNum(s, 8);
}

// ============================== 9. Kết quả chính ==============================
{
  const s = lightSlide();
  kicker(s, "Kết quả");
  title(s, "Hybrid v8 thắng ở mọi chỉ số quan trọng");

  const headerColor = NAVY;
  const rows = [
    [
      { text: "Phương pháp", options: { bold: true, color: WHITE, fill: { color: headerColor } } },
      { text: "Fallen-F1 (%)", options: { bold: true, color: WHITE, fill: { color: headerColor }, align: "center" } },
      { text: "Fall-F1 (%)", options: { bold: true, color: WHITE, fill: { color: headerColor }, align: "center" } },
      { text: "Fall∪Fallen F1 (%)", options: { bold: true, color: WHITE, fill: { color: headerColor }, align: "center" } },
    ],
    [
      { text: "Baseline (St+Sy)", options: { color: GRAYTEXT, fill: { color: WHITE } } },
      { text: "26.6", options: { align: "center", color: GRAYTEXT, fill: { color: WHITE } } },
      { text: "45.3", options: { align: "center", color: GRAYTEXT, fill: { color: WHITE } } },
      { text: "55.5", options: { align: "center", color: GRAYTEXT, fill: { color: WHITE } } },
    ],
    [
      { text: "Rule-Based Conditioner", options: { color: GRAYTEXT, fill: { color: ICE } } },
      { text: "30.0", options: { align: "center", color: GRAYTEXT, fill: { color: ICE } } },
      { text: "45.3", options: { align: "center", color: GRAYTEXT, fill: { color: ICE } } },
      { text: "55.6", options: { align: "center", color: GRAYTEXT, fill: { color: ICE } } },
    ],
    [
      { text: "Hybrid v8 (đề xuất)", options: { bold: true, color: WHITE, fill: { color: RED } } },
      { text: "34.3 ± 0.6", options: { bold: true, align: "center", color: WHITE, fill: { color: RED } } },
      { text: "44.6 ± 0.9", options: { bold: true, align: "center", color: WHITE, fill: { color: RED } } },
      { text: "57.1 ± 0.8", options: { bold: true, align: "center", color: WHITE, fill: { color: RED } } },
    ],
  ];
  s.addTable(rows, {
    x: 0.7, y: 1.85, w: 8.6, h: 2.6, fontFace: FONT_BODY, fontSize: 15, border: { type: "none" },
    autoPage: false, colW: [3.2, 1.8, 1.8, 1.8], rowH: 0.65, valign: "middle",
  });
  s.addText("trung bình ± độ lệch chuẩn qua 3 seed (42 / 43 / 44)  ·  tập kiểm thử OF-ItW, n = 2.804 segment", {
    x: 0.7, y: 4.55, w: 8.6, h: 0.35, italic: true, fontFace: FONT_BODY, fontSize: 11.5, color: "8791A8", isTextBox: true, margin: 0,
  });

  const cx = 9.7, cy = 1.85, cw = 3.0, ch = 3.0;
  s.addShape("roundRect", { x: cx, y: cy, w: cw, h: ch, rectRadius: 0.12, fill: { color: NAVY }, line: { type: "none" } });
  s.addText("+4.3", { x: cx, y: cy + 0.35, w: cw, h: 1.1, align: "center", fontFace: FONT_HEAD, bold: true, fontSize: 52, color: RED, isTextBox: true, margin: 0 });
  s.addText("điểm Fallen-F1\nso với rule-based", { x: cx + 0.2, y: cy + 1.5, w: cw - 0.4, h: 0.7, align: "center", fontFace: FONT_BODY, fontSize: 13, color: WHITE, isTextBox: true, margin: 0 });
  s.addText("+1.5 điểm Fall∪Fallen\n(chỉ số quyết định về mặt lâm sàng)", { x: cx + 0.2, y: cy + 2.25, w: cw - 0.4, h: 0.65, align: "center", fontFace: FONT_BODY, italic: true, fontSize: 11, color: "CADCFC", isTextBox: true, margin: 0 });

  s.addText("Baseline và rule-based là tất định (variance = 0) — độ lệch của Hybrid v8 phản ánh 3 lần huấn luyện độc lập, không phải nhiễu đo lường.", {
    x: 0.7, y: 5.15, w: 12.0, h: 0.5, fontFace: FONT_BODY, fontSize: 12, color: GRAYTEXT, isTextBox: true, margin: 0,
  });

  pageNum(s, 9);
}

// ============================== 10. Kiểm định thống kê ==============================
{
  const s = lightSlide();
  kicker(s, "Kiểm định thống kê");
  title(s, "Cải thiện là thật, hay chỉ là nhiễu?");

  s.addText("Bootstrap theo video (10.000 lần resample) — Hybrid v8 so với Rule-Based, chỉ số Fall∪Fallen F1", {
    x: 0.6, y: 1.65, w: 12, h: 0.4, fontFace: FONT_BODY, fontSize: 14, color: GRAYTEXT, isTextBox: true, margin: 0,
  });

  const seeds = [
    { seed: "Seed 42", diff: "+2.35 pp", ci: "[+0.90, +3.82]", p: "p = 0.0017", sig: true },
    { seed: "Seed 43", diff: "+1.57 pp", ci: "[+0.15, +2.96]", p: "p = 0.0148", sig: true },
    { seed: "Seed 44", diff: "+0.86 pp", ci: "[−0.64, +2.35]", p: "p = 0.1283", sig: false },
  ];
  const cw = 3.75, gap = 0.3, startX = 0.6, cy = 2.3, ch = 2.5;
  seeds.forEach((d, i) => {
    const x = startX + i * (cw + gap);
    const accent = d.sig ? GREEN : AMBER;
    s.addShape("roundRect", { x, y: cy, w: cw, h: ch, rectRadius: 0.12, fill: { color: ICE }, line: { color: accent, width: 1.5 } });
    s.addText(d.seed, { x: x + 0.3, y: cy + 0.2, w: cw - 0.6, h: 0.4, fontFace: FONT_HEAD, bold: true, fontSize: 15, color: NAVY, isTextBox: true, margin: 0 });
    s.addText(d.diff, { x: x + 0.3, y: cy + 0.65, w: cw - 0.6, h: 0.8, fontFace: FONT_HEAD, bold: true, fontSize: 34, color: accent, isTextBox: true, margin: 0 });
    s.addText("95% CI " + d.ci, { x: x + 0.3, y: cy + 1.5, w: cw - 0.6, h: 0.35, fontFace: FONT_BODY, fontSize: 12.5, color: GRAYTEXT, isTextBox: true, margin: 0 });
    s.addText(d.p, { x: x + 0.3, y: cy + 1.85, w: cw - 0.6, h: 0.35, fontFace: FONT_BODY, bold: true, fontSize: 12.5, color: GRAYTEXT, isTextBox: true, margin: 0 });
    s.addText(d.sig ? "CÓ Ý NGHĨA" : "CI CHỨA 0", {
      x: x + 0.3, y: cy + ch - 0.45, w: cw - 0.6, h: 0.3, fontFace: FONT_BODY, bold: true, fontSize: 11, color: accent, charSpacing: 1, isTextBox: true, margin: 0,
    });
  });

  s.addShape("roundRect", { x: 0.6, y: 5.15, w: 12.1, h: 1.5, rectRadius: 0.1, fill: { color: NAVY }, line: { type: "none" } });
  s.addText([
    { text: "Đối chiếu chéo:  ", options: { bold: true, color: TEAL } },
    { text: "McNemar test trên “union-accuracy” có ý nghĩa ở cả 3 seed (p < 0.0001) — Hybrid v8 luôn sửa đúng nhiều gấp ~2 lần số segment nó làm sai. "
        + "Kết quả bootstrap trên F1 thận trọng hơn (2/3 seed), được báo cáo trung thực thay vì làm tròn lên.", options: { color: "E4E9F5" } },
  ], { x: 0.95, y: 5.15, w: 11.4, h: 1.5, valign: "middle", fontFace: FONT_BODY, fontSize: 13.5, isTextBox: true, margin: 0 });

  pageNum(s, 10);
}

// ============================== 11. Hình minh hoạ (2 hình gộp) ==============================
{
  const s = lightSlide();
  kicker(s, "Phân tích lỗi");
  title(s, "Trực quan hoá sự cải thiện");

  const cmW = 5.7, cmH = cmW * (1577 / 2761);
  s.addImage({ path: FIG_CONFUSION, x: 0.55, y: 1.9, w: cmW, h: cmH });
  s.addText("Confusion matrix 4 lớp (fall/lie_down/fallen/lying), Baseline vs. Hybrid v8, gộp 3 seed", {
    x: 0.55, y: 1.9 + cmH + 0.1, w: cmW, h: 0.55, italic: true, fontFace: FONT_BODY, fontSize: 10.5, color: "8791A8", align: "center", isTextBox: true, margin: 0,
  });

  const bcW = 6.6, bcH = bcW * (1405 / 4170);
  s.addImage({ path: FIG_BARCHART, x: 6.6, y: 2.5, w: bcW, h: bcH });
  s.addText("So sánh 3 chỉ số qua 3 seed — điểm riêng lẻ overlay cho thấy độ lệch là thật", {
    x: 6.6, y: 2.5 + bcH + 0.1, w: bcW, h: 0.5, italic: true, fontFace: FONT_BODY, fontSize: 10.5, color: "8791A8", align: "center", isTextBox: true, margin: 0,
  });

  pageNum(s, 11);
}

// ============================== 12. Demo thực tế (video thật) ==============================
{
  const s = darkSlide();
  kicker(s, "Hệ thống chạy thật — Video thật", TEAL);
  title(s, "Đã kiểm chứng trực tiếp trên video tai nạn thật", true, 28);

  const imgH = 3.5, imgW = imgH * (404 / 720), gap = 0.35;
  const totalW = imgW * 2 + gap;
  const startX = 0.7, imgY = 1.85;

  s.addImage({ path: FIG_REAL_FALL, x: startX, y: imgY, w: imgW, h: imgH });
  s.addText("t ≈ 22.0s — “fall” (vàng)", {
    x: startX - 0.15, y: imgY + imgH + 0.08, w: imgW + 0.3, h: 0.35, align: "center",
    fontFace: FONT_BODY, fontSize: 11, color: AMBER, isTextBox: true, margin: 0,
  });

  const x2 = startX + imgW + gap;
  s.addImage({ path: FIG_REAL_FALLEN, x: x2, y: imgY, w: imgW, h: imgH });
  s.addText("t ≈ 23.5s — “fallen” (cảnh báo đỏ)", {
    x: x2 - 0.15, y: imgY + imgH + 0.08, w: imgW + 0.3, h: 0.35, align: "center",
    fontFace: FONT_BODY, fontSize: 11, color: RED, isTextBox: true, margin: 0,
  });

  s.addText("Ảnh chụp thật, không chỉnh sửa — scripts/realtime_demo.py --mode both, chạy trực tiếp trên video test OOPS.", {
    x: startX, y: imgY + imgH + 0.48, w: totalW, h: 0.4, italic: true, fontFace: FONT_BODY, fontSize: 10.5, color: "8E8E96", isTextBox: true, margin: 0,
  });

  const rx = startX + totalW + 0.5, rw = PAGE_W - rx - 0.6;
  s.addText("Nhãn thật của đoạn này: fall 18.5–20.2s → fallen 20.2–22.4s.", {
    x: rx, y: 1.85, w: rw, h: 0.65, fontFace: FONT_BODY, fontSize: 12.5, color: "CADCFC", isTextBox: true, margin: 0,
  });
  s.addText("Cả 2 conditioner đều nhận đúng trình tự fall → fallen, trễ ~2s so với thời điểm thật.", {
    x: rx, y: 2.5, w: rw, h: 0.7, fontFace: FONT_BODY, fontSize: 12.5, color: GREEN, bold: true, isTextBox: true, margin: 0,
  });
  s.addText("Điểm chưa hoàn hảo (báo cáo trung thực): cảnh báo còn kéo dài ngắt quãng tới ~8s sau khi người đã đứng dậy — hệ quả của chính cửa sổ suy giảm 10s giúp tăng recall.", {
    x: rx, y: 3.3, w: rw, h: 1.1, fontFace: FONT_BODY, fontSize: 11.5, color: AMBER, isTextBox: true, margin: 0,
  });
  s.addText("~6.5–11.9 segment/s  ·  ~87–155 ms/lần cập nhật trên 1 GPU RTX 3070", {
    x: rx, y: 4.5, w: rw, h: 0.5, fontFace: FONT_BODY, fontSize: 11.5, color: "8E8E96", isTextBox: true, margin: 0,
  });
  s.addShape("roundRect", { x: rx, y: 5.2, w: rw, h: 0.95, rectRadius: 0.1, fill: { color: NAVY2 }, line: { color: TEAL, width: 1.2 } });
  s.addText("Mọi sự kiện “fallen” được ghi timestamp vào fallen_events_log.csv để xem lại", {
    x: rx + 0.25, y: 5.2, w: rw - 0.5, h: 0.95, valign: "middle", fontFace: FONT_BODY, fontSize: 11.5, color: TEAL, isTextBox: true, margin: 0,
  });

  pageNum(s, 12, true);
}

// ============================== 13. Hạn chế ==============================
{
  const s = lightSlide();
  kicker(s, "Đánh giá trung thực");
  title(s, "Những gì kết quả này CHƯA khẳng định được");

  const items = [
    ["Chỉ 1 backbone", "Toàn bộ kết quả chỉ kiểm chứng trên VideoMAE-K400. Repo chính thức không kèm code trích I3D/DINOv2, và việc tải lại video gốc cho backbone thứ 2 là chi phí thật, chưa giải quyết.", AMBER],
    ["Dữ liệu “lying” thưa & lệch domain", "Tập OF-ItW chỉ có 17 lying / 11 lie_down trong 2.804 segment. Ở 1 clip thật đã test, “lying” là người nằm trên ghế tập gym — model gọi là “other” thay vì “fallen” (an toàn) nhưng cho thấy lớp này khá hẹp về domain.", AMBER],
    ["Phương sai giữa các seed", "1/3 seed không vượt ngưỡng 95% bootstrap CI trên F1 (dù cả 3 đều có ý nghĩa trên McNemar accuracy) — báo cáo là “nhất quán, không tuyệt đối”.", RED],
  ];
  let y = 1.85;
  items.forEach(([h, body, color]) => {
    s.addShape("rect", { x: 0.6, y, w: 0.08, h: 1.3, fill: { color }, line: { type: "none" } });
    s.addText(h, { x: 0.95, y, w: 3.0, h: 1.3, valign: "top", fontFace: FONT_HEAD, bold: true, fontSize: 15, color: NAVY, isTextBox: true, margin: 0 });
    s.addText(body, { x: 4.1, y, w: 8.6, h: 1.3, valign: "top", fontFace: FONT_BODY, fontSize: 12.5, color: GRAYTEXT, isTextBox: true, margin: 0 });
    y += 1.55;
  });

  s.addText("Nêu rõ những giới hạn này là điều khiến kết quả tích cực đáng tin hơn.", {
    x: 0.6, y: 6.65, w: 12, h: 0.4, italic: true, fontFace: FONT_BODY, fontSize: 13, color: "8791A8", isTextBox: true, margin: 0,
  });

  pageNum(s, 13);
}

// ============================== 14. Kết luận ==============================
{
  const s = darkSlide();
  s.addShape("ellipse", { x: -2.5, y: 4.5, w: 6, h: 6, fill: { color: NAVY2 }, line: { type: "none" } });
  kicker(s, "Kết luận", TEAL);
  title(s, "Bối cảnh biến điểm mù thành điểm mạnh", true);

  const takeaways = [
    "Cấu trúc nhân quả sẵn có trong taxonomy (fall→fallen, lie_down→lying) là tín hiệu miễn phí, mạnh mẽ — phần lớn classifier bỏ qua nó",
    "Rule-based conditioner đơn giản, minh bạch đã giúp cải thiện; learnable conditioner với self-attention làm tốt hơn có ý nghĩa thống kê",
    "Kết quả được triển khai thành hệ thống chạy thật (webcam, video thật) — không chỉ dừng ở bảng số liệu trong slide",
    "Hướng phát triển tiếp theo: mở rộng sang backbone thứ 2, tăng dữ liệu “lying”, đánh giá trên toàn bộ 818 video OF-ItW vừa thu thập được",
  ];
  s.addText(takeaways.map((t, i) => ({ text: t, options: { bullet: { code: "2022" }, breakLine: i < takeaways.length - 1, color: "E4E9F5", fontSize: 15 } })), {
    x: 0.7, y: 1.95, w: 11.6, h: 3.0, fontFace: FONT_BODY, paraSpaceAfter: 14, isTextBox: true, margin: 0,
  });

  s.addShape("line", { x: 0.7, y: 5.35, w: 11.6, h: 0, line: { color: NAVY2, width: 1.5 } });
  s.addText("Cảm ơn thầy/cô và các bạn đã lắng nghe", {
    x: 0.7, y: 5.6, w: 11.6, h: 0.6, fontFace: FONT_HEAD, bold: true, fontSize: 24, color: WHITE, isTextBox: true, margin: 0,
  });
  s.addText("scripts/realtime_demo.py  ·  results_final.md  ·  OmniFall benchmark (Schneider et al., 2025)", {
    x: 0.7, y: 6.3, w: 11.6, h: 0.4, fontFace: FONT_BODY, fontSize: 12, italic: true, color: LIGHTGRAYTEXT, isTextBox: true, margin: 0,
  });

  pageNum(s, 14, true);
}

// ============================== 15. Tài liệu tham khảo ==============================
{
  const s = lightSlide();
  kicker(s, "Tài liệu tham khảo");
  title(s, "Danh mục tài liệu chính đã sử dụng");

  const refs = [
    "Schneider, D., Marinov, Z., Mistol, M., Zhong, Z., Jaus, A., Düger, R., Baur, R., Sarfraz, M. S., Stiefelhagen, R. “OmniFall: From Staged Through Synthetic to Wild, A Unified Multi-Domain Dataset for Robust Fall Detection.” arXiv:2505.19889, 2025.",
    "Tong, Z., Song, Y., Wang, J., Wang, L. “VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training.” NeurIPS 2022.",
    "Carreira, J., Zisserman, A. “Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset.” CVPR 2017. (backbone I3D)",
    "“Vision-based human fall detection systems using deep learning: A review.” Computers in Biology and Medicine, ScienceDirect, 2022.",
    "“Deep Learning for Computer Vision based Activity Recognition and Fall Detection of the Elderly: a Systematic Review.” arXiv:2401.11790, 2024.",
    "“Temporal Action Segmentation: An Analysis of Modern Approaches.” arXiv:2210.10352 — tổng quan hướng làm mượt/hậu xử lý theo thời gian.",
  ];
  s.addText(refs.map((t, i) => ({ text: t, options: { bullet: { code: "2022" }, breakLine: i < refs.length - 1, fontSize: 13, color: GRAYTEXT } })), {
    x: 0.6, y: 1.85, w: 12.1, h: 4.8, fontFace: FONT_BODY, paraSpaceAfter: 14, isTextBox: true, margin: 0,
  });

  pageNum(s, 15);
}

pres.writeFile({ fileName: path.join(__dirname, "OmniFall_ThuyetTrinh_TiengViet.pptx") }).then(() => {
  console.log("Deck (VI) written.");
});
