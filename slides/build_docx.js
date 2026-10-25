const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, HeadingLevel,
  AlignmentType, WidthType, BorderStyle, ShadingType, Footer, PageNumber, LevelFormat,
} = require("docx");

const S = require("./script_data.js");
const mmss = (t) => Math.floor(t / 60) + ":" + String(t % 60).padStart(2, "0");
const total = S.reduce((a, b) => a + b.sec, 0);

// Câu hỏi thường gặp: lấy từ file .md để chỉ có một nguồn nội dung
const md = fs.readFileSync(path.join(__dirname, "..", "LOI_THUYET_TRINH_BoDuLieu.md"), "utf-8");
const qaPart = md.split("## Câu hỏi có thể gặp và cách trả lời")[1] || "";
const qa = [];
qaPart.split("\n").forEach((line) => {
  const m = line.match(/^\*\*(.+?)\*\*\s*(.*)$/);
  if (m) qa.push([m[1], m[2]]);
});

const FONT = "Calibri";
const NAVY = "1B2A4A", TEAL = "2E86AB", GRAY = "44506B";
const run = (text, o = {}) => new TextRun({ text, font: FONT, ...o });

const children = [];
children.push(new Paragraph({ spacing: { after: 80 }, children: [run("LỜI THUYẾT TRÌNH", { bold: true, size: 20, color: TEAL, characterSpacing: 40 })] }));
children.push(new Paragraph({ heading: HeadingLevel.TITLE, spacing: { after: 120 }, children: [run("Bộ dữ liệu OmniFall của dự án phát hiện té ngã", { bold: true, size: 40, color: NAVY })] }));
children.push(new Paragraph({ spacing: { after: 200 }, children: [run(
  "Dành cho bộ slide “OmniFall_BoDuLieu_TiengViet.pptx” (13 slide). Tổng thời gian ước tính " + mmss(total) +
  " nếu nói chậm rãi. Đây là lời nói gợi ý, không cần đọc nguyên văn. Cùng nội dung đã có trong phần Ghi chú của từng slide.",
  { size: 22, color: GRAY })] }));

// Bảng thời gian
const colW = [900, 5600, 1300, 1226];
const cell = (text, w, o = {}) => new TableCell({
  width: { size: w, type: WidthType.DXA },
  margins: { top: 60, bottom: 60, left: 100, right: 100 },
  shading: o.fill ? { type: ShadingType.CLEAR, fill: o.fill, color: "auto" } : undefined,
  borders: { top: { style: BorderStyle.SINGLE, size: 4, color: "D5DBE8" }, bottom: { style: BorderStyle.SINGLE, size: 4, color: "D5DBE8" }, left: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" }, right: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" } },
  children: [new Paragraph({ alignment: o.align || AlignmentType.LEFT, children: [run(text, { size: 21, bold: !!o.bold, color: o.color || GRAY })] })],
});
let acc = 0;
const rows = [new TableRow({ tableHeader: true, children: [
  cell("Slide", colW[0], { fill: NAVY, bold: true, color: "FFFFFF" }),
  cell("Nội dung", colW[1], { fill: NAVY, bold: true, color: "FFFFFF" }),
  cell("Thời lượng", colW[2], { fill: NAVY, bold: true, color: "FFFFFF", align: AlignmentType.CENTER }),
  cell("Cộng dồn", colW[3], { fill: NAVY, bold: true, color: "FFFFFF", align: AlignmentType.CENTER }),
] })];
S.forEach((s, i) => {
  acc += s.sec;
  const fill = i % 2 ? "F1F4FA" : undefined;
  rows.push(new TableRow({ children: [
    cell(String(i + 1), colW[0], { fill }),
    cell(s.title, colW[1], { fill }),
    cell(mmss(s.sec), colW[2], { fill, align: AlignmentType.CENTER }),
    cell(mmss(acc), colW[3], { fill, align: AlignmentType.CENTER }),
  ] }));
});
children.push(new Table({ width: { size: 9026, type: WidthType.DXA }, columnWidths: colW, rows }));

// Lời từng slide
children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 400, after: 160 }, pageBreakBefore: true, children: [run("Lời nói theo từng slide", { bold: true, size: 32, color: NAVY })] }));
S.forEach((s, i) => {
  children.push(new Paragraph({ heading: HeadingLevel.HEADING_2, keepNext: true, spacing: { before: 280, after: 80 }, children: [
    run("Slide " + (i + 1) + ". " + s.title, { bold: true, size: 26, color: NAVY }),
    run("   (" + mmss(s.sec) + ")", { size: 22, color: TEAL }),
  ] }));
  children.push(new Paragraph({ spacing: { after: 120, line: 320 }, children: [run(s.text, { size: 23 })] }));
});

// Q&A
children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 400, after: 160 }, pageBreakBefore: true, children: [run("Câu hỏi có thể gặp và cách trả lời", { bold: true, size: 32, color: NAVY })] }));
qa.forEach(([q, a]) => {
  children.push(new Paragraph({ keepNext: true, spacing: { before: 200, after: 60 }, children: [run(q, { bold: true, size: 23, color: NAVY })] }));
  children.push(new Paragraph({ spacing: { after: 100, line: 320 }, children: [run(a, { size: 23 })] }));
});

const doc = new Document({
  creator: "OmniFall", title: "Lời thuyết trình - Bộ dữ liệu OmniFall",
  styles: {
    default: { document: { run: { font: FONT, size: 23 } } },
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal", run: { font: FONT, size: 40, bold: true, color: NAVY } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 32, bold: true, color: NAVY }, paragraph: { outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 26, bold: true, color: NAVY }, paragraph: { outlineLevel: 1 } },
    ],
  },
  numbering: { config: [{ reference: "b", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [run("Trang ", { size: 18, color: "8791A8" }), new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18, color: "8791A8" })] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  const out = path.join(__dirname, "..", "LOI_THUYET_TRINH_BoDuLieu.docx");
  fs.writeFileSync(out, buf);
  console.log("written", out, "QA items:", qa.length);
});
