import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const outDir = "/Users/zhiy/Documents/Rheology ML/outputs/ml_ready_xanthan_positive_20260529";
const workbookPath = path.join(outDir, "yeast_xanthan_rheology_ML_ready_xanthan_positive_only.xlsx");

const sheetOrder = [
  "README",
  "source_inventory",
  "qc_summary",
  "formulation_master",
  "replicate_master",
  "viscosity_long",
  "frequency_long",
  "strain_long",
  "strain_summary_replicate",
  "strain_summary_formulation",
];

function columnLetter(n) {
  let s = "";
  while (n > 0) {
    const m = (n - 1) % 26;
    s = String.fromCharCode(65 + m) + s;
    n = Math.floor((n - m - 1) / 26);
  }
  return s;
}

function asCell(value) {
  if (value === null || value === undefined) return null;
  if (typeof value === "number") return Number.isFinite(value) ? value : null;
  return value;
}

function widthFor(header, values) {
  const samples = [header, ...values.slice(0, 80).map((v) => (v === null || v === undefined ? "" : String(v)))];
  const maxLen = Math.max(...samples.map((s) => String(s).length));
  return Math.max(72, Math.min(230, maxLen * 7 + 18));
}

const workbook = Workbook.create();

for (let i = 0; i < sheetOrder.length; i += 1) {
  const sheetName = sheetOrder[i];
  const jsonPath = path.join(outDir, `${sheetName}.json`);
  const rows = JSON.parse(await fs.readFile(jsonPath, "utf8"));
  const headers = rows.length ? Object.keys(rows[0]) : [];
  const matrix = [headers, ...rows.map((row) => headers.map((h) => asCell(row[h])))];
  const sheet = i === 0
    ? workbook.worksheets.getOrAdd(sheetName, { renameFirstIfOnlyNewSpreadsheet: true })
    : workbook.worksheets.add(sheetName);

  if (!headers.length) continue;
  const lastCol = columnLetter(headers.length);
  const lastRow = matrix.length;
  const fullRange = sheet.getRange(`A1:${lastCol}${lastRow}`);
  fullRange.values = matrix;

  sheet.getRange(`A1:${lastCol}1`).format = {
    fill: "#1F4E5F",
    font: { color: "#FFFFFF", bold: true },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
  };
  fullRange.format.font = { name: "Calibri", size: 10 };
  fullRange.format.verticalAlignment = "top";
  sheet.freezePanes.freezeRows(1);

  for (let c = 0; c < headers.length; c += 1) {
    const letter = columnLetter(c + 1);
    const colValues = rows.map((row) => row[headers[c]]);
    sheet.getRange(`${letter}:${letter}`).format.columnWidthPx = widthFor(headers[c], colValues);
  }

  if (sheetName.includes("master") || sheetName.includes("summary") || sheetName === "qc_summary") {
    fullRange.format.borders = { preset: "inside", style: "thin", color: "#D9E2E7" };
  }
}

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);

await workbook.render({ sheetName: "formulation_master", range: "A1:N18", scale: 2 });
await workbook.render({ sheetName: "replicate_master", range: "A1:L18", scale: 2 });

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(workbookPath);
console.log(workbookPath);
