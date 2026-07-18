const { Document, Packer, Paragraph, TextRun, AlignmentType } = require('docx');
const L = { width: 12240, height: 15840 };
const p = (text, opts={}) => new Paragraph({
  alignment: opts.center ? AlignmentType.CENTER : AlignmentType.LEFT,
  spacing: { after: opts.after ?? 160 },
  children: [new TextRun({ text, bold: !!opts.bold, size: opts.size ?? 22 })],
});
const title = "When Does Memory Help? A Cost-Aware Evaluation of Long-Term Memory in Tool-Using LLM Agents";
const titlePage = new Document({ sections: [{ properties: { page: { size: L } }, children: [
  p("Title Page", { bold: true, size: 28, center: true, after: 240 }),
  p("Manuscript title:", { bold: true }),
  p(title, { after: 240 }),
  p("Authors and affiliations:", { bold: true }),
  p("Shweta Mishra - Independent Researcher (e-mail: 1196shweta@gmail.com)"),
  p("Shashank Mishra - Independent Researcher (e-mail: 24shashankm@gmail.com)", { after: 240 }),
  p("Corresponding author:", { bold: true }),
  p("Shweta Mishra, e-mail: 1196shweta@gmail.com", { after: 240 }),
  p("Acknowledgments:", { bold: true }),
  p("The authors used Claude (Anthropic) as an assistive tool for drafting text, developing harness and analysis code (including the figure-generation scripts), and preparing this manuscript, disclosed per the IEEE guidelines on AI-generated text and cited in the manuscript. All experimental design decisions, preregistered hypotheses, data collection, and final content were reviewed and are the sole responsibility of the authors. All figures are deterministic plots of measured experimental data, reproducible from the released code and traces. The authors declare no conflict of interest."),
]}]});
const coi = new Document({ sections: [{ properties: { page: { size: L } }, children: [
  p("Conflict of Interest Statement", { bold: true, size: 28, center: true, after: 240 }),
  p("Manuscript: " + title, { after: 240 }),
  p("None of the authors (Shweta Mishra, Shashank Mishra) have a conflict of interest to disclose. The research received no external funding; all API costs were self-funded by the authors."),
]}]});
const cover = new Document({ sections: [{ properties: { page: { size: L } }, children: [
  p("Cover Letter", { bold: true, size: 28, center: true, after: 240 }),
  p("To the Editor-in-Chief, IEEE Transactions on Artificial Intelligence:", { after: 240 }),
  p('Please consider the enclosed manuscript, "' + title + '," for publication as a regular paper.'),
  p("The manuscript presents MERIT, a benchmark and evaluation harness that measures whether long-term memory changes what a tool-using LLM agent does - and at what cost - rather than what it can recall. It reports 23,440 scored episodes across three agent models and three seeds with preregistered hypotheses, plus gated spot-checks on latest-generation models."),
  p("This manuscript is original, has not been published previously, and is not under consideration by any other journal or conference. All authors have approved the manuscript and its submission. Consistent with the double-anonymized review policy, author-identifying information and the public repository link appear only on the Title Page; the benchmark code, data, and full episode traces are publicly released and will be linked in the final version."),
  p("The use of an AI assistant (Claude, Anthropic) in preparing the manuscript and code is disclosed in the Acknowledgments and cited in the references, per the IEEE guidelines on AI-generated text."),
  p("Thank you for your consideration.", { after: 240 }),
  p("Sincerely,"),
  p("Shweta Mishra (corresponding author, 1196shweta@gmail.com)"),
  p("Shashank Mishra"),
]}]});
(async () => {
  require('fs').writeFileSync('title-page.docx', await Packer.toBuffer(titlePage));
  require('fs').writeFileSync('conflict-of-interest.docx', await Packer.toBuffer(coi));
  require('fs').writeFileSync('cover-letter.docx', await Packer.toBuffer(cover));
  console.log('wrote 3 docx');
})();
