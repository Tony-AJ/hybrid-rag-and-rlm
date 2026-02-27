import { ESLint } from "eslint";
async function run() {
  const eslint = new ESLint();
  const results = await eslint.lintFiles(["src/pages/GraphView.tsx", "src/services/api.ts"]);
  const formatter = await eslint.loadFormatter("stylish");
  const resultText = formatter.format(results);
  console.log(resultText);
}
run().catch(console.error);
