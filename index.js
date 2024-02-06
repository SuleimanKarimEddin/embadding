import { pipeline } from "@sroussey/transformers";
import express from "express";
import cors from "cors";

const app = express();
app.use(cors());
const PORT = 3000;
const extractor = await pipeline(
  "feature-extraction",
  "Xenova/all-mpnet-base-v2"
);

app.get("/", async (req, res) => {
  if (typeof req.query.word !== "string")
    return res.send("Word is not a string");
  const word = req.query.word;
  const output = await extractor(word, { pooling: "mean", normalize: true });
  const dataArray = Array.from(output.data.values());
  const data = {
    data: dataArray,
  };
  res.send(data);
});

app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
