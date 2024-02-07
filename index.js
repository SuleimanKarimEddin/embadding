import { pipeline } from "@sroussey/transformers";
import express from "express";
import cors from "cors";
import bodyParser from "body-parser";
const app = express();
app.use(cors());

app.use(bodyParser.json({ limit: "50mb" }));
app.use(bodyParser.urlencoded({ limit: "50mb", extended: true }));

const PORT = 3005;
const extractor = await pipeline(
  "feature-extraction",
  "Xenova/all-MiniLM-L6-v2"
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
  return res.send(data);
});

app.post("/list", async (req, res) => {
  if (!Array.isArray(req.body))
    return res.send("Word is not an array of strings");
  const word = req.body;
        console.log("start");

  const output = await extractor(word, { pooling: "mean", normalize: true });
  const data = {
    data: output.tolist(),
  };
        console.log("finish");

  return res.send(data);
});

app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
