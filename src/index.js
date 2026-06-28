import { pipeline } from "@xenova/transformers";

const extractor = await pipeline(
  "feature-extraction",
  "Xenova/all-MiniLM-L6-v2",
);

// const embedding = await extractor(
//   "The mitochondria is the powerhouse of the cell.",
//   {
//     pooling: "mean",
//     normalize: true,
//   },
// );

// console.log(embedding.data);

const embeddedNotes = [];

const notes = [
  "Newton's First Law states that an object remains at rest or in motion unless acted upon by a force.",

  "The mitochondria is the powerhouse of the cell.",

  "JavaScript is a programming language used for web development.",

  "Photosynthesis converts sunlight into energy.",
];

for (const note of notes) {
  const emb = await extractor(note, {
    pooling: "mean",
    normalize: true,
  });

  embeddedNotes.push({
    text: note,
    embedding: Array.from(emb.data),
  });
}

console.log(embeddedNotes);
