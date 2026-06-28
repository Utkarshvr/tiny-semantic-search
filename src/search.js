import { pipeline } from "@xenova/transformers";

const extractor = await pipeline(
  "feature-extraction",
  "Xenova/all-MiniLM-L6-v2",
);

const query = "the powerhouse of the cell is";
// const query = "how to develop a web application?";
// const query = "What part of the cell creates energy?";

// // Cosine similarity function (FAILED, YOU NEED CONFIDENCE BW 0 AND 1 NOT EXACTLY 1 OR -1 OR 0)
// function cosineSimilarity(a, b) {
//   let dot = 0;

//   for (let i = 0; i < a.length; i++) {
//     dot += a[i] * b[i];
//   }

//   //   Can't be used:
//   // Understand the concept: we are basically calculating angle bw two vectors. If the angle is 0, then they are similar, if the angle is 90 degrees, then they are not similar, if the angle is 180 degrees, then they are opposite. So we need to return a value between -1 and 1. But we need to return a value between 0 and 1. So we can use the following formula to convert the value to a range between 0 and 1.
//   return dot > 0 ? 1 : dot < 0 ? -1 : 0;
// }

// Cosine similarity function
function cosineSimilarity(query, note, noteText) {
  let dot = 0;
  let magA = 0;
  let magB = 0;

  // query.length = note.length
  // Same model creates embeddings of same dimensions regardless of anything
  for (let i = 0; i < query.length; i++) {
    dot += query[i] * note[i];
    magA += query[i] * query[i];
    magB += note[i] * note[i];
  }

  const similarity = dot / (Math.sqrt(magA) * Math.sqrt(magB)); // = cos(theta) = similarity score as "Dot Product = [A].[B] = |A||B|cos(theta)"
  console.log({ text: noteText, similarity });
  return similarity;
}

// Embedding the notes
const embeddedNotes = [];

const notes = [
  "Newton's First Law states that an object remains at rest or in motion unless acted upon by a force.",

  "The mitochondria is the powerhouse of the cell.",

  "Power = Work / Time.",

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

// Embedding the query
const queryEmbedding = await extractor(query, {
  pooling: "mean",
  normalize: true,
});

// Finding the best match
let bestMatch = null;
let highestScore = -1;

for (const note of embeddedNotes) {
  const score = cosineSimilarity(
    queryEmbedding.data,
    note.embedding,
    note.text,
  );

  if (score > highestScore) {
    highestScore = score;
    bestMatch = note.text;
  }
}

console.log(`Query: ${query}`);
console.log(`Best match: ${bestMatch}`);
console.log(`Similarity score: ${highestScore}`);
