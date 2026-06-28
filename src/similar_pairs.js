import { pipeline } from "@xenova/transformers";
import { WORDS } from "./constants/WORDS.js";
import { cosineSimilarity } from "./utils/cosineSimilarity.js";

const extractor = await pipeline(
  "feature-extraction",
  "Xenova/all-MiniLM-L6-v2",
);

const EMBEDDED_WORDS = []; // {word, emb}

for (const word of WORDS) {
  const emb = await extractor(word, {
    pooling: "mean",
    normalize: true,
  });

  EMBEDDED_WORDS.push(Array.from(emb.data));
}

// console.log(WORDS);
// console.log(EMBEDDED_WORDS);

const PAIRED_WORDS_ARRAY = []; // {word, paired_word,  similarity_score}[]

for (const index in EMBEDDED_WORDS) {
  const first_word_emb = EMBEDDED_WORDS[index];
  //   console.log({ original_word: WORDS[index], emb: first_word_emb });
  let similarity_score_array = []; // {score, index} (index -> embedded_words array)
  for (const i in EMBEDDED_WORDS) {
    if (i !== index) {
      const second_word_emb = EMBEDDED_WORDS[i];
      const similarity_score = cosineSimilarity(
        first_word_emb,
        second_word_emb,
      );

      similarity_score_array.push({ score: similarity_score, index: i });
    }
  }
  const highest_score = Math.max(...similarity_score_array.map((e) => e.score));

  PAIRED_WORDS_ARRAY.push({
    word: WORDS[index],
    paired_word:
      WORDS[
        similarity_score_array.find((e) => e.score === highest_score).index
      ],
    similarity: highest_score,
  });
}

// console.log(PAIRED_WORDS_ARRAY);
// PAIRED_WORDS_ARRAY.forEach((pw) =>
//   console.log(`${pw.word} ↔ ${pw.paired_word} (${pw.similarity})`),
// );

// const PAIRED_WORDS_ARRAY_UNIQUE = PAIRED_WORDS_ARRAY.filter((paired_word) => {
//   if (
//     PAIRED_WORDS_ARRAY.some(
//       (pw) =>
//         pw.paired_word === paired_word.word &&
//         pw.word === paired_word.paired_word,
//     )
//   )
//     return null;
//   return paired_word;
// });
const PAIRED_WORDS_ARRAY_UNIQUE = PAIRED_WORDS_ARRAY;

for (let i = 0; i < PAIRED_WORDS_ARRAY_UNIQUE.length; i++) {
  const paired_word = PAIRED_WORDS_ARRAY_UNIQUE[i];
  const duplicate = PAIRED_WORDS_ARRAY.find(
    (pw) =>
      pw.paired_word === paired_word.word &&
      pw.word === paired_word.paired_word,
  );
  if (duplicate) {
    PAIRED_WORDS_ARRAY_UNIQUE.splice(i, 1);
    // Restart loop
    i = 0;
  }
}

console.log(PAIRED_WORDS_ARRAY_UNIQUE.length); // 10 => ?
PAIRED_WORDS_ARRAY_UNIQUE.forEach((pw) =>
  console.log(`${pw.word} ↔ ${pw.paired_word} (${pw.similarity})`),
);
