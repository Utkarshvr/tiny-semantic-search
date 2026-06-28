export function cosineSimilarity(w1, w2) {
  let dot = 0;
  let magW1 = 0;
  let magW2 = 0;

  w1.forEach((w1_token, index) => {
    dot += w1_token * w2[index];
    magW1 += w1_token * w1_token;
    magW2 += w2[index] * w2[index];
  });

  const similarity = (dot / Math.sqrt(magW1)) * Math.sqrt(magW2);

  return similarity;
}
