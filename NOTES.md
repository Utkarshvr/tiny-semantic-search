**Summary**
- **Files:** [src/index.js](src/index.js) and [src/search.js](src/search.js)
- **Purpose:** Short notes describing what each file does and lessons learned while implementing embeddings and similarity search with the `@xenova/transformers` pipeline.

**src/index.js**
- **Purpose:** Create embeddings for a small list of notes and print the resulting embeddings.
- **Key points:**
  - Uses `pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2")` to create an `extractor` function.
  - Embeds each note with `pooling: "mean"` and `normalize: true` so each embedding is approximately a unit vector.
  - Converts the returned embedding to a plain array via `Array.from(emb.data)` and prints `embeddedNotes`.
- **What we learned:** Normalizing embeddings (`normalize: true`) yields unit-length vectors; this simplifies similarity computations because the dot product approximates cosine similarity.

**src/search.js**
- **Purpose:** Embed a set of notes and a query, compute cosine similarity between the query and each note, and return the best match.
- **Key points:**
  - Reuses the same `feature-extraction` pipeline and model `Xenova/all-MiniLM-L6-v2`.
  - Implements a numerical `cosineSimilarity(a, b)` that computes `dot/(|A||B|)`.
  - Since embeddings are normalized, magnitudes are ~1 so the dot product equals the cosine score, but computing magnitudes keeps the function robust if `normalize` is changed.
  - The script embeds each note and the query with `pooling: "mean"` and `normalize: true`.
  - Iterates over `embeddedNotes`, computes similarity, and tracks `bestMatch` using `highestScore` initialized to `-1` (safe because cosine ranges from -1 to 1).
- **What we learned / Notes:**
  - There was an earlier, commented-out cosine attempt that returned only -1, 0 or 1; that approach lost important similarity resolution. Replacing it with the standard dot/(|A||B|) formula fixes this.
  - When comparing floating-point similarity scores, avoid equality checks; prefer thresholds or rank-based selection.

**Run instructions**

1. Install dependencies (if not already installed):

```
npm install
```

2. Run the example scripts:

```
node src/index.js
node src/search.js
```

**Next steps / Suggestions**
- Add a small CLI wrapper to accept queries at runtime.
- Persist embeddings to a file or lightweight DB to avoid re-embedding notes on every run.
- Add a similarity threshold or return top-N matches instead of only the single best match.

---
File created: [NOTES.md](NOTES.md)
