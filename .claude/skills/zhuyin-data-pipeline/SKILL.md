---
name: zhuyin-data-pipeline
description: Build the offline benchmark data from Taiwan Traditional Chinese text with g2pW, including the three input conditions (full zhuyin with tone, no tone, initials only). Use when preparing, splitting, or validating dev and test sets.
---

# Zhuyin data pipeline

## Source text

Taiwan Traditional Chinese: news, zh-TW Wikipedia, PTT. Normalize with OpenCC
only when the source is not already zh-TW (`s2twp` for Simplified sources);
never convert text that is already Traditional, since that may change
Taiwan specific forms the benchmark wants to test.

## Steps

1. Clean: strip markup, URLs, latin runs, and digits that would need reading
   normalization. Keep sentence punctuation for splitting only.
2. Split into short sentences of 5 to 20 Han characters. Keep the previous
   sentence as `context` for each example.
3. Convert each sentence to zhuyin with g2pW (`from g2pw import G2PWConverter`,
   `style="bopomofo"`). The model files download on first use into the HF
   cache or package dir; run once inside the container to warm it.
4. Drop sentences where any character has no reading or the reading count
   differs from the character count.
5. Derive the three input conditions from the same reading:
   - `full`: syllable with tone mark
   - `notone`: tone marks removed
   - `initial`: only the initial (shengmu) of each syllable; syllables with no
     initial keep their first final symbol
6. Split into dev and test, about 5,000 sentences each, with a fixed seed.
   Split by document, not by sentence, to avoid context leakage.

## Output format

JSON Lines, one object per sentence:

```json
{"id": "news-000123-04", "context": "...", "text": "...", "zhuyin_full": ["..."], "zhuyin_notone": ["..."], "zhuyin_initial": ["..."]}
```

Store under `data/` (gitignored). Commit only the build script and a small
fixture under `tests/fixtures/`.

## Quality checks

- Manually inspect a random sample of about 200 sentences for g2pW errors and
  report the error rate; polyphone errors bound achievable accuracy.
- Track counts of the focus error types: high frequency homophones (zai, de,
  ta variants), Taiwan forms (li, zhe variants), proper nouns.
