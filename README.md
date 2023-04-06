# nlp-tutor

## File Description
* grammar_match: 使用 spacy 將英文句子 parsing 成 dependency path，並使用 sentence_transformer 將 parsing 出來的內容轉換成 specific layer 的 embedding，再用 faiss 為資料庫(dict)中的 sentence 建立 index 與搜索。其中 `search()` 和 `search_similar_sentences()` 的差異是前者支援多個英文句子的輸入，並會在資料庫尋找與這些輸入句子都相似的句子並輸出（取交集）。