# Lab Day 19 - GraphRAG với corpus 100 bài Wikipedia về công ty AI

Repo này triển khai bài lab GraphRAG end-to-end:

- Cào 100 bài Wikipedia về AI/tech companies.
- Dùng OpenAI `gpt-4o-mini` để trích xuất triples `(subject, relation, object)`.
- Ghi knowledge graph vào Neo4j.
- So sánh Flat RAG và GraphRAG trên 20 câu hỏi benchmark multi-hop.
- Xuất bảng kết quả, token/cost/time và hình minh họa graph.

Graph backend chính là **Neo4j**. Ảnh visualization chính là `outputs/visualization.png`, được chụp từ Neo4j Browser. File `outputs/knowledge_graph.png` là ảnh backup sinh bằng NetworkX/Matplotlib.

## 1. Clone và cài đặt

```powershell
git clone <repo-url>
cd lab_19
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Mở `.env` và điền OpenAI API key:

```env
OPENAI_API_KEY=sk-...
OPENAI_TEXT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=labday19pass
```

## 2. Chạy Neo4j bằng Docker

Chạy lần đầu:

```powershell
docker run --name lab19-neo4j `
  -p 7474:7474 `
  -p 7687:7687 `
  -e NEO4J_AUTH=neo4j/labday19pass `
  -v lab19_neo4j_data:/data `
  -d neo4j:2026.03.1
```

Nếu container đã tồn tại nhưng đang tắt:

```powershell
docker start lab19-neo4j
```

Kiểm tra Neo4j đang chạy:

```powershell
docker ps
```

## 3. Tạo lại corpus Wikipedia

Corpus thật đã có sẵn trong repo tại:

```text
data/wikipedia_ai_company_corpus.jsonl
data/wikipedia_ai_company_manifest.json
```

Nếu muốn cào lại từ đầu:

```powershell
python scrape_wikipedia_ai_companies.py --limit 100
```

Mỗi dòng corpus có `id`, `title`, `url`, và `text`. File manifest lưu lại URL Wikipedia để kiểm tra nguồn dữ liệu.

## 4. Chạy pipeline end-to-end

Chạy toàn bộ pipeline, bao gồm triple extraction bằng OpenAI, build graph Neo4j, Flat RAG, GraphRAG và evaluation:

```powershell
python graph_rag_lab.py --run
```

Nếu chỉ muốn test 3 câu để giảm chi phí:

```powershell
python graph_rag_lab.py --run --limit-questions 3
```

Nếu đã có `outputs/triples.json` và chỉ muốn chạy lại benchmark mới mà không trích triples lại:

```powershell
python graph_rag_lab.py --run --reuse-triples outputs/triples.json
```

Lệnh `--reuse-triples` vẫn dùng OpenAI cho phần embedding và trả lời, nhưng bỏ qua bước triple extraction nên nhanh và rẻ hơn.

## 5. Output sau khi chạy

Sau khi chạy xong, kiểm tra thư mục `outputs/`:

```text
outputs/triples.json
outputs/evaluation.csv
outputs/evaluation.md
outputs/token_usage.json
outputs/knowledge_graph.png
```

Ý nghĩa:

- `triples.json`: triples do LLM trích xuất.
- `evaluation.csv`: bảng so sánh Flat RAG và GraphRAG.
- `evaluation.md`: bảng Markdown để đưa vào báo cáo.
- `token_usage.json`: số token, thời gian chạy, chi phí ước tính.
- `knowledge_graph.png`: ảnh graph backup bằng Matplotlib.

## 6. Xem visualization trong Neo4j

Mở Neo4j Browser:

```text
http://localhost:7474
```

Đăng nhập:

```text
Username: neo4j
Password: labday19pass
```

Chạy Cypher:

```cypher
MATCH p=(n:Lab19Entity)-[r]->(m:Lab19Entity)
RETURN p
LIMIT 120;
```

Chụp màn hình Neo4j Browser và lưu thành:

```text
outputs/visualization.png
```

Đây là ảnh graph chính để nộp báo cáo.

## 7. Benchmark

Benchmark nằm trong:

```text
data/benchmark_questions.json
```

Bộ câu hỏi hiện tại được thiết kế theo hướng multi-hop/graph-shaped. Mỗi câu bắt đầu từ một entity cụ thể và yêu cầu gom nhiều quan hệ, ví dụ:

```text
Starting from OpenAI Global, LLC, which company invested in it,
what investment amount is connected, and which AI products did it develop?
```

Cách thiết kế này giúp kiểm tra đúng thế mạnh của GraphRAG: duyệt graph 2-hop và tổng hợp các cạnh liên quan.

## 8. Cấu hình GraphRAG

GraphRAG dùng:

- Entity selection để chọn node bắt đầu.
- Neo4j 2-hop traversal để lấy graph context.
- Giới hạn tối đa 50 cạnh trong context.
- OpenAI `gpt-4o-mini` để sinh câu trả lời cuối cùng.

Trong code:

```python
MAX_GRAPH_CONTEXT_EDGES = 50
```

