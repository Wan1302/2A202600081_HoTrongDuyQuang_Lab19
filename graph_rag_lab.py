from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
DEFAULT_CORPUS_PATH = DATA_DIR / "wikipedia_ai_company_corpus.jsonl"

DEFAULT_TEXT_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_NEO4J_URI = "bolt://localhost:7687"
DEFAULT_NEO4J_USER = "neo4j"
DEFAULT_NEO4J_PASSWORD = "labday19pass"
MAX_GRAPH_CONTEXT_EDGES = 50

TRIPLES_RESPONSE_FORMAT = {
    "type": "json_schema",
    "name": "knowledge_graph_triples",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "triples": {
                "type": "array",
                "maxItems": 20,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "subject": {"type": "string"},
                        "relation": {"type": "string"},
                        "object": {"type": "string"},
                    },
                    "required": ["subject", "relation", "object"],
                },
            }
        },
        "required": ["triples"],
    },
}


@dataclass(frozen=True)
class CorpusDoc:
    id: str
    text: str


@dataclass(frozen=True)
class Triple:
    subject: str
    relation: str
    object: str
    source_id: str

    def normalized_key(self) -> tuple[str, str, str]:
        return (
            normalize_entity(self.subject),
            normalize_relation(self.relation),
            normalize_entity(self.object),
        )


@dataclass
class Usage:
    text_input_tokens: int = 0
    text_output_tokens: int = 0
    embedding_tokens: int = 0
    text_calls: int = 0
    embedding_calls: int = 0
    seconds: float = 0.0

    def add_text(self, response: Any) -> None:
        self.text_calls += 1
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        self.text_input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
        self.text_output_tokens += int(getattr(usage, "output_tokens", 0) or 0)

    def add_embeddings(self, response: Any) -> None:
        self.embedding_calls += 1
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        self.embedding_tokens += int(getattr(usage, "total_tokens", 0) or 0)

    def estimated_cost_usd(self) -> float:
        input_price = float(os.getenv("OPENAI_INPUT_PRICE_PER_MTOK", "0.15"))
        output_price = float(os.getenv("OPENAI_OUTPUT_PRICE_PER_MTOK", "0.60"))
        embed_price = float(os.getenv("OPENAI_EMBED_PRICE_PER_MTOK", "0.02"))
        return (
            self.text_input_tokens / 1_000_000 * input_price
            + self.text_output_tokens / 1_000_000 * output_price
            + self.embedding_tokens / 1_000_000 * embed_price
        )


def normalize_entity(value: str) -> str:
    value = re.sub(r"\s+", " ", value.strip())
    return value.casefold()


def normalize_relation(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().upper())
    return value.strip("_")


def canonical_label(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def load_corpus(path: Path) -> list[CorpusDoc]:
    docs: list[CorpusDoc] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            docs.append(CorpusDoc(id=item["id"], text=item["text"]))
    return docs


def load_questions(path: Path) -> list[str]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_cached_triples(path: Path) -> list[Triple]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [Triple(**row) for row in rows]


def response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return text
    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            value = getattr(content, "text", None)
            if value:
                chunks.append(value)
    return "\n".join(chunks)


def call_openai_text(
    client: OpenAI,
    model: str,
    instructions: str,
    prompt: str,
    usage: Usage,
    max_output_tokens: int = 1200,
    text_format: dict[str, Any] | None = None,
) -> str:
    kwargs: dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": prompt,
        "max_output_tokens": max_output_tokens,
    }
    if text_format is not None:
        kwargs["text"] = {"format": text_format}
    response = client.responses.create(**kwargs)
    usage.add_text(response)
    return response_text(response).strip()


def parse_json_object(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
        raw = re.sub(r"```$", "", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", raw):
            try:
                obj, _ = decoder.raw_decode(raw[match.start() :])
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
        raise


def write_bad_json(doc_id: str, raw: str) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / f"bad_json_{doc_id}.txt"
    path.write_text(raw, encoding="utf-8")


def repair_triples_json(
    client: OpenAI,
    model: str,
    raw: str,
    usage: Usage,
) -> dict[str, Any]:
    instructions = (
        "Repair invalid model output into valid JSON that matches the schema. "
        "Do not add facts. Preserve only triples that are clearly present."
    )
    prompt = (
        "Convert this invalid output into valid JSON with this shape:\n"
        '{"triples":[{"subject":"...","relation":"...","object":"..."}]}\n\n'
        f"Invalid output:\n{raw[:6000]}"
    )
    repaired = call_openai_text(
        client,
        model,
        instructions,
        prompt,
        usage,
        max_output_tokens=2000,
        text_format=TRIPLES_RESPONSE_FORMAT,
    )
    return parse_json_object(repaired)


def extract_triples_with_openai(
    client: OpenAI,
    docs: list[CorpusDoc],
    model: str,
    usage: Usage,
) -> list[Triple]:
    instructions = (
        "You extract knowledge graph triples from short technology-company text. "
        "Return only valid JSON. Use concise entity names. Relations must be "
        "UPPER_SNAKE_CASE. Keep years, products, founders, acquisitions, owners, "
        "investments, infrastructure, and hardware links when present."
    )
    triples: list[Triple] = []
    for index, doc in enumerate(docs, start=1):
        prompt = (
            "Extract triples from this document.\n\n"
            "JSON schema:\n"
            '{"triples":[{"subject":"...","relation":"...","object":"..."}]}\n\n'
            "Rules:\n"
            "- Extract at most 20 high-value triples.\n"
            "- Prefer company, founder, product, acquisition, owner, investor, location, and year facts.\n"
            "- Return compact JSON only.\n\n"
            f"document_id: {doc.id}\n"
            f"text: {doc.text}"
        )
        print(f"Extracting triples {index}/{len(docs)}: {doc.id}", flush=True)
        raw = call_openai_text(
            client,
            model,
            instructions,
            prompt,
            usage,
            max_output_tokens=2500,
            text_format=TRIPLES_RESPONSE_FORMAT,
        )
        try:
            data = parse_json_object(raw)
        except json.JSONDecodeError:
            try:
                data = repair_triples_json(client, model, raw, usage)
            except json.JSONDecodeError:
                write_bad_json(doc.id, raw)
                print(f"Skipped {doc.id}: invalid JSON saved to outputs/bad_json_{doc.id}.txt", flush=True)
                continue
        for row in data.get("triples", []):
            subject = canonical_label(str(row.get("subject", "")))
            relation = normalize_relation(str(row.get("relation", "")))
            obj = canonical_label(str(row.get("object", "")))
            if subject and relation and obj:
                triples.append(Triple(subject, relation, obj, doc.id))
    return dedupe_triples(triples)


def dedupe_triples(triples: list[Triple]) -> list[Triple]:
    seen: dict[tuple[str, str, str], Triple] = {}
    for triple in triples:
        seen.setdefault(triple.normalized_key(), triple)
    return sorted(seen.values(), key=lambda t: (t.subject, t.relation, t.object))


def build_graph(triples: list[Triple]) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    labels: dict[str, str] = {}
    for triple in triples:
        s_key = normalize_entity(triple.subject)
        o_key = normalize_entity(triple.object)
        labels.setdefault(s_key, canonical_label(triple.subject))
        labels.setdefault(o_key, canonical_label(triple.object))
        graph.add_node(s_key, label=labels[s_key])
        graph.add_node(o_key, label=labels[o_key])
        graph.add_edge(
            s_key,
            o_key,
            key=normalize_relation(triple.relation),
            relation=normalize_relation(triple.relation),
            source_id=triple.source_id,
        )
    return graph


def neo4j_config() -> tuple[str, str, str]:
    return (
        os.getenv("NEO4J_URI", DEFAULT_NEO4J_URI),
        os.getenv("NEO4J_USER", DEFAULT_NEO4J_USER),
        os.getenv("NEO4J_PASSWORD", DEFAULT_NEO4J_PASSWORD),
    )


def write_triples_to_neo4j(triples: list[Triple]) -> None:
    from neo4j import GraphDatabase

    uri, user, password = neo4j_config()
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        session.run("MATCH (n:Lab19Entity) DETACH DELETE n")
        session.run("MATCH (d:Lab19Document) DETACH DELETE d")
        session.run(
            "CREATE CONSTRAINT lab19_entity_name IF NOT EXISTS "
            "FOR (n:Lab19Entity) REQUIRE n.name IS UNIQUE"
        )
        session.run(
            "CREATE CONSTRAINT lab19_document_id IF NOT EXISTS "
            "FOR (d:Lab19Document) REQUIRE d.id IS UNIQUE"
        )
        for triple in triples:
            relation = normalize_relation(triple.relation)
            query = (
                "MERGE (s:Lab19Entity {name: $subject}) "
                "MERGE (o:Lab19Entity {name: $object}) "
                "MERGE (d:Lab19Document {id: $source_id}) "
                f"MERGE (s)-[r:`{relation}`]->(o) "
                "SET r.source_id = $source_id "
                "MERGE (s)-[:MENTIONED_IN]->(d) "
                "MERGE (o)-[:MENTIONED_IN]->(d)"
            )
            session.run(
                query,
                subject=canonical_label(triple.subject),
                object=canonical_label(triple.object),
                source_id=triple.source_id,
            )
    driver.close()


def neo4j_context(
    center: str,
    hops: int = 2,
    max_edges: int = MAX_GRAPH_CONTEXT_EDGES,
) -> str:
    from neo4j import GraphDatabase

    uri, user, password = neo4j_config()
    driver = GraphDatabase.driver(uri, auth=(user, password))
    query = (
        "MATCH (c:Lab19Entity {name: $center}) "
        f"MATCH p=(c)-[*1..{hops}]-(n:Lab19Entity) "
        "UNWIND relationships(p) AS r "
        "WITH DISTINCT startNode(r) AS s, type(r) AS rel, endNode(r) AS o "
        "WHERE s:Lab19Entity AND o:Lab19Entity "
        "RETURN s.name AS subject, rel AS relation, o.name AS object "
        "ORDER BY subject, relation, object "
        "LIMIT $max_edges"
    )
    with driver.session() as session:
        rows = list(session.run(query, center=center, max_edges=max_edges))
    driver.close()
    return "\n".join(
        f"{row['subject']} --{row['relation']}--> {row['object']}" for row in rows
    )


def graph_to_context(
    graph: nx.MultiDiGraph,
    center: str,
    hops: int = 2,
    max_edges: int = MAX_GRAPH_CONTEXT_EDGES,
) -> str:
    center_key = normalize_entity(center)
    if center_key not in graph:
        return ""
    neighborhood = nx.single_source_shortest_path_length(
        graph.to_undirected(), center_key, cutoff=hops
    )
    nodes = set(neighborhood)
    lines: list[str] = []
    for u, v, data in graph.edges(data=True):
        if u in nodes and v in nodes:
            s_label = graph.nodes[u]["label"]
            o_label = graph.nodes[v]["label"]
            lines.append(f"{s_label} --{data['relation']}--> {o_label}")
    return "\n".join(sorted(set(lines))[:max_edges])


def find_entity_heuristic(question: str, graph: nx.MultiDiGraph) -> str:
    q = question.casefold()
    labels = [data["label"] for _, data in graph.nodes(data=True)]
    labels.sort(key=len, reverse=True)
    for label in labels:
        if label.casefold() in q:
            return label
    tokens = set(re.findall(r"[A-Za-z][A-Za-z0-9]+", q))
    best = ""
    best_score = 0
    for label in labels:
        label_tokens = set(re.findall(r"[A-Za-z][A-Za-z0-9]+", label.casefold()))
        score = len(tokens & label_tokens)
        if score > best_score:
            best = label
            best_score = score
    return best


def find_entity_with_openai(
    client: OpenAI,
    model: str,
    question: str,
    graph: nx.MultiDiGraph,
    usage: Usage,
) -> str:
    labels = sorted(data["label"] for _, data in graph.nodes(data=True))
    instructions = (
        "Select the single best knowledge graph entity for a question. "
        "Return only JSON with key entity. The entity must be copied exactly "
        "from the candidate list. If unsure, choose the closest entity."
    )
    prompt = (
        f"Question: {question}\n\n"
        f"Candidate entities:\n{json.dumps(labels, ensure_ascii=False)}\n\n"
        'Return: {"entity":"..."}'
    )
    raw = call_openai_text(client, model, instructions, prompt, usage, max_output_tokens=200)
    try:
        entity = canonical_label(str(parse_json_object(raw).get("entity", "")))
    except Exception:
        entity = ""
    if normalize_entity(entity) in graph:
        return entity
    return find_entity_heuristic(question, graph)


def answer_from_context_with_openai(
    client: OpenAI,
    model: str,
    question: str,
    context: str,
    usage: Usage,
) -> str:
    instructions = (
        "Answer questions using only the provided context. Be concise. "
        "If the context is insufficient, say what is missing."
    )
    prompt = f"Context:\n{context or '(empty)'}\n\nQuestion: {question}"
    return call_openai_text(client, model, instructions, prompt, usage, max_output_tokens=450)


def offline_answer(question: str, context: str) -> str:
    if not context:
        return "No graph/vector context was found."
    facts = context.splitlines()[:8]
    return "Offline demo context facts: " + "; ".join(facts)


def embed_texts(
    client: OpenAI,
    model: str,
    texts: list[str],
    usage: Usage,
) -> np.ndarray:
    response = client.embeddings.create(model=model, input=texts)
    usage.add_embeddings(response)
    vectors = np.array([item.embedding for item in response.data], dtype=np.float32)
    return normalize_vectors(vectors)


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def rank_chunks_numpy(query_vec: np.ndarray, doc_vecs: np.ndarray, top_k: int) -> list[int]:
    scores = doc_vecs @ query_vec[0]
    return list(np.argsort(-scores)[:top_k])


def rank_chunks_faiss(query_vec: np.ndarray, doc_vecs: np.ndarray, top_k: int) -> list[int] | None:
    try:
        import faiss  # type: ignore
    except Exception:
        return None
    index = faiss.IndexFlatIP(doc_vecs.shape[1])
    index.add(doc_vecs)
    _, indices = index.search(query_vec, top_k)
    return [int(i) for i in indices[0] if i >= 0]


def flat_context(
    client: OpenAI | None,
    embedding_model: str,
    docs: list[CorpusDoc],
    question: str,
    usage: Usage,
    offline_demo: bool,
    top_k: int = 3,
) -> str:
    if offline_demo:
        terms = set(re.findall(r"[A-Za-z][A-Za-z0-9]+", question.casefold()))
        scored: list[tuple[int, CorpusDoc]] = []
        for doc in docs:
            doc_terms = set(re.findall(r"[A-Za-z][A-Za-z0-9]+", doc.text.casefold()))
            scored.append((len(terms & doc_terms), doc))
        best = [doc for score, doc in sorted(scored, key=lambda x: -x[0])[:top_k] if score > 0]
        return "\n\n".join(f"[{doc.id}] {doc.text}" for doc in best)

    assert client is not None
    doc_texts = [doc.text for doc in docs]
    doc_vecs = embed_texts(client, embedding_model, doc_texts, usage)
    query_vec = embed_texts(client, embedding_model, [question], usage)
    indices = rank_chunks_faiss(query_vec, doc_vecs, top_k)
    if indices is None:
        indices = rank_chunks_numpy(query_vec, doc_vecs, top_k)
    return "\n\n".join(f"[{docs[i].id}] {docs[i].text}" for i in indices)


def answer_graph_question(
    client: OpenAI | None,
    model: str,
    graph: nx.MultiDiGraph,
    question: str,
    usage: Usage,
    offline_demo: bool,
    use_neo4j: bool,
) -> tuple[str, str, str]:
    if offline_demo:
        entity = find_entity_heuristic(question, graph)
        context = neo4j_context(entity) if use_neo4j else graph_to_context(graph, entity)
        return entity, context, offline_answer(question, context)

    assert client is not None
    entity = find_entity_with_openai(client, model, question, graph, usage)
    context = neo4j_context(entity) if use_neo4j else graph_to_context(graph, entity)
    answer = answer_from_context_with_openai(client, model, question, context, usage)
    return entity, context, answer


def visualize_graph(graph: nx.MultiDiGraph, output_path: Path) -> None:
    plt.figure(figsize=(18, 12))
    pos = nx.spring_layout(graph.to_undirected(), seed=19, k=0.8)
    node_sizes = [650 + 120 * graph.degree(node) for node in graph.nodes]
    labels = {node: data["label"] for node, data in graph.nodes(data=True)}
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_color="#9BD4D1",
        edgecolors="#1F2937",
        linewidths=0.8,
        node_size=node_sizes,
    )
    nx.draw_networkx_edges(
        graph,
        pos,
        arrows=True,
        arrowstyle="-|>",
        width=0.8,
        alpha=0.35,
        edge_color="#374151",
        connectionstyle="arc3,rad=0.08",
    )
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8)
    plt.title("Tech Company Knowledge Graph", fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close()


def write_triples(triples: list[Triple], path: Path) -> None:
    path.write_text(
        json.dumps([asdict(t) for t in triples], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_evaluation(rows: list[dict[str, str]], csv_path: Path, md_path: Path) -> None:
    fieldnames = [
        "question",
        "graph_entity",
        "flat_answer",
        "graph_answer",
        "graph_context_size",
        "notes",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "| # | Question | Flat RAG | GraphRAG | Notes |",
        "|---:|---|---|---|---|",
    ]
    for idx, row in enumerate(rows, start=1):
        lines.append(
            "| {idx} | {question} | {flat} | {graph} | {notes} |".format(
                idx=idx,
                question=escape_md(row["question"]),
                flat=escape_md(row["flat_answer"]),
                graph=escape_md(row["graph_answer"]),
                notes=escape_md(row["notes"]),
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def escape_md(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def write_usage(usage: Usage, path: Path) -> None:
    payload = asdict(usage)
    payload["estimated_cost_usd"] = round(usage.estimated_cost_usd(), 6)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    load_dotenv()
    start = time.perf_counter()
    OUTPUT_DIR.mkdir(exist_ok=True)

    text_model = os.getenv("OPENAI_TEXT_MODEL", DEFAULT_TEXT_MODEL)
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    usage = Usage()

    corpus_path = Path(args.corpus_path)
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus file not found: {corpus_path}. "
            "Run: python scrape_wikipedia_ai_companies.py --limit 100"
        )
    docs = load_corpus(corpus_path)
    questions = load_questions(DATA_DIR / "benchmark_questions.json")
    if args.limit_questions:
        questions = questions[: args.limit_questions]

    client: OpenAI | None = None
    if not args.offline_demo:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is required for real lab runs. "
                "Use --offline-demo only for local smoke testing."
            )
        client = OpenAI()
        if args.reuse_triples:
            triples = load_cached_triples(Path(args.reuse_triples))
        else:
            triples = extract_triples_with_openai(client, docs, text_model, usage)
    else:
        triples = load_cached_triples(DATA_DIR / "cached_triples.json")

    triples = dedupe_triples(triples)
    graph = build_graph(triples)
    write_triples(triples, OUTPUT_DIR / "triples.json")
    visualize_graph(graph, OUTPUT_DIR / "knowledge_graph.png")
    if not args.skip_neo4j:
        write_triples_to_neo4j(triples)

    rows: list[dict[str, str]] = []
    for question in questions:
        flat_ctx = flat_context(
            client,
            embedding_model,
            docs,
            question,
            usage,
            args.offline_demo,
            top_k=args.top_k,
        )
        if args.offline_demo:
            flat_answer = offline_answer(question, flat_ctx)
        else:
            assert client is not None
            flat_answer = answer_from_context_with_openai(
                client, text_model, question, flat_ctx, usage
            )

        entity, graph_ctx, graph_answer = answer_graph_question(
            client,
            text_model,
            graph,
            question,
            usage,
            args.offline_demo,
            not args.skip_neo4j,
        )
        rows.append(
            {
                "question": question,
                "graph_entity": entity,
                "flat_answer": flat_answer,
                "graph_answer": graph_answer,
                "graph_context_size": str(len(graph_ctx.splitlines())),
                "notes": "Review whether Flat RAG missed a multi-hop relation.",
            }
        )

    usage.seconds = round(time.perf_counter() - start, 3)
    write_evaluation(rows, OUTPUT_DIR / "evaluation.csv", OUTPUT_DIR / "evaluation.md")
    write_usage(usage, OUTPUT_DIR / "token_usage.json")

    print(f"Done. Graph nodes={graph.number_of_nodes()} edges={graph.number_of_edges()}")
    print(f"Wrote outputs to: {OUTPUT_DIR}")
    print(f"Estimated API cost: ${usage.estimated_cost_usd():.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lab Day 19 GraphRAG pipeline")
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run the pipeline with the real OpenAI API.",
    )
    parser.add_argument(
        "--offline-demo",
        action="store_true",
        help="Use cached triples and non-LLM answers for local smoke testing only.",
    )
    parser.add_argument(
        "--limit-questions",
        type=int,
        default=0,
        help="Limit benchmark questions for cheaper debugging.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Flat RAG retrieval count.")
    parser.add_argument(
        "--corpus-path",
        type=Path,
        default=DEFAULT_CORPUS_PATH,
        help="JSONL corpus path. Default is the 100-article Wikipedia corpus.",
    )
    parser.add_argument(
        "--reuse-triples",
        type=Path,
        default=None,
        help="Reuse an existing triples JSON file instead of extracting triples again.",
    )
    parser.add_argument(
        "--skip-neo4j",
        action="store_true",
        help="Skip Neo4j writes/traversal and use NetworkX only. Useful for smoke tests.",
    )
    args = parser.parse_args()
    if not args.run and not args.offline_demo:
        parser.error("Choose --run for OpenAI or --offline-demo for smoke testing.")
    if args.run and args.offline_demo:
        parser.error("Choose only one of --run or --offline-demo.")
    return args


if __name__ == "__main__":
    run(parse_args())
