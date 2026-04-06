"""
RAGSearch - Busca na base de conhecimento local.
Usa ChromaDB para armazenar e recuperar documentos relevantes.

Como adicionar conhecimento:
  from tools.rag import RAGSearch
  rag = RAGSearch()
  rag.add_text("Meu projeto X funciona assim...", source="projetos")
  rag.add_url("https://exemplo.com/artigo")
  rag.add_file("/caminho/para/documento.txt")
"""

import os
from core.config import Config


class RAGSearch:
    def __init__(self):
        self.config = Config()
        self._collection = None
        self._setup()

    def _setup(self):
        """Inicializa o ChromaDB."""
        try:
            import chromadb
            from chromadb.config import Settings

            os.makedirs(self.config.RAG_DB_PATH, exist_ok=True)

            client = chromadb.PersistentClient(path=self.config.RAG_DB_PATH)
            self._collection = client.get_or_create_collection(
                name="companion_knowledge",
                metadata={"hnsw:space": "cosine"}
            )
            count = self._collection.count()
            print(f"[RAG] Base de conhecimento carregada. {count} documento(s).")
        except ImportError:
            print("[RAG] ChromaDB não instalado. Execute: pip install chromadb")
        except Exception as e:
            print(f"[RAG] Erro ao inicializar: {e}")

    def search(self, query: str) -> str:
        """Busca os trechos mais relevantes para a query."""
        if self._collection is None:
            return ""

        try:
            count = self._collection.count()
            if count == 0:
                return "A base de conhecimento está vazia. Use rag.add_text() para adicionar conteúdo."

            results = self._collection.query(
                query_texts=[query],
                n_results=min(self.config.RAG_TOP_K, count)
            )

            docs = results.get("documents", [[]])[0]
            if not docs:
                return ""

            return "\n---\n".join(docs)
        except Exception as e:
            print(f"[RAG] Erro na busca: {e}")
            return ""

    def add_text(self, text: str, source: str = "manual", chunk: bool = True):
        """Adiciona texto à base de conhecimento."""
        if self._collection is None:
            return

        chunks = self._split_text(text) if chunk else [text]

        ids = []
        docs = []
        metas = []
        for i, chunk in enumerate(chunks):
            ids.append(f"{source}_{i}_{hash(chunk)}")
            docs.append(chunk)
            metas.append({"source": source})

        self._collection.upsert(ids=ids, documents=docs, metadatas=metas)
        print(f"[RAG] Adicionados {len(chunks)} trecho(s) de '{source}'.")

    def add_url(self, url: str):
        """Baixa uma página web e adiciona à base."""
        try:
            import requests
            from bs4 import BeautifulSoup

            resp = requests.get(url, timeout=10)
            soup = BeautifulSoup(resp.text, "html.parser")

            # Remove scripts e estilos
            for tag in soup(["script", "style", "nav", "footer"]):
                tag.decompose()

            text = soup.get_text(separator="\n", strip=True)
            self.add_text(text, source=url)
        except ImportError:
            print("[RAG] Instale beautifulsoup4: pip install beautifulsoup4")
        except Exception as e:
            print(f"[RAG] Erro ao processar URL '{url}': {e}")

    def add_file(self, path: str):
        """Adiciona o conteúdo de um arquivo de texto à base."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            self.add_text(text, source=os.path.basename(path))
        except Exception as e:
            print(f"[RAG] Erro ao ler arquivo '{path}': {e}")

    def _split_text(self, text: str) -> list[str]:
        """Divide o texto em chunks menores."""
        size = self.config.RAG_CHUNK_SIZE
        words = text.split()
        chunks = []
        for i in range(0, len(words), size):
            chunk = " ".join(words[i:i + size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    def list_sources(self) -> list[str]:
        """Lista as fontes cadastradas na base."""
        if self._collection is None:
            return []
        results = self._collection.get(include=["metadatas"])
        sources = set(m["source"] for m in results.get("metadatas", []))
        return sorted(sources)

    def clear(self):
        """Limpa toda a base de conhecimento."""
        if self._collection:
            self._collection.delete(where={"source": {"$ne": ""}})
            print("[RAG] Base de conhecimento limpa.")
