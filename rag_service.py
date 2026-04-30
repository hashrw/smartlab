from __future__ import annotations
from sql_service import SQLService

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import error as urllib_error
from urllib import request as urllib_request

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.llms import MockLLM
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

try:
    from llama_index.retrievers.bm25 import BM25Retriever

    BM25_AVAILABLE = True
except Exception:
    BM25Retriever = None
    BM25_AVAILABLE = False

try:
    from llama_index.core.retrievers import QueryFusionRetriever

    FUSION_AVAILABLE = True
except Exception:
    QueryFusionRetriever = None
    FUSION_AVAILABLE = False

try:
    from sentence_transformers import CrossEncoder

    CROSS_ENCODER_AVAILABLE = True
except Exception:
    CrossEncoder = None
    CROSS_ENCODER_AVAILABLE = False


class RAGService:

    # =========================================================================
    # Inicialización
    # =========================================================================
    def __init__(
        self,
        data_dir: str = "data",
        chunk_size: int = 512,
        chunk_overlap: int = 80,
        similarity_top_k: int = 12,
        use_hybrid: bool = True,
        debug: bool = True,
        embed_model_name: str = "BAAI/bge-small-en-v1.5",
        use_cross_encoder: bool = True,
        cross_encoder_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        default_mode: str = "option_a",
        llm_backend: str = "ollama",
        llm_base_url: str = "http://localhost:11434",
        llm_model_name: str = "mistral",
        llm_timeout_seconds: int = 90,
        llm_max_context_sentences: int = 8,
        integration_mode: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_top_k = similarity_top_k
        self.use_hybrid = use_hybrid
        self.debug = debug
        self.embed_model_name = embed_model_name
        self.use_cross_encoder = use_cross_encoder
        self.cross_encoder_model_name = cross_encoder_model_name

        self.default_mode = default_mode

        self.llm_backend = llm_backend
        self.llm_base_url = llm_base_url.rstrip("/")
        self.llm_model_name = llm_model_name
        self.llm_timeout_seconds = llm_timeout_seconds
        self.llm_max_context_sentences = llm_max_context_sentences

        self.sql_service = SQLService(debug=self.debug)

        self.documents: List[Document] = []
        self.nodes = []
        self.index: Optional[VectorStoreIndex] = None
        self.retriever = None
        self.response_synthesizer = None
        self.cross_encoder = None
        self.cross_encoder_enabled = False
        self.integration_mode = integration_mode

        self._initialize()

    def _initialize(self) -> None:
        """Carga corpus, genera nodos, construye índice y retrievers."""
        self._validate_data_dir()
        self.documents = self._load_documents()
        self.documents = self._normalize_document_text(self.documents)
        self.nodes = self._build_nodes(self.documents)

        Settings.embed_model = HuggingFaceEmbedding(model_name=self.embed_model_name)
        Settings.llm = MockLLM(max_tokens=256)

        self.index = VectorStoreIndex(self.nodes)
        self.retriever = self._build_retriever()
        self.response_synthesizer = None
        self.cross_encoder = self._load_cross_encoder()

        if self.debug:
            print(f"[RAG] data_dir={self.data_dir}")
            print(f"[RAG] documents={len(self.documents)}")
            print(f"[RAG] nodes={len(self.nodes)}")
            print(f"[RAG] embed_model={self.embed_model_name}")
            print("[RAG] llm=MockLLM")
            print(
                "[RAG] hybrid_enabled="
                f"{self.use_hybrid and BM25_AVAILABLE and FUSION_AVAILABLE}"
            )
            print(f"[RAG] cross_encoder_enabled={self.cross_encoder_enabled}")
            if self.cross_encoder_enabled:
                print(f"[RAG] cross_encoder_model={self.cross_encoder_model_name}")

    # =========================================================================
    # Construcción del corpus e índice
    # =========================================================================

    def _validate_data_dir(self) -> None:
        """Verifica que el directorio del corpus existe."""
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"No existe el directorio de corpus: {self.data_dir}"
            )

    def _load_documents(self) -> List[Document]:
        """
        Lee todos los PDF del corpus y crea documentos con metadatos clínicos.
        """
        from pypdf import PdfReader

        documents: List[Document] = []
        pdf_files = list(self.data_dir.rglob("*.pdf"))

        for pdf_path in pdf_files:
            try:
                reader = PdfReader(str(pdf_path))
                text_parts: List[str] = []

                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)

                full_text = "\n".join(text_parts).strip()

                if not full_text or len(full_text) < 500:
                    continue

                metadata = self._extract_clinical_metadata(str(pdf_path))

                documents.append(
                    Document(
                        text=full_text,
                        metadata=metadata,
                        doc_id=str(pdf_path),
                    )
                )

            except Exception as e:
                if self.debug:
                    print(f"[RAG] Error leyendo PDF {pdf_path}: {e}")
                continue

        if self.debug:
            print(f"[RAG] Documentos cargados (texto real): {len(documents)}")

        return documents

    def _normalize_document_text(self, documents: List[Document]) -> List[Document]:
        """
        Limpia artefactos típicos de PDFs antes del chunking.
        """
        normalized_docs: List[Document] = []

        for doc in documents:
            text = doc.text or ""

            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\b\d+\s+\d+\s+obj\b", " ", text, flags=re.IGNORECASE)
            text = re.sub(r"\bendobj\b", " ", text, flags=re.IGNORECASE)
            text = re.sub(r"\bstream\b", " ", text, flags=re.IGNORECASE)
            text = re.sub(r"\bendstream\b", " ", text, flags=re.IGNORECASE)

            text = re.sub(
                r"(Producer|CreatorTool|MetadataDate|CreateDate|ModifyDate|pdf:|xmp:|dc:|rdf:|xmlns).*",
                " ",
                text,
                flags=re.IGNORECASE,
            )

            text = text.replace("\x00", " ")
            text = text.replace("\r", "\n")
            text = re.sub(r"[^\w\s\.\,\;\:\-\(\)\/%\[\]]{8,}", " ", text)
            text = re.sub(r"\s+", " ", text).strip()

            lowered = text.lower()

            if (
                len(text) < 300
                or "antenna house pdf output library" in lowered
                or "endstream" in lowered
                or "endobj" in lowered
                or "obj stream" in lowered
            ):
                continue

            normalized_docs.append(
                Document(
                    text=text,
                    metadata=doc.metadata,
                    doc_id=getattr(doc, "doc_id", None),
                )
            )

        return normalized_docs

    def _extract_clinical_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Infiera metadatos clínicos a partir de la ruta del PDF
        según la nueva estructura del corpus.
        """
        path = Path(file_path)
        path_str = str(path).replace("\\", "/")
        parts = [p.lower() for p in path_str.split("/")]

        metadata: Dict[str, Any] = {
            "file_name": path.name,
            "file_path": str(path),
            "source": "bibliography",
            "domain": "gvhd",
            "source_type": "clinical_pdf",
            "block": "unknown",
            "diagnosis_type": None,
            "organ": None,
            "doc_category": "unknown",
            "year": None,
        }

        try:
            metadata["year"] = int(path.name[:4])
        except Exception:
            metadata["year"] = None

        organ_names = {
            "gastrointestinal",
            "liver",
            "skin",
            "genital",
            "ocular",
            "oral",
            "pulmonary",
            "musculoskeletal",
        }

        if "diagnosis" in parts:
            idx = parts.index("diagnosis")
            metadata["block"] = "diagnosis"
            metadata["doc_category"] = "diagnostic_review"

            if len(parts) > idx + 1:
                next_part = parts[idx + 1]
                if next_part in {"acute", "chronic"}:
                    metadata["diagnosis_type"] = next_part

        else:
            for organ in organ_names:
                if organ in parts:
                    metadata["block"] = "organ"
                    metadata["organ"] = organ
                    metadata["doc_category"] = "organ_review"
                    break

        return metadata

    def _build_nodes(self, documents: List[Document]):
        """
        Fragmenta documentos en chunks con solape controlado.
        """
        Settings.chunk_size = self.chunk_size
        Settings.chunk_overlap = self.chunk_overlap

        try:
            pipeline = IngestionPipeline(
                transformations=[
                    TokenTextSplitter(
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.chunk_overlap,
                        separator=" ",
                    )
                ]
            )

            nodes = pipeline.run(documents=documents)

            if self.debug:
                print(f"[RAG] Nodos generados: {len(nodes)}")

            return nodes

        except Exception as exc:
            raise RuntimeError(f"Error generando nodos del corpus: {exc}") from exc

    def _build_retriever(self):
        """
        Construye retriever vectorial puro o híbrido vector + BM25.
        """
        if self.index is None:
            raise RuntimeError("El índice no está inicializado")

        vector_retriever = self.index.as_retriever(
            similarity_top_k=self.similarity_top_k
        )

        if not self.use_hybrid or not BM25_AVAILABLE or not FUSION_AVAILABLE:
            if self.debug:
                print("[RAG] Usando retrieval vectorial puro")
            return vector_retriever

        try:
            bm25_retriever = BM25Retriever.from_defaults(
                docstore=self.index.docstore,
                similarity_top_k=self.similarity_top_k,
            )

            hybrid_retriever = QueryFusionRetriever(
                [vector_retriever, bm25_retriever],
                similarity_top_k=self.similarity_top_k,
                num_queries=1,
                mode="relative_score",
                use_async=False,
                retriever_weights=[0.7, 0.3],
            )

            if self.debug:
                print("[RAG] Usando retrieval híbrido vector + BM25")

            return hybrid_retriever

        except Exception as exc:
            if self.debug:
                print(f"[RAG] Fallback a vectorial puro por error en híbrido: {exc}")
            return vector_retriever

    def _load_cross_encoder(self):
        """
        Carga el cross-encoder usado como reranker final.
        """
        if not self.use_cross_encoder or not CROSS_ENCODER_AVAILABLE:
            return None

        try:
            model = CrossEncoder(self.cross_encoder_model_name)
            self.cross_encoder_enabled = True
            return model
        except Exception as exc:
            self.cross_encoder_enabled = False
            if self.debug:
                print(f"[RAG] No se pudo cargar cross-encoder: {exc}")
            return None

    # =========================================================================
    # API pública
    # =========================================================================

    def query(
        self,
        query: str,
        mode: Optional[str] = None,
        paciente_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Punto de entrada público.

        Modos:
        - option_a: retrieval + scoring + selección + extracción heurística
        - option_b: retrieval + scoring + selección + contexto limpio + LLM

        Si no se indica modo, usa self.default_mode.
        """
        selected_mode = (mode or self.default_mode or "option_a").lower()

        if selected_mode == "option_a":
            return self.query_option_a(query)

        if selected_mode == "option_b":
            return self.query_option_b(query, paciente_id=paciente_id)

        raise ValueError(
            f"Modo no soportado: {selected_mode}. Usa 'option_a' o 'option_b'."
        )

    def query_option_a(self, query: str) -> Dict[str, Any]:
        """
        Opción A:
        Retrieval clínico controlado sin LLM.
        """
        if not query or not str(query).strip():
            raise ValueError("La query no puede estar vacía")

        # raw_nodes = self._retrieve_multiquery(query)
        raw_nodes = (
            self._retrieve(query)
            if self.integration_mode
            else self._retrieve_multiquery(query)
        )
        cleaned_entries = self._postprocess_retrieved_nodes(query, raw_nodes)

        sources = self._build_sources(cleaned_entries)
        answer = self._fallback_answer(cleaned_entries, query)

        return {
            "mode": "option_a",
            "answer": answer,
            "sources": sources,
        }

    def query_option_b(
        self,
        query: str,
        paciente_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Opción B:
        Reutiliza el retrieval validado de Opción A y añade una capa LLM
        para síntesis clínica final.
        """
        if not query or not str(query).strip():
            raise ValueError("La query no puede estar vacía")

        if self.debug:
            print("\n================ OPTION B START ================")
            print("[DEBUG] mode=option_b")
            print("[DEBUG] query:", query)
            print("[DEBUG] paciente_id:", paciente_id)

        # raw_nodes = self._retrieve_multiquery(query)
        raw_nodes = (
            self._retrieve(query)
            if self.integration_mode
            else self._retrieve_multiquery(query)
        )

        if self.debug:
            print("\n================ RAW NODES DEBUG ================")
            print("[DEBUG] query:", query)
            print("[DEBUG] total_raw_nodes:", len(raw_nodes))

            for i, node in enumerate(raw_nodes[:12], 1):
                try:
                    text = ""
                    metadata = {}
                    score = getattr(node, "score", None)

                    if hasattr(node, "node"):
                        text = getattr(node.node, "text", "") or ""
                        metadata = getattr(node.node, "metadata", {}) or {}
                    else:
                        text = getattr(node, "text", "") or ""
                        metadata = getattr(node, "metadata", {}) or {}

                    print(f"\n--- RAW NODE #{i} ---")
                    print("score:", score)
                    print("file_name:", metadata.get("file_name"))
                    print("organ:", metadata.get("organ"))
                    print("block:", metadata.get("block"))
                    print("diagnosis_type:", metadata.get("diagnosis_type"))
                    print("doc_category:", metadata.get("doc_category"))
                    print("year:", metadata.get("year"))
                    print("text:", text[:300].replace("\n", " "))
                except Exception as e:
                    print(f"[DEBUG] Error leyendo raw node {i}:", e)

        cleaned_entries = self._postprocess_retrieved_nodes(query, raw_nodes)

        if self.debug:
            print(
                "\n================ QUERY OPTION B CLEANED ENTRIES DEBUG ================"
            )
            print("[DEBUG] cleaned_entries_count:", len(cleaned_entries))
            for i, entry in enumerate(cleaned_entries[:12], 1):
                meta = entry.get("meta", {}) or {}
                print(f"[DEBUG] cleaned_entry[{i}] file_name:", meta.get("file_name"))
                print(f"[DEBUG] cleaned_entry[{i}] block:", meta.get("block"))
                print(
                    f"[DEBUG] cleaned_entry[{i}] diagnosis_type:",
                    meta.get("diagnosis_type"),
                )
                print(f"[DEBUG] cleaned_entry[{i}] organ:", meta.get("organ"))
                print(
                    f"[DEBUG] cleaned_entry[{i}] final_score:",
                    entry.get("final_score"),
                )

        sources = self._build_sources(cleaned_entries)

        query_lower = query.lower()
        organ_focus_terms = {
            "skin": ["skin", "cutaneous", "rash", "erythema"],
            "gastrointestinal": [
                "gastrointestinal",
                "gi",
                "gut",
                "diarrhea",
                "abdominal pain",
                "nausea",
                "vomiting",
            ],
            "liver": ["liver", "bilirubin", "jaundice", "hepatic"],
            "ocular": ["ocular", "eye", "dry eye", "keratoconjunctivitis"],
            "oral": ["oral", "mouth", "mucosa", "buccal", "tongue", "lip", "gingiva"],
            "pulmonary": ["pulmonary", "lung", "bronchiolitis"],
        }

        requested_organs: List[str] = []
        for organ_name, terms in organ_focus_terms.items():
            if any(term in query_lower for term in terms):
                requested_organs.append(organ_name)

        is_specific_query = len(requested_organs) > 0

        if not cleaned_entries:
            if self.debug:
                print(
                    "\n================ OPTION B END EMPTY RETRIEVAL ================"
                )
                print("[DEBUG] llm_used:", False)
                print(
                    "[DEBUG] fallback_reason:",
                    (
                        "no_retrieved_entries_specific_query"
                        if is_specific_query
                        else "no_retrieved_entries"
                    ),
                )
                print("[DEBUG] sources_count:", len(sources))

            if is_specific_query:
                return {
                    "mode": "option_b",
                    "answer": "The current corpus does not contain direct evidence specific to the requested organ or manifestation.",
                    "sources": sources,
                    "llm_used": False,
                    "fallback_reason": "no_retrieved_entries_specific_query",
                    "paciente_id": paciente_id,
                }

            return {
                "mode": "option_b",
                "answer": self._fallback_answer(cleaned_entries, query),
                "sources": sources,
                "llm_used": False,
                "fallback_reason": "no_retrieved_entries",
                "paciente_id": paciente_id,
            }

        llm_context = self._build_combined_llm_context(
            query=query,
            retrieved_entries=cleaned_entries,
            paciente_id=paciente_id,
        )

        if self.debug:
            print(
                "\n================ QUERY OPTION B LLM CONTEXT DEBUG ================"
            )
            print("[DEBUG] llm_context_is_empty:", not bool(llm_context.strip()))
            print("[DEBUG] llm_context_length:", len(llm_context or ""))
            print("[DEBUG] llm_context_preview:", (llm_context or "")[:2000])

        if not llm_context.strip():
            if self.debug:
                print("\n================ OPTION B END EMPTY CONTEXT ================")
                print("[DEBUG] llm_used:", False)
                print(
                    "[DEBUG] fallback_reason:",
                    (
                        "empty_llm_context_specific_query"
                        if is_specific_query
                        else "empty_llm_context"
                    ),
                )
                print("[DEBUG] sources_count:", len(sources))

            if is_specific_query:
                return {
                    "mode": "option_b",
                    "answer": "The current corpus does not contain direct evidence specific to the requested organ or manifestation.",
                    "sources": sources,
                    "llm_used": False,
                    "fallback_reason": "empty_llm_context_specific_query",
                    "paciente_id": paciente_id,
                }

            return {
                "mode": "option_b",
                "answer": self._fallback_answer(cleaned_entries, query),
                "sources": sources,
                "llm_used": False,
                "fallback_reason": "empty_llm_context",
                "paciente_id": paciente_id,
            }

        prompt = self._build_llm_prompt(query, llm_context)

        if self.debug:
            print("\n================ PROMPT DEBUG ================")
            print("[DEBUG] prompt_length:", len(prompt or ""))
            print("[DEBUG] prompt_preview:", (prompt or "")[:2500])

        try:
            if self.debug:
                print("\n================ LLM CALL DEBUG ================")
                print("[DEBUG] llm_call=true")
                print("[DEBUG] llm_backend:", self.llm_backend)
                print("[DEBUG] llm_model_name:", self.llm_model_name)
                print("[DEBUG] cleaned_entries_count_for_llm:", len(cleaned_entries))
                print("[DEBUG] sources_count_for_llm:", len(sources))

            llm_answer = self._call_llm(prompt)

            if self.debug:
                print("\n================ LLM RESPONSE DEBUG ================")
                print("[DEBUG] llm_answer_length:", len(llm_answer or ""))
                print("[DEBUG] llm_answer_preview:", (llm_answer or "")[:2000])

            if not llm_answer.strip():
                raise RuntimeError("El LLM devolvió una respuesta vacía")

            if self.debug:
                print("\n================ OPTION B END SUCCESS ================")
                print("[DEBUG] llm_used:", True)
                print("[DEBUG] sources_count:", len(sources))
                print("[DEBUG] paciente_id:", paciente_id)

            return {
                "mode": "option_b",
                "answer": llm_answer.strip(),
                "sources": sources,
                "llm_used": True,
                "llm_model": self.llm_model_name,
                "paciente_id": paciente_id,
            }

        except Exception as exc:
            if self.debug:
                print(f"[RAG] Fallback a Opción A por error en LLM: {exc}")
                print("\n================ OPTION B END LLM EXCEPTION ================")
                print("[DEBUG] llm_used:", False)
                print("[DEBUG] exception:", str(exc))
                print("[DEBUG] sources_count:", len(sources))

            if is_specific_query:
                return {
                    "mode": "option_b",
                    "answer": "A direct answer for the requested organ could not be generated from the current evidence context.",
                    "sources": sources,
                    "llm_used": False,
                    "fallback_reason": str(exc),
                    "paciente_id": paciente_id,
                }

            return {
                "mode": "option_b",
                "answer": self._fallback_answer(cleaned_entries, query),
                "sources": sources,
                "llm_used": False,
                "fallback_reason": str(exc),
                "paciente_id": paciente_id,
            }

    def generate_clinical_report(
        self,
        caso_clinico: Dict[str, Any],
        resultado_inferencia: Optional[Dict[str, Any]] = None,
        paciente_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Genera un informe clínico estructurado a partir del caso clínico recibido
        desde Laravel.

        En este flujo de entrega:
            - Laravel es la fuente de verdad clínica.
            - Flask no consulta SQL.
            - El RAG usa el caso clínico para construir una consulta interna.
            - El LLM devuelve un informe estructurado en español.
        """

        if not isinstance(caso_clinico, dict):
            raise ValueError("caso_clinico debe ser un objeto JSON válido")

        retrieval_query = self._build_retrieval_query_from_case(caso_clinico)

        raw_nodes = (
            self._retrieve(retrieval_query)
            if self.integration_mode
            else self._retrieve_multiquery(retrieval_query)
        )

        cleaned_entries = self._postprocess_retrieved_nodes(
            retrieval_query,
            raw_nodes,
        )

        sources = self._build_sources(cleaned_entries)

        literature_context = self._build_llm_context(
            retrieval_query,
            cleaned_entries,
        )

        if not literature_context.strip():
            return {
                "mode": "clinical_report",
                "clinical_report": self._fallback_clinical_report(
                    caso_clinico=caso_clinico,
                    reason="No se ha podido construir contexto bibliográfico suficiente.",
                ),
                "sources": sources,
                "warnings": [
                    "No se ha podido construir contexto bibliográfico suficiente."
                ],
                "llm_used": False,
                "fallback_reason": "empty_literature_context",
                "paciente_id": paciente_id,
            }

        prompt = self._build_clinical_report_prompt(
            caso_clinico=caso_clinico,
            resultado_inferencia=resultado_inferencia,
            literature_context=literature_context,
        )

        try:
            llm_answer = self._call_llm(prompt)

            if not llm_answer.strip():
                raise RuntimeError("El LLM devolvió una respuesta vacía")

            parsed_report = self._parse_clinical_report_json(llm_answer)

            return {
                "mode": "clinical_report",
                "clinical_report": parsed_report,
                "sources": sources,
                "warnings": [],
                "llm_used": True,
                "llm_model": self.llm_model_name,
                "paciente_id": paciente_id,
            }

        except Exception as exc:
            return {
                "mode": "clinical_report",
                "clinical_report": self._fallback_clinical_report(
                    caso_clinico=caso_clinico,
                    reason=str(exc),
                ),
                "sources": sources,
                "warnings": [str(exc)],
                "llm_used": False,
                "fallback_reason": str(exc),
                "paciente_id": paciente_id,
            }

    # helpers de clinical report
    def _build_retrieval_query_from_case(self, caso_clinico: Dict[str, Any]) -> str:
        """
        Construye una consulta interna en inglés a partir del caso clínico recibido.
        No es una query dinámica de usuario, sino una normalización técnica para retrieval.
        """
        sintomas = caso_clinico.get("active_aliases_canonical", []) or []
        organos = caso_clinico.get("organo_score_nih_by_nombre", {}) or {}

        alias_map = {
            "o1_diarrea_acuosa": "watery diarrhea",
            "o1_diarrea_con_sangre": "bloody diarrhea",
            "o1_dolor_abdominal": "abdominal pain",
            "o1_nauseas": "nausea",
            "o1_vomitos": "vomiting",
            "o1_anorexia": "anorexia",
            "o2_alt_elevada": "elevated ALT",
            "o2_fosfatasa_alcalina_elevada": "elevated alkaline phosphatase",
            "o2_hiperbilirrubinemia": "hyperbilirubinemia",
            "o7_exantema_maculopapular": "maculopapular rash",
        }

        organ_map = {
            "Hígado": "liver",
            "Higado": "liver",
            "Tracto gastrointestinal": "gastrointestinal",
            "Piel": "skin",
            "Ojos": "ocular",
            "Boca": "oral",
            "Pulmón": "pulmonary",
            "Pulmon": "pulmonary",
            "Pulmones": "pulmonary",
        }

        symptom_terms = [
            alias_map.get(alias, str(alias).replace("_", " ")) for alias in sintomas[:8]
        ]

        organ_terms = [
            organ_map.get(str(org), str(org)) for org in list(organos.keys())[:6]
        ]

        score_terms = []
        for org, score in list(organos.items())[:6]:
            organ_name = organ_map.get(str(org), str(org))
            score_terms.append(f"{organ_name} NIH score {score}")

        terms = (
            ["graft versus host disease", "GVHD", "clinical manifestations"]
            + symptom_terms
            + organ_terms
            + score_terms
        )

        return (
            "acute graft versus host disease clinical manifestations "
            + " ".join(symptom_terms)
            + " "
            + " ".join(organ_terms)
            + " "
            + " ".join(score_terms)
        ).strip()

    def _build_clinical_report_prompt(
        self,
        caso_clinico: Dict[str, Any],
        resultado_inferencia: Optional[Dict[str, Any]],
        literature_context: str,
    ) -> str:
        """
        Prompt final para generar informe clínico estructurado.
        """

        case_json = json.dumps(
            caso_clinico,
            ensure_ascii=False,
            indent=2,
        )

        inference_json = json.dumps(
            resultado_inferencia or {},
            ensure_ascii=False,
            indent=2,
        )

        return f"""
                You are a clinical assistant specialized in graft-versus-host disease.

                You receive a structured clinical case in Spanish from a deterministic expert system built in Laravel.

                Your task:
                    - Interpret the Spanish clinical JSON.
                    - Internally translate relevant clinical terms into English when reasoning over the English literature context.
                    - Use ONLY the provided literature context as scientific support.
                    - Generate the final answer in Spanish.
                    - Produce a structured clinical report for a physician.
                    - Do not invent data.
                    - Do not diagnose beyond the clinical case and the provided evidence.
                    - If the evidence is insufficient, state it clearly.
                    - The report must help the physician validate the case faster, not replace medical judgement.

            Return ONLY valid JSON.
            Do not use markdown.
            Do not add text outside the JSON.
            Keep the JSON concise. Use short clinical sentences. Do not produce long paragraphs.

            Required JSON structure:

            {{
                "titulo": "",
                "resumen_ejecutivo": "",
                "diagnostico_dss": {
                "tipo_enfermedad": "",
                "grado_eich": "",
                "estado_injerto": "",
                "regla_aplicada": "",
                "interpretacion": ""
            },
                "justificacion_clinica": [
            {
                "organo": "",
                "score_nih": null,
                "hallazgos_del_caso": [],
                "relacion_con_eich": "",
                "nivel_alerta": "bajo | moderado | alto"
            }
            ],
                "evidencia_cientifica": {
                "resumen": "",
                "coherencia_con_el_caso": "",
                "incertidumbres": []
            },
                "alertas_clinicas": [],
                "limitaciones": [],
                "validacion_medica_recomendada": [],
                "conclusion": ""
            }}

            Clinical case JSON:
            {case_json}

            Literature context:
            {literature_context}

            Deterministic DSS inference result:
            {inference_json}
            """

    def _parse_clinical_report_json(self, llm_answer: str) -> Dict[str, Any]:
        """
        Intenta parsear la respuesta del LLM como JSON.
        """
        cleaned = llm_answer.strip()

        cleaned = re.sub(r"^```json", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"^```", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

        try:
            parsed = json.loads(cleaned)
        except Exception as exc:
            raise RuntimeError(
                f"No se pudo parsear el informe clínico como JSON: {exc}"
            )

        if not isinstance(parsed, dict):
            raise RuntimeError("El informe clínico generado no es un objeto JSON")

        return parsed

    def _fallback_clinical_report(
        self,
        caso_clinico: Dict[str, Any],
        reason: str,
    ) -> Dict[str, Any]:
        """
        Informe mínimo controlado si falla el LLM o no hay contexto suficiente.
        """
        return {
            "titulo": "Informe clínico de apoyo DSS-RAG",
            "resumen_clinico": "No se ha podido generar un informe clínico completo con el contexto disponible.",
            "sospecha_diagnostica": None,
            "organos_afectados": [],
            "hallazgos_relevantes": [],
            "evidencia_clinica_resumida": None,
            "limitaciones": [
                reason,
            ],
            "recomendaciones_validacion_medica": [
                "Validar manualmente el caso clínico desde la información estructurada disponible en el sistema experto.",
            ],
            "conclusion": "Informe generado en modo fallback controlado.",
        }

    def debug_query(self, query: str) -> None:
        """
        Muestra el detalle de scoring de los nodos seleccionados.
        """
        results = self._retrieve(query)
        cleaned = self._postprocess_retrieved_nodes(query, results)

        print("\n=== DEBUG NODES ===")
        for i, entry in enumerate(cleaned, start=1):
            meta = entry["meta"]

            print(f"\n[{i}] base_score={round(entry['base_score'], 6)}")
            print(f"heuristic_score={round(entry['heuristic_score'], 6)}")
            print(f"cross_encoder_raw={round(entry['cross_encoder_raw'], 6)}")
            print(f"cross_encoder_score={round(entry['cross_encoder_score'], 6)}")
            print(f"final_score={round(entry['final_score'], 6)}")
            print(f"file_name={meta.get('file_name')}")
            print(f"block={meta.get('block')}")
            print(f"diagnosis_type={meta.get('diagnosis_type')}")
            print(f"organ={meta.get('organ')}")
            print(f"year={meta.get('year')}")
            print(self._clean_preview(entry["text"], 800))

    # =========================================================================
    # 4. Fase 1: expansión y retrieval
    # =========================================================================

    def _expand_clinical_subqueries(self, query: str) -> List[str]:
        """
        Expande una consulta a subconsultas clínicas más concretas.
        """
        q = query.lower()
        subqueries: List[str] = [query]

        organ_map = {
            "skin": ["skin", "cutaneous", "rash", "erythema"],
            "gastrointestinal": [
                "gastrointestinal",
                "gi",
                "gut",
                "diarrhea",
                "abdominal pain",
                "nausea",
                "vomiting",
            ],
            "liver": ["liver", "bilirubin", "jaundice", "hepatic"],
            "ocular": ["ocular", "eye", "dry eye"],
            "oral": ["oral", "mouth"],
            "pulmonary": ["pulmonary", "lung", "bronchiolitis"],
        }

        requested_organs: List[str] = []
        for organ_name, terms in organ_map.items():
            if any(term in q for term in terms):
                requested_organs.append(organ_name)

        is_specific_query = len(requested_organs) > 0

        if is_specific_query:
            for organ in requested_organs:
                if "acute" in q:
                    if organ == "skin":
                        subqueries.append(
                            "acute graft versus host disease skin rash erythema cutaneous manifestations"
                        )
                    elif organ == "gastrointestinal":
                        subqueries.append(
                            "acute graft versus host disease gastrointestinal diarrhea nausea abdominal pain vomiting manifestations"
                        )
                    elif organ == "liver":
                        subqueries.append(
                            "acute graft versus host disease liver bilirubin jaundice hepatic involvement manifestations"
                        )
                    elif organ == "ocular":
                        subqueries.append(
                            "acute graft versus host disease ocular manifestations dry eye vision involvement"
                        )
                    elif organ == "oral":
                        subqueries.append(
                            "acute graft versus host disease oral manifestations mouth ulcers mucositis"
                        )
                    elif organ == "pulmonary":
                        subqueries.append(
                            "acute graft versus host disease pulmonary manifestations lung involvement bronchiolitis"
                        )

                elif "chronic" in q:
                    if organ == "skin":
                        subqueries.append(
                            "chronic graft versus host disease skin sclerosis rash erythema cutaneous manifestations"
                        )
                    elif organ == "gastrointestinal":
                        subqueries.append(
                            "chronic graft versus host disease gastrointestinal manifestations diarrhea abdominal pain nausea"
                        )
                    elif organ == "liver":
                        subqueries.append(
                            "chronic graft versus host disease liver manifestations bilirubin hepatic dysfunction"
                        )
                    elif organ == "ocular":
                        subqueries.append(
                            "chronic graft versus host disease ocular manifestations dry eye keratoconjunctivitis sicca"
                        )
                    elif organ == "oral":
                        subqueries.append(
                            "chronic graft versus host disease oral manifestations lichen planus mouth lesions mucosa"
                        )
                    elif organ == "pulmonary":
                        subqueries.append(
                            "chronic graft versus host disease pulmonary manifestations bronchiolitis obliterans"
                        )
                else:
                    subqueries.append(
                        f"graft versus host disease {organ} clinical manifestations"
                    )
        else:
            if "acute" in q:
                subqueries.extend(
                    [
                        "acute graft versus host disease skin rash erythema manifestations",
                        "acute graft versus host disease gastrointestinal diarrhea nausea abdominal pain vomiting manifestations",
                        "acute graft versus host disease liver bilirubin jaundice hepatic involvement manifestations",
                    ]
                )
            elif "chronic" in q:
                subqueries.extend(
                    [
                        "chronic graft versus host disease skin sclerosis rash erythema cutaneous manifestations",
                        "chronic graft versus host disease oral ocular pulmonary manifestations dry eye mouth bronchiolitis",
                        "chronic graft versus host disease liver gastrointestinal manifestations bilirubin diarrhea abdominal pain",
                    ]
                )
            else:
                subqueries.extend(
                    [
                        "graft versus host disease skin gastrointestinal liver manifestations",
                        "graft versus host disease symptoms rash diarrhea jaundice",
                    ]
                )

        seen = set()
        out: List[str] = []
        for sq in subqueries:
            key = sq.strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(sq)

        if self.debug:
            print("\n================ SUBQUERY EXPANSION DEBUG ================")
            print("[DEBUG] original_query:", query)
            print("[DEBUG] requested_organs:", requested_organs)
            print("[DEBUG] is_specific_query:", is_specific_query)
            print("[DEBUG] expanded_subqueries_count:", len(out))
            for i, sq in enumerate(out, 1):
                print(f"[DEBUG] subquery[{i}]:", sq)

        return out

    def _retrieve(self, query: str):
        """Ejecuta retrieval simple sobre el retriever configurado."""
        if self.retriever is None:
            raise RuntimeError("Retriever no inicializado")
        return self.retriever.retrieve(query)

    def _retrieve_multiquery(self, query: str):
        """
        Ejecuta varias subconsultas y fusiona resultados sin duplicados.
        """
        if self.retriever is None:
            raise RuntimeError("Retriever no inicializado")

        subqueries = self._expand_clinical_subqueries(query)
        merged = []
        seen_ids = set()

        if self.debug:
            print("\n=== RAG QUERY ===")
            print(query)
            print(f"[RAG] subqueries={len(subqueries)}")

        for sq in subqueries:
            if self.debug:
                print(f"[RAG] subquery -> {sq}")

            try:
                results = self.retriever.retrieve(sq)

                if self.debug:
                    print("\n---------------- SUBQUERY RETRIEVE DEBUG ----------------")
                    print("[DEBUG] subquery:", sq)
                    print("[DEBUG] retrieved_count_for_subquery:", len(results))

                    for i, item in enumerate(results[:5], 1):
                        node = item.node if hasattr(item, "node") else item
                        meta = getattr(node, "metadata", {}) or {}
                        text = getattr(node, "text", "") or ""
                        score = getattr(item, "score", None)

                        print(
                            f"[DEBUG] subquery_result[{i}] file_name:",
                            meta.get("file_name"),
                        )
                        print(f"[DEBUG] subquery_result[{i}] block:", meta.get("block"))
                        print(
                            f"[DEBUG] subquery_result[{i}] diagnosis_type:",
                            meta.get("diagnosis_type"),
                        )
                        print(f"[DEBUG] subquery_result[{i}] organ:", meta.get("organ"))
                        print(f"[DEBUG] subquery_result[{i}] score:", score)
                        print(
                            f"[DEBUG] subquery_result[{i}] text:",
                            text[:250].replace("\n", " "),
                        )

            except Exception as exc:
                if self.debug:
                    print(f"[DEBUG] retrieve error for subquery '{sq}': {exc}")
                continue

            for item in results:
                node = item.node if hasattr(item, "node") else item
                node_id = (
                    getattr(node, "node_id", None)
                    or getattr(node, "id_", None)
                    or id(node)
                )

                if node_id in seen_ids:
                    continue

                seen_ids.add(node_id)
                merged.append(item)

        if self.debug:
            print(f"[RAG] retrieved_nodes_raw={len(merged)}")
            print("\n================ MERGED RAW NODES DEBUG ================")
            print("[DEBUG] merged_unique_nodes_count:", len(merged))

        return merged

    # =========================================================================
    # 5. Fase 2: scoring y filtrado
    # =========================================================================

    def _postprocess_retrieved_nodes(self, query: str, retrieved_nodes):
        """
        Puntúa nodos recuperados, filtra basura y aplica selección diversa.
        """
        if self.debug:
            print(">>> USING DIVERSE SELECTION <<<")
            print("\n================ POSTPROCESS INPUT DEBUG ================")
            print("[DEBUG] query:", query)
            print("[DEBUG] retrieved_nodes_input_count:", len(retrieved_nodes))

        scored_entries: List[Dict[str, Any]] = []

        for item in retrieved_nodes:
            node = item.node if hasattr(item, "node") else item
            text = getattr(node, "text", "") or ""
            meta = getattr(node, "metadata", {}) or {}

            base_score = float(getattr(item, "score", 0.0) or 0.0)
            heuristic_score, score_breakdown = self._score_node_for_query(
                query, text, meta
            )

            if self._bibliography_signals(text) >= 4:
                cross_encoder_raw = 0.0
                cross_encoder_score = -2.0
            else:
                cross_encoder_raw = self._cross_encoder_raw_score(query, text)
                cross_encoder_score = self._normalize_cross_encoder_score(
                    cross_encoder_raw
                )

            final_score = base_score + heuristic_score + cross_encoder_score

            entry = {
                "item": item,
                "node": node,
                "text": text,
                "meta": meta,
                "base_score": base_score,
                "heuristic_score": heuristic_score,
                "cross_encoder_raw": cross_encoder_raw,
                "cross_encoder_score": cross_encoder_score,
                "final_score": final_score,
                "score_breakdown": score_breakdown,
            }

            if self.debug:
                chunk_intent = self._classify_chunk_intent_type(text)
                print("\n---------------- NODE SCORING DEBUG ----------------")
                print("[DEBUG] file_name:", meta.get("file_name"))
                print("[DEBUG] block:", meta.get("block"))
                print("[DEBUG] diagnosis_type:", meta.get("diagnosis_type"))
                print("[DEBUG] organ:", meta.get("organ"))
                print("[DEBUG] chunk_intent:", chunk_intent)
                print("[DEBUG] base_score:", base_score)
                print("[DEBUG] heuristic_score:", heuristic_score)
                print("[DEBUG] cross_encoder_raw:", cross_encoder_raw)
                print("[DEBUG] cross_encoder_score:", cross_encoder_score)
                print("[DEBUG] final_score:", final_score)
                print("[DEBUG] score_breakdown:", score_breakdown)
                print("[DEBUG] text:", text[:250].replace("\n", " "))
                print("[DEBUG] is_valid_node:", self._is_valid_node(text, final_score))

            if self._is_valid_node(text, final_score):
                scored_entries.append(entry)

        scored_entries.sort(key=lambda x: x["final_score"], reverse=True)

        filtered = self._select_diverse_chunks(scored_entries, query)
        if not filtered:
            filtered = scored_entries[:3]

        if self.debug:
            print(f"[RAG] retrieved_nodes_clean={len(filtered)}")
            for idx, entry in enumerate(filtered, start=1):
                print(
                    "[RAG] top[{idx}] base={base:.4f} heur={heur:.4f} "
                    "ce_raw={ce_raw:.4f} ce={ce:.4f} final={final:.4f} file={file}".format(
                        idx=idx,
                        base=entry["base_score"],
                        heur=entry["heuristic_score"],
                        ce_raw=entry["cross_encoder_raw"],
                        ce=entry["cross_encoder_score"],
                        final=entry["final_score"],
                        file=entry["meta"].get("file_name"),
                    )
                )

            print("\n================ CLEANED ENTRIES FINAL DEBUG ================")
            print("[DEBUG] cleaned_entries_count:", len(filtered))

            for i, entry in enumerate(filtered[:12], 1):
                meta = entry.get("meta", {}) or {}
                print(f"\n--- CLEANED ENTRY #{i} ---")
                print("file_name:", meta.get("file_name"))
                print("block:", meta.get("block"))
                print("diagnosis_type:", meta.get("diagnosis_type"))
                print("organ:", meta.get("organ"))
                print("doc_category:", meta.get("doc_category"))
                print("base_score:", entry.get("base_score"))
                print("heuristic_score:", entry.get("heuristic_score"))
                print("cross_encoder_raw:", entry.get("cross_encoder_raw"))
                print("cross_encoder_score:", entry.get("cross_encoder_score"))
                print("final_score:", entry.get("final_score"))
                print("score_breakdown:", entry.get("score_breakdown"))
                print("text:", (entry.get("text") or "")[:300].replace("\n", " "))

        return filtered

    def _score_node_for_query(
        self,
        query: str,
        text: str,
        meta: Dict[str, Any],
    ) -> Tuple[float, Dict[str, float]]:
        """
        Score heurístico central.
        """

        if not text:
            return -3.0, {"empty_text": -3.0}

        lowered = text.lower()
        query_lower = query.lower()
        score = 0.0
        breakdown: Dict[str, float] = {}

        hard_bad_patterns = [
            "antenna house pdf output library",
            "endstream",
            "endobj",
            "obj stream",
        ]

        if any(p in lowered for p in hard_bad_patterns):
            return -10.0, {"hard_bad_pattern": -10.0}

        query_asks_acute = "acute" in query_lower
        query_asks_chronic = "chronic" in query_lower
        diag_type = (meta.get("diagnosis_type") or "").lower()
        block = (meta.get("block") or "").lower()
        file_name = (meta.get("file_name") or "").lower()

        if query_asks_acute:
            if diag_type == "acute":
                score += 2.5
                breakdown["acute_match"] = 2.5
            elif diag_type == "chronic":
                score -= 2.0
                breakdown["acute_vs_chronic_penalty"] = -2.0

        if query_asks_chronic:
            if diag_type == "chronic":
                score += 2.5
                breakdown["chronic_match"] = 2.5
            elif diag_type == "acute":
                score -= 2.0
                breakdown["chronic_vs_acute_penalty"] = -2.0

        if block == "diagnosis":
            score += 0.4
            breakdown["diagnosis_block_bonus"] = 0.4

        density_bonus = self._clinical_density_bonus(text)
        score += density_bonus
        breakdown["clinical_density"] = density_bonus

        positive_terms = [
            "acute graft-versus-host disease",
            "clinical manifestations",
            "manifestations",
            "symptoms",
            "skin",
            "liver",
            "gastrointestinal",
            "rash",
            "diarrhea",
            "abdominal pain",
            "nausea",
            "vomiting",
            "jaundice",
            "diagnosis",
            "management",
            "allogeneic hematopoietic stem cell transplantation",
            "incidence",
        ]

        positive_bonus = 0.0
        for term in positive_terms:
            if term in lowered:
                positive_bonus += 0.22

        score += positive_bonus
        if positive_bonus:
            breakdown["positive_terms"] = positive_bonus

        soft_bad_terms = [
            "department of",
            "university of",
            "faculty of medicine",
            "perelman school of medicine",
            "division of hematology",
            "blood and marrow transplant",
            "philadelphia",
            "boston",
            "bethesda",
            "ankara",
            "turkey",
            "doi:",
            "correspondence",
        ]

        soft_penalty = 0.0
        for term in soft_bad_terms:
            if term in lowered:
                soft_penalty -= 1.0

        score += soft_penalty
        if soft_penalty:
            breakdown["soft_bad_terms"] = soft_penalty

        biblio_score = self._bibliography_signals(text)
        biblio_penalty = float(biblio_score) * -0.9
        score += biblio_penalty
        if biblio_penalty:
            breakdown["bibliography_penalty"] = biblio_penalty

        intent = self._detect_query_intent(query)
        chunk_intent = self._classify_chunk_intent_type(text)

        if intent == "manifestations" and block == "organ":
            pathogenesis_signals = [
                "pathogenesis",
                "etiopathogenesis",
                "development of gvhd",
                "cytokines",
                "cytokine receptors",
                "complement activation",
                "donor-derived",
                "mesenchymal",
                "immune response",
            ]

            diagnosis_signals = [
                "diagnosis",
                "update on",
                "diagnostic approach",
            ]

            pathogenesis_hits = sum(1 for s in pathogenesis_signals if s in lowered)
            diagnosis_hits = sum(1 for s in diagnosis_signals if s in lowered)

            organ_penalty = 0.0

            if pathogenesis_hits >= 2:
                organ_penalty -= 0.35
            elif pathogenesis_hits == 1:
                organ_penalty -= 0.15

            if diagnosis_hits >= 2:
                organ_penalty -= 0.30
            elif diagnosis_hits == 1 and pathogenesis_hits >= 1:
                organ_penalty -= 0.15
            elif "diagnosis" in file_name and diagnosis_hits >= 1:
                organ_penalty -= 0.10

            if organ_penalty != 0.0:
                score += organ_penalty
                breakdown["organ_not_manifestation_penalty"] = organ_penalty

        query_specific_organs = {
            "skin": ["skin", "cutaneous", "rash", "erythema"],
            "gastrointestinal": [
                "gastrointestinal",
                "diarrhea",
                "abdominal pain",
                "nausea",
                "vomiting",
                "gut",
                "gi",
            ],
            "liver": ["liver", "bilirubin", "jaundice", "hepatic"],
            "ocular": ["ocular", "eye", "dry eye"],
            "oral": ["oral", "mouth"],
            "pulmonary": ["pulmonary", "lung", "bronchiolitis"],
        }

        # Nuevo: síntomas concretos por órgano para favorecer chunks clínicos específicos
        organ_manifestation_markers = {
            "skin": [
                "rash",
                "erythema",
                "cutaneous",
                "pruritus",
                "sclerotic",
                "fibrotic",
                "lichen",
            ],
            "gastrointestinal": [
                "diarrhea",
                "abdominal pain",
                "nausea",
                "vomiting",
                "anorexia",
                "gastrointestinal bleeding",
                "bleeding",
                "gut involvement",
                "intestinal involvement",
                "upper gastrointestinal",
                "lower gastrointestinal",
            ],
            "liver": [
                "bilirubin",
                "jaundice",
                "hepatic dysfunction",
                "cholestatic",
                "alkaline phosphatase",
                "serum bilirubin",
            ],
            "ocular": [
                "dry eye",
                "keratoconjunctivitis sicca",
                "ocular involvement",
                "conjunctiva",
                "corneal",
            ],
            "oral": [
                "mouth lesions",
                "oral lesions",
                "oral involvement",
                "mucosa",
                "ulcers",
                "lichen planus",
            ],
            "pulmonary": [
                "bronchiolitis obliterans",
                "cough",
                "dyspnea",
                "airflow obstruction",
                "lung involvement",
            ],
        }

        requested_organs: List[str] = []
        for organ_name, organ_terms in query_specific_organs.items():
            if any(term in query_lower for term in organ_terms):
                requested_organs.append(organ_name)

        if intent == "manifestations" and block == "organ" and requested_organs:
            organ_meta = (meta.get("organ") or "").lower()
            specific_organ_bonus = 0.0

            if organ_meta in requested_organs:
                specific_organ_bonus += 2.0

            for requested_organ in requested_organs:
                requested_terms = query_specific_organs[requested_organ]
                if any(term in lowered for term in requested_terms):
                    specific_organ_bonus += 0.5
                    break

            if organ_meta and organ_meta not in requested_organs:
                specific_organ_bonus -= 1.5

            if specific_organ_bonus != 0.0:
                score += specific_organ_bonus
                breakdown["specific_organ_alignment"] = specific_organ_bonus

            # Nuevo: bonus por síntomas explícitos del órgano pedido
            organ_symptom_bonus = 0.0
            if organ_meta in requested_organs:
                markers = organ_manifestation_markers.get(organ_meta, [])
                marker_hits = sum(1 for marker in markers if marker in lowered)

                if marker_hits >= 3:
                    organ_symptom_bonus += 2.0
                elif marker_hits == 2:
                    organ_symptom_bonus += 1.2
                elif marker_hits == 1:
                    organ_symptom_bonus += 0.5

                if organ_symptom_bonus != 0.0:
                    score += organ_symptom_bonus
                    breakdown["specific_organ_manifestation_bonus"] = (
                        organ_symptom_bonus
                    )

        intent_bonus = 0.0

        if intent == "manifestations":
            if chunk_intent == "manifestations":
                intent_bonus += 3.0
            elif chunk_intent == "intro":
                if (
                    block == "diagnosis"
                    and diag_type
                    and (
                        (query_asks_acute and diag_type == "acute")
                        or (query_asks_chronic and diag_type == "chronic")
                    )
                ):
                    intent_bonus += 0.4
                else:
                    intent_bonus -= 1.2
            elif chunk_intent == "treatment":
                if (
                    block == "diagnosis"
                    and diag_type
                    and (
                        (query_asks_acute and diag_type == "acute")
                        or (query_asks_chronic and diag_type == "chronic")
                    )
                ):
                    intent_bonus -= 0.2
                else:
                    intent_bonus -= 1.5
            elif chunk_intent == "differential":
                intent_bonus -= 3.0

        elif intent == "differential":
            if chunk_intent == "differential":
                intent_bonus += 3.0
            elif chunk_intent == "manifestations":
                intent_bonus -= 0.5
            elif chunk_intent == "intro":
                intent_bonus -= 1.0

        elif intent == "treatment":
            if chunk_intent == "treatment":
                intent_bonus += 3.0
            elif chunk_intent == "manifestations":
                intent_bonus -= 0.5
            elif chunk_intent == "intro":
                intent_bonus -= 1.0

        score += intent_bonus
        if intent_bonus:
            breakdown["intent_bonus"] = intent_bonus

        organ_bonus = self._organ_intent_bonus(query, meta, text)
        score += organ_bonus
        if organ_bonus:
            breakdown["organ_bonus"] = organ_bonus

        return score, breakdown

    def _clinical_density_bonus(self, text: str) -> float:
        """
        Bonificación por densidad de términos clínicos.
        """
        lowered = text.lower()
        clinical_terms = [
            "manifestations",
            "clinical findings",
            "clinical features",
            "symptoms",
            "signs",
            "skin",
            "rash",
            "erythema",
            "liver",
            "bilirubin",
            "jaundice",
            "gastrointestinal",
            "diarrhea",
            "abdominal pain",
            "nausea",
            "vomiting",
            "oral",
            "ocular",
            "pulmonary",
            "bronchiolitis obliterans",
            "mouth",
            "dry eye",
        ]
        hits = sum(1 for term in clinical_terms if term in lowered)

        if hits >= 8:
            return 2.0
        if hits >= 5:
            return 1.2
        if hits >= 3:
            return 0.6
        return -0.4

    # bibliography_signals
    def _bibliography_signals(self, text: str) -> int:
        """
        Señales de que el chunk parece bibliografía o referencias.
        """
        if not text:
            return 0

        score = 0
        lowered = text.lower()

        strong_terms = [
            "references",
            "bibliography",
            "et al.",
            "doi:",
            "pmid",
            "biol blood marrow transplant",
            "bone marrow transplant",
            "j clin oncol",
            "leukemia",
            "bbmt",
        ]
        for term in strong_terms:
            if term in lowered:
                score += 1

        score += len(re.findall(r"\[\d+\]", text))
        score += len(re.findall(r"\b\d{4};\d{1,3}:\d{1,5}-\d{1,5}\b", text))

        author_like = len(re.findall(r"\b[A-Z][a-zA-Z\-']+\s+[A-Z]{1,3}\b", text))
        if author_like >= 8:
            score += 3
        elif author_like >= 5:
            score += 2

        years = len(re.findall(r"\b(?:19|20)\d{2}\b", text))
        if years >= 5:
            score += 3
        elif years >= 3:
            score += 2

        return score

    # _detect_query_intent
    def _detect_query_intent(self, query: str) -> str:
        """
        Clasifica la intención principal de la query.
        """
        q = query.lower()

        if any(
            x in q
            for x in [
                "clinical manifestations",
                "manifestations",
                "symptoms",
                "clinical features",
                "signs",
            ]
        ):
            return "manifestations"

        if any(
            x in q
            for x in [
                "differential diagnosis",
                "differential",
                "diagnosis of",
                "how to diagnose",
            ]
        ):
            return "differential"

        if any(
            x in q
            for x in [
                "treatment",
                "therapy",
                "management",
                "first line",
                "second line",
            ]
        ):
            return "treatment"

        return "general"

    # _classify_chunk_intent_type
    def _classify_chunk_intent_type(self, text: str) -> str:
        """
        Clasifica el tipo de chunk.
        """
        t = text.lower()

        differential_terms = [
            "differential",
            "diagnosis of",
            "work-up",
            "histopathologic",
            "molecular tests",
            "diagnostic approach",
            "establish the diagnosis",
        ]

        manifestation_terms = [
            "clinical manifestations",
            "manifestations",
            "clinical features",
            "clinical findings",
            "symptoms",
            "signs",
            "skin",
            "rash",
            "erythema",
            "gastrointestinal",
            "diarrhea",
            "abdominal pain",
            "nausea",
            "vomiting",
            "liver",
            "bilirubin",
            "jaundice",
            "oral",
            "ocular",
            "pulmonary",
        ]

        treatment_terms = [
            "treatment",
            "therapy",
            "management",
            "steroids",
            "first-line",
            "second-line",
        ]

        intro_terms = [
            "introduction",
            "summary",
            "important complication",
            "incidence",
            "prophylaxis",
            "pathobiology",
            "molecular biology",
            "etiopathogenesis",
            "update on",
        ]

        if any(x in t for x in differential_terms):
            # solo diferencial si NO hay señal clínica real
            manifestation_hits = sum(1 for x in manifestation_terms if x in t)

            if manifestation_hits == 0:
                return "differential"

        manifestation_hits = sum(1 for x in manifestation_terms if x in t)
        treatment_hits = sum(1 for x in treatment_terms if x in t)
        intro_hits = sum(1 for x in intro_terms if x in t)

        if manifestation_hits >= 2:
            return "manifestations"

        if treatment_hits >= 2 and manifestation_hits == 0 and intro_hits == 0:
            return "treatment"

        if intro_hits >= 1:
            return "intro"

        if manifestation_hits >= 1:
            return "manifestations"

        if treatment_hits >= 1:
            return "treatment"

        return "neutral"

    # _organ_intent_bonus
    def _organ_intent_bonus(self, query: str, meta: Dict[str, Any], text: str) -> float:
        """
        Bonifica coincidencias de órgano entre query, metadatos y contenido.
        """
        q = query.lower()
        organ = (meta.get("organ") or "").lower()
        t = text.lower()

        organ_map = {
            "skin": ["skin", "rash", "erythema", "cutaneous"],
            "liver": ["liver", "bilirubin", "jaundice", "hepatic"],
            "gastrointestinal": [
                "gastrointestinal",
                "diarrhea",
                "abdominal pain",
                "nausea",
                "vomiting",
                "gi",
            ],
            "oral": ["oral", "mouth"],
            "ocular": ["ocular", "eye", "dry eye"],
            "pulmonary": ["pulmonary", "lung", "bronchiolitis"],
        }

        bonus = 0.0
        for organ_name, terms in organ_map.items():
            if any(term in q for term in terms):
                if organ == organ_name:
                    bonus += 2.0
                if any(term in t for term in terms):
                    bonus += 0.5
        return bonus

    # _cross_encoder_raw_score
    def _cross_encoder_raw_score(self, query: str, text: str) -> float:
        """
        Score bruto del cross-encoder.
        """
        if not self.cross_encoder_enabled or self.cross_encoder is None:
            return 0.0

        try:
            snippet = self._clean_preview(text, max_len=1200)
            score = self.cross_encoder.predict([(query, snippet)])
            return float(score[0])
        except Exception as exc:
            if self.debug:
                print(f"[RAG] Cross-encoder score error: {exc}")
            return 0.0

    # _normalize_cross_encoder_score
    def _normalize_cross_encoder_score(self, raw_score: float) -> float:
        """
        Normaliza el score bruto del cross-encoder a una escala útil para suma.
        """
        if raw_score >= 8:
            return 2.0
        if raw_score >= 6:
            return 1.5
        if raw_score >= 4:
            return 0.8
        if raw_score >= 2:
            return 0.2
        if raw_score > 0:
            return -0.2
        return 0.0

    def _is_valid_node(self, text: str, final_score: float) -> bool:
        """
        Filtro duro previo a la selección.
        """
        if not text or len(text.split()) < 40:
            return False

        lowered = text.lower()

        if any(
            bad in lowered
            for bad in [
                "antenna house pdf output library",
                "endstream",
                "endobj",
                "obj stream",
            ]
        ):
            return False

        biblio_score = self._bibliography_signals(text)

        if biblio_score >= 4:
            return False

        if "references" in lowered and biblio_score >= 2:
            return False

        if final_score < 1.0:
            return False

        return True

    def _is_valid_candidate(
        self,
        entry: Dict[str, Any],
        intent: str,
        wants_acute: bool,
        wants_chronic: bool,
    ) -> bool:
        """
        Filtra candidatos antes de la selección diversa final.
        """
        meta = entry.get("meta", {}) or {}
        diag_type = (meta.get("diagnosis_type") or "").lower()
        block = (meta.get("block") or "").lower()
        final_score = float(entry.get("final_score", 0.0) or 0.0)
        text = entry["text"]
        lowered = text.lower()
        chunk_intent = self._classify_chunk_intent_type(text)

        if final_score < 0.0:
            return False

        if self._looks_non_english_chunk(text):
            return False

        if wants_acute and diag_type == "chronic":
            return False

        if wants_chronic and diag_type == "acute":
            return False

        if block == "organ":
            if wants_acute:
                chronic_signals = [
                    "chronic graft versus host disease",
                    "chronic graft-versus-host disease",
                    "cgvhd",
                    "chronic gvhd",
                    "sclerotic manifestations of chronic graft versus host disease",
                    "fibrotic and sclerotic manifestations of chronic graft versus host disease",
                ]
                if any(signal in lowered for signal in chronic_signals):
                    return False

            if wants_chronic:
                acute_signals = [
                    "acute graft versus host disease",
                    "acute graft-versus-host disease",
                    "agvhd",
                    "acute gvhd",
                ]
                if any(signal in lowered for signal in acute_signals):
                    return False

        if intent == "manifestations" and chunk_intent == "differential":
            return False

        if any(
            x in lowered
            for x in [
                "work-up",
                "establish the diagnosis",
                "diagnostic approach",
                "molecular tests",
            ]
        ):
            return False

        return True

    def _is_primary_diagnosis(self, entry: Dict[str, Any], target_diag: str) -> bool:
        """
        Comprueba si un chunk pertenece al bloque de diagnóstico principal
        del tipo buscado (acute/chronic).
        """
        if not target_diag:
            return False

        meta = entry.get("meta", {}) or {}
        block = (meta.get("block") or "").lower()
        diag_type = (meta.get("diagnosis_type") or "").lower()

        return block == "diagnosis" and diag_type == target_diag

    def _select_diverse_chunks(self, scored_entries, query: str):
        """
        Selecciona una mezcla útil de evidencias.
        """

        q = query.lower()
        intent = self._detect_query_intent(query)

        wants_acute = "acute" in q
        wants_chronic = "chronic" in q
        target_diag = "acute" if wants_acute else ("chronic" if wants_chronic else "")

        requested_organs: List[str] = []
        specific_organ_terms = {
            "skin": ["skin", "cutaneous", "rash", "erythema"],
            "gastrointestinal": [
                "gastrointestinal",
                "diarrhea",
                "abdominal pain",
                "nausea",
                "vomiting",
                "gut",
                "gi",
            ],
            "liver": ["liver", "bilirubin", "jaundice", "hepatic"],
            "ocular": ["ocular", "eye", "dry eye"],
            "oral": ["oral", "mouth"],
            "pulmonary": ["pulmonary", "lung", "bronchiolitis"],
        }

        for organ_name, terms in specific_organ_terms.items():
            if any(term in q for term in terms):
                requested_organs.append(organ_name)

        candidates = [
            entry
            for entry in scored_entries
            if self._is_valid_candidate(entry, intent, wants_acute, wants_chronic)
        ]

        if self.debug:
            print("\n================ DIVERSE SELECTION INPUT DEBUG ================")
            print("[DEBUG] query:", query)
            print("[DEBUG] intent:", intent)
            print("[DEBUG] wants_acute:", wants_acute)
            print("[DEBUG] wants_chronic:", wants_chronic)
            print("[DEBUG] target_diag:", target_diag)
            print("[DEBUG] requested_organs:", requested_organs)
            print("[DEBUG] scored_entries_count:", len(scored_entries))
            print("[DEBUG] candidates_count:", len(candidates))

            candidate_organs = [
                ((entry.get("meta", {}) or {}).get("organ") or "None")
                for entry in candidates
            ]
            print("[DEBUG] candidate_organs_raw:", candidate_organs)

            for i, entry in enumerate(candidates[:10], 1):
                meta = entry.get("meta", {}) or {}
                print(f"[DEBUG] candidate[{i}] file_name:", meta.get("file_name"))
                print(f"[DEBUG] candidate[{i}] block:", meta.get("block"))
                print(
                    f"[DEBUG] candidate[{i}] diagnosis_type:",
                    meta.get("diagnosis_type"),
                )
                print(f"[DEBUG] candidate[{i}] organ:", meta.get("organ"))
                print(f"[DEBUG] candidate[{i}] final_score:", entry.get("final_score"))

        if intent != "manifestations":
            if self.debug:
                print(
                    "\n================ DIVERSE SELECTION NON-MANIFESTATIONS RETURN ================"
                )
                for i, entry in enumerate(candidates[:5], 1):
                    meta = entry.get("meta", {}) or {}
                    print(
                        f"[DEBUG] return_candidate[{i}] file_name:",
                        meta.get("file_name"),
                    )
                    print(f"[DEBUG] return_candidate[{i}] block:", meta.get("block"))
                    print(
                        f"[DEBUG] return_candidate[{i}] diagnosis_type:",
                        meta.get("diagnosis_type"),
                    )
                print(f"[DEBUG] return_candidate[{i}] organ:", meta.get("organ"))
                print(
                    f"[DEBUG] return_candidate[{i}] final_score:",
                    entry.get("final_score"),
                )
            return candidates[:5]

        diagnosis_base: List[Dict[str, Any]] = []
        organ_manifestations: List[Dict[str, Any]] = []
        supportive_neutral: List[Dict[str, Any]] = []

        for entry in candidates:
            text = entry["text"]
            meta = entry.get("meta", {}) or {}
            chunk_intent = self._classify_chunk_intent_type(text)
            block = (meta.get("block") or "").lower()

            is_primary_diag = self._is_primary_diagnosis(entry, target_diag)

            if is_primary_diag and chunk_intent in {
                "manifestations",
                "intro",
                "neutral",
            }:
                diagnosis_base.append(entry)
                continue

            if block == "organ" and chunk_intent == "manifestations":
                organ_manifestations.append(entry)
                continue

            supportive_neutral.append(entry)

        if self.debug:
            print("\n================ DIVERSE CLASSIFICATION DEBUG ================")
            print("[DEBUG] diagnosis_base_count:", len(diagnosis_base))
            print("[DEBUG] organ_manifestations_count:", len(organ_manifestations))
            print("[DEBUG] supportive_neutral_count:", len(supportive_neutral))

            organ_manifestation_organs = [
                ((entry.get("meta", {}) or {}).get("organ") or "None")
                for entry in organ_manifestations
            ]
            print("[DEBUG] organ_manifestation_organs_raw:", organ_manifestation_organs)

            for i, entry in enumerate(diagnosis_base[:5], 1):
                meta = entry.get("meta", {}) or {}
                print(f"[DEBUG] diagnosis_base[{i}] file_name:", meta.get("file_name"))
                print(
                    f"[DEBUG] diagnosis_base[{i}] diagnosis_type:",
                    meta.get("diagnosis_type"),
                )
                print(
                    f"[DEBUG] diagnosis_base[{i}] final_score:",
                    entry.get("final_score"),
                )

            for i, entry in enumerate(organ_manifestations[:10], 1):
                meta = entry.get("meta", {}) or {}
                print(
                    f"[DEBUG] organ_manifestations[{i}] file_name:",
                    meta.get("file_name"),
                )
                print(f"[DEBUG] organ_manifestations[{i}] organ:", meta.get("organ"))
                print(
                    f"[DEBUG] organ_manifestations[{i}] final_score:",
                    entry.get("final_score"),
                )

            for i, entry in enumerate(supportive_neutral[:5], 1):
                meta = entry.get("meta", {}) or {}
                print(
                    f"[DEBUG] supportive_neutral[{i}] file_name:", meta.get("file_name")
                )
                print(f"[DEBUG] supportive_neutral[{i}] block:", meta.get("block"))
                print(
                    f"[DEBUG] supportive_neutral[{i}] diagnosis_type:",
                    meta.get("diagnosis_type"),
                )
                print(f"[DEBUG] supportive_neutral[{i}] organ:", meta.get("organ"))
                print(
                    f"[DEBUG] supportive_neutral[{i}] final_score:",
                    entry.get("final_score"),
                )

        result: List[Dict[str, Any]] = []

        if requested_organs:
            specific_organ_chunks = [
                entry
                for entry in organ_manifestations
                if ((entry.get("meta", {}) or {}).get("organ") or "").lower()
                in requested_organs
            ]

            if self.debug:
                print(
                    "\n================ SPECIFIC ORGAN SELECTION DEBUG ================"
                )
                print("[DEBUG] requested_organs:", requested_organs)
                print(
                    "[DEBUG] specific_organ_chunks_count:", len(specific_organ_chunks)
                )

                specific_organ_names = [
                    ((entry.get("meta", {}) or {}).get("organ") or "None")
                    for entry in specific_organ_chunks
                ]
                print("[DEBUG] specific_organ_chunks_organs_raw:", specific_organ_names)

                for i, entry in enumerate(specific_organ_chunks[:10], 1):
                    meta = entry.get("meta", {}) or {}
                    print(
                        f"[DEBUG] specific_organ_chunk[{i}] file_name:",
                        meta.get("file_name"),
                    )
                    print(
                        f"[DEBUG] specific_organ_chunk[{i}] organ:", meta.get("organ")
                    )
                    print(
                        f"[DEBUG] specific_organ_chunk[{i}] final_score:",
                        entry.get("final_score"),
                    )

            seen_organs = set()
            for entry in specific_organ_chunks:
                organ = ((entry.get("meta", {}) or {}).get("organ") or "").lower()
                key = organ or entry.get("meta", {}).get("file_name") or id(entry)

                if key in seen_organs:
                    continue

                seen_organs.add(key)

                if entry not in result:
                    result.append(entry)

                if self.debug:
                    meta = entry.get("meta", {}) or {}
                    print("\n================ RESULT BUILD DEBUG ================")
                    print(
                        "[DEBUG] added_specific_organ file_name:", meta.get("file_name")
                    )
                    print("[DEBUG] added_specific_organ organ:", meta.get("organ"))
                    print("[DEBUG] current_result_count:", len(result))

                if len(result) >= 4:
                    if self.debug:
                        print("\n================ FINAL RESULT DEBUG ================")
                        for i, r in enumerate(result[:4], 1):
                            meta = r.get("meta", {}) or {}
                            print(
                                f"[DEBUG] final_result[{i}] file_name:",
                                meta.get("file_name"),
                            )
                            print(
                                f"[DEBUG] final_result[{i}] block:", meta.get("block")
                            )
                            print(
                                f"[DEBUG] final_result[{i}] diagnosis_type:",
                                meta.get("diagnosis_type"),
                            )
                            print(
                                f"[DEBUG] final_result[{i}] organ:", meta.get("organ")
                            )
                            print(
                                f"[DEBUG] final_result[{i}] final_score:",
                                r.get("final_score"),
                            )
                return result[:4]

            if diagnosis_base:
                if diagnosis_base[0] not in result:
                    result.append(diagnosis_base[0])
                    if self.debug:
                        meta = diagnosis_base[0].get("meta", {}) or {}
                        print("\n================ RESULT BUILD DEBUG ================")
                        print(
                            "[DEBUG] added_diagnosis_base file_name:",
                            meta.get("file_name"),
                        )
                        print(
                            "[DEBUG] added_diagnosis_base diagnosis_type:",
                            meta.get("diagnosis_type"),
                        )
                        print("[DEBUG] current_result_count:", len(result))

            if not specific_organ_chunks:
                if self.debug:
                    print(
                        "\n================ SPECIFIC ORGAN FALLBACK DEBUG ================"
                    )
                    print(
                        "[DEBUG] No specific organ chunks found for requested organs."
                    )
                    print("[DEBUG] requested_organs:", requested_organs)
                    print("[DEBUG] diagnosis_base_kept_count:", len(result))
                    for i, entry in enumerate(result[:10], 1):
                        meta = entry.get("meta", {}) or {}
                        print(
                            f"[DEBUG] fallback_result[{i}] file_name:",
                            meta.get("file_name"),
                        )
                        print(f"[DEBUG] fallback_result[{i}] block:", meta.get("block"))
                        print(
                            f"[DEBUG] fallback_result[{i}] diagnosis_type:",
                            meta.get("diagnosis_type"),
                        )
                        print(f"[DEBUG] fallback_result[{i}] organ:", meta.get("organ"))
                        print(
                            f"[DEBUG] fallback_result[{i}] final_score:",
                            entry.get("final_score"),
                        )
                return result[:6]

            for entry in supportive_neutral:
                if entry not in result:
                    result.append(entry)
                    if self.debug:
                        meta = entry.get("meta", {}) or {}
                        print("\n================ RESULT BUILD DEBUG ================")
                        print(
                            "[DEBUG] added_supportive_neutral file_name:",
                            meta.get("file_name"),
                        )
                        print(
                            "[DEBUG] added_supportive_neutral block:", meta.get("block")
                        )
                        print(
                            "[DEBUG] added_supportive_neutral organ:", meta.get("organ")
                        )
                        print("[DEBUG] current_result_count:", len(result))
                if len(result) >= 4:
                    break

            if self.debug:
                print("\n================ FINAL RESULT DEBUG ================")
                for i, r in enumerate(result[:4], 1):
                    meta = r.get("meta", {}) or {}
                    print(
                        f"[DEBUG] final_result[{i}] file_name:", meta.get("file_name")
                    )
                    print(f"[DEBUG] final_result[{i}] block:", meta.get("block"))
                    print(
                        f"[DEBUG] final_result[{i}] diagnosis_type:",
                        meta.get("diagnosis_type"),
                    )
                    print(f"[DEBUG] final_result[{i}] organ:", meta.get("organ"))
                    print(
                        f"[DEBUG] final_result[{i}] final_score:", r.get("final_score")
                    )

            return result[:4]

        if diagnosis_base:
            result.append(diagnosis_base[0])

        seen_organs = set()
        for entry in organ_manifestations:
            organ = ((entry.get("meta", {}) or {}).get("organ") or "").lower()
            key = organ or entry.get("meta", {}).get("file_name") or id(entry)

            if key in seen_organs:
                continue

            seen_organs.add(key)

            if entry not in result:
                result.append(entry)

            if len(result) >= 4:
                break

        for entry in supportive_neutral:
            if entry not in result:
                result.append(entry)

            if len(result) >= 4:
                break

        if self.debug:
            print("\n================ FINAL RESULT DEBUG ================")
            for i, r in enumerate(result[:4], 1):
                meta = r.get("meta", {}) or {}
                print(f"[DEBUG] final_result[{i}] file_name:", meta.get("file_name"))
                print(f"[DEBUG] final_result[{i}] block:", meta.get("block"))
                print(
                    f"[DEBUG] final_result[{i}] diagnosis_type:",
                    meta.get("diagnosis_type"),
                )
                print(f"[DEBUG] final_result[{i}] organ:", meta.get("organ"))
                print(f"[DEBUG] final_result[{i}] final_score:", r.get("final_score"))

        return result[:4]

    # =========================================================================
    # Construcción de contexto
    # =========================================================================

    # _build_combined_llm_context
    def _build_combined_llm_context(
        self,
        query: str,
        retrieved_entries: List[Dict[str, Any]],
        paciente_id: Optional[int] = None,
    ) -> str:
        """
        Construye contexto mixto para Opción B:
        1. contexto estructurado del paciente desde SQL
        2. contexto bibliográfico filtrado desde el corpus
        """
        sql_context = self._build_sql_context(paciente_id)
        literature_context = self._build_llm_context(query, retrieved_entries)

        if self.debug:
            print("\n================ COMBINED CONTEXT DEBUG ================")
            print("[DEBUG] paciente_id:", paciente_id)
            print("[DEBUG] sql_context_length:", len(sql_context or ""))
            print("[DEBUG] literature_context_length:", len(literature_context or ""))
            print("[DEBUG] sql_context_preview:", (sql_context or "")[:1000])
            print(
                "[DEBUG] literature_context_preview:",
                (literature_context or "")[:1500],
            )

        blocks: List[str] = []

        if sql_context.strip():
            blocks.append("[PATIENT_STRUCTURED_CONTEXT]\n" + sql_context.strip())

        if literature_context.strip():
            blocks.append("[LITERATURE_CONTEXT]\n" + literature_context.strip())

        combined_context = "\n\n".join(blocks)

        if self.debug:
            print("[DEBUG] combined_context_length:", len(combined_context))
            print("[DEBUG] combined_context_preview:", combined_context[:2000])

        return combined_context

    # _build_sql_context
    def _build_sql_context(self, paciente_id: Optional[int]) -> str:
        """
        Convierte el contexto estructurado del paciente a texto compacto.
        """
        if not paciente_id:
            return ""

        try:
            return self.sql_service.build_patient_context_text(paciente_id)
        except Exception as exc:
            if self.debug:
                print(f"[RAG] SQL context text error: {exc}")
            return ""

    # _get_sql_patient_context
    def _get_sql_patient_context(self, paciente_id: Optional[int]) -> Dict[str, Any]:
        """
        Recupera contexto estructurado del paciente desde SQL.
        """
        if not paciente_id:
            return {}

        try:
            return self.sql_service.get_patient_context(paciente_id)
        except Exception as exc:
            if self.debug:
                print(f"[RAG] SQL patient context error: {exc}")
            return {}

    # _build_llm_context
    def _build_llm_context(
        self, query: str, retrieved_entries: List[Dict[str, Any]]
    ) -> str:
        """
        Construye un contexto limpio, compacto y orientado a la query para el LLM.
        """
        query_lower = query.lower()
        context_blocks: List[str] = []
        used_sentences = 0

        organ_focus_terms = {
            "skin": ["skin", "cutaneous", "rash", "erythema"],
            "gastrointestinal": [
                "gastrointestinal",
                "gi",
                "gut",
                "diarrhea",
                "abdominal pain",
                "nausea",
                "vomiting",
            ],
            "liver": ["liver", "bilirubin", "jaundice", "hepatic"],
            "ocular": ["ocular", "eye", "dry eye", "keratoconjunctivitis"],
            "oral": ["oral", "mouth", "mucosa", "buccal", "tongue", "lip", "gingiva"],
            "pulmonary": ["pulmonary", "lung", "bronchiolitis"],
        }

        requested_organs: List[str] = []
        for organ_name, terms in organ_focus_terms.items():
            if any(term in query_lower for term in terms):
                requested_organs.append(organ_name)

        is_specific_query = len(requested_organs) > 0
        requested_terms: List[str] = []
        for organ_name in requested_organs:
            requested_terms.extend(organ_focus_terms[organ_name])

        organ_exclusion_terms = {
            "skin": [
                "ocular",
                "eye",
                "mouth",
                "oral",
                "genital",
                "vulvovaginal",
                "balanitis",
                "labium",
                "vulvar",
                "vulva",
                "vaginal",
                "coronal sulcus",
            ],
            "gastrointestinal": [
                "ocular",
                "eye",
                "mouth",
                "oral",
                "genital",
                "vulvovaginal",
                "balanitis",
                "labium",
                "vulvar",
                "vulva",
                "vaginal",
                "coronal sulcus",
            ],
            "ocular": [
                "genital",
                "vulvovaginal",
                "balanitis",
                "coronal sulcus",
                "labium",
                "vulva",
                "vulvar",
                "vaginal",
            ],
            "oral": [
                "genital",
                "vulvovaginal",
                "balanitis",
                "coronal sulcus",
                "labium",
                "vulva",
                "vulvar",
                "vaginal",
            ],
            "liver": [
                "genital",
                "vulvovaginal",
                "balanitis",
                "labium",
                "vulvar",
                "vulva",
                "vaginal",
            ],
            "pulmonary": [
                "genital",
                "vulvovaginal",
                "balanitis",
                "labium",
                "vulvar",
                "vulva",
                "vaginal",
            ],
        }

        disallowed_terms: List[str] = []
        for organ_name in requested_organs:
            disallowed_terms.extend(organ_exclusion_terms.get(organ_name, []))

        if self.debug:
            print("\n================ LLM CONTEXT INPUT DEBUG ================")
            print("[DEBUG] query:", query)
            print("[DEBUG] requested_organs:", requested_organs)
            print("[DEBUG] is_specific_query:", is_specific_query)
            print("[DEBUG] requested_terms:", requested_terms)
            print("[DEBUG] disallowed_terms:", disallowed_terms)
            print("[DEBUG] retrieved_entries_count:", len(retrieved_entries))

        for entry in retrieved_entries:
            meta = entry.get("meta", {}) or {}
            block = (meta.get("block") or "unknown").lower()
            organ = (meta.get("organ") or "").lower()
            diagnosis_type = (meta.get("diagnosis_type") or "").lower()
            file_name = meta.get("file_name") or "unknown_source"

            sentences = self._extract_best_clinical_sentences_from_text(entry["text"])

            if self.debug:
                print("\n---------------- LLM ENTRY DEBUG ----------------")
                print("[DEBUG] file_name:", file_name)
                print("[DEBUG] block:", block)
                print("[DEBUG] organ:", organ)
                print("[DEBUG] diagnosis_type:", diagnosis_type)
                print("[DEBUG] extracted_sentences_count:", len(sentences))
                for i, s in enumerate(sentences[:6], 1):
                    print(f"[DEBUG] extracted_sentence[{i}]:", s)

            if not sentences:
                continue

            selected_sentences: List[str] = []

            for sentence in sentences:
                if used_sentences >= self.llm_max_context_sentences:
                    break

                sentence_lower = sentence.lower()

                if "acute" in query_lower and (
                    "chronic graft-versus-host disease" in sentence_lower
                    or "chronic graft versus host disease" in sentence_lower
                    or "cgvhd" in sentence_lower
                    or "chronic gvhd" in sentence_lower
                ):
                    continue

                if "chronic" in query_lower and (
                    "acute graft-versus-host disease" in sentence_lower
                    or "acute graft versus host disease" in sentence_lower
                    or "agvhd" in sentence_lower
                    or "acute gvhd" in sentence_lower
                ):
                    continue

                if is_specific_query:
                    has_requested_signal = any(
                        term in sentence_lower for term in requested_terms
                    )
                    has_disallowed_signal = any(
                        term in sentence_lower for term in disallowed_terms
                    )

                    entry_matches_requested_organ = (
                        organ in requested_organs if organ else False
                    )

                    if block == "organ":
                        if not entry_matches_requested_organ:
                            continue
                    elif block == "diagnosis":
                        if not has_requested_signal:
                            continue
                    else:
                        if not has_requested_signal:
                            continue

                    if has_disallowed_signal:
                        continue

                if self.debug:
                    print("[DEBUG] sentence_selected:", sentence)

                selected_sentences.append(sentence)
                used_sentences += 1

            if self.debug:
                print(
                    "[DEBUG] selected_sentences_count_for_entry:",
                    len(selected_sentences),
                )
                for i, s in enumerate(selected_sentences[:6], 1):
                    print(f"[DEBUG] selected_sentence[{i}]:", s)

            if not selected_sentences:
                if self.debug:
                    print("[DEBUG] entry_discarded_from_llm_context:", file_name)
                continue

            header_parts = [f"source={file_name}", f"block={block}"]
            if diagnosis_type:
                header_parts.append(f"diagnosis_type={diagnosis_type}")
            if organ:
                header_parts.append(f"organ={organ}")

            block_header = "[" + " | ".join(header_parts) + "]"
            block_text = " ".join(selected_sentences)

            context_blocks.append(f"{block_header}\n{block_text}")

            if used_sentences >= self.llm_max_context_sentences:
                break

        final_context = "\n\n".join(context_blocks)

        if self.debug:
            print("\n================ LLM CONTEXT OUTPUT DEBUG ================")
            print("[DEBUG] used_sentences:", used_sentences)
            print("[DEBUG] context_blocks_count:", len(context_blocks))
            print("[DEBUG] llm_context_preview:", final_context[:2000])

        return final_context

    # _extract_best_clinical_sentences_from_text
    # _extract_best_clinical_sentences_from_text
    def _extract_best_clinical_sentences_from_text(self, text: str) -> List[str]:
        """
        Extrae las mejores frases clínicas de un chunk.
        Prioriza frases con manifestaciones clínicas concretas.
        """

        text = re.sub(r"\s+", " ", text).strip()
        sentences = re.split(r"(?<=[\.\!\?])\s+", text)

        selected: List[str] = []

        title_like_prefixes = [
            "reviews ",
            "review ",
            "original article ",
            "original articles ",
            "clinical review ",
            "update on ",
        ]

        high_value_markers = [
            "diarrhea",
            "nausea",
            "vomiting",
            "abdominal pain",
            "anorexia",
            "gastrointestinal bleeding",
            "bleeding",
            "gut involvement",
            "intestinal involvement",
            "gastrointestinal manifestations",
            "upper gastrointestinal",
            "lower gastrointestinal",
            "rash",
            "erythema",
            "jaundice",
            "bilirubin",
            "dry eye",
            "keratoconjunctivitis sicca",
            "mouth lesions",
            "oral lesions",
            "bronchiolitis obliterans",
            "sclerotic",
            "fibrotic",
            "clinical characteristics",
            "clinical findings",
            "clinical features",
            "endoscopic findings",
            "histopathological findings",
        ]

        weak_context_only_patterns = [
            "studies concerning",
            "are limited",
            "have been limited",
            "is limited",
            "reviewed in this article",
            "in developing countries",
            "in developed countries",
        ]

        for sentence in sentences:
            s = sentence.strip()

            if self.debug:
                print("\n================ SENTENCE EXTRACTION DEBUG ================")
                print("[DEBUG] raw_sentence:", s)

            if not s:
                if self.debug:
                    print("[DEBUG] discard_reason: empty_sentence")
                continue

            lowered = s.lower()

            if any(
                bad in lowered
                for bad in [
                    "antenna house pdf output library",
                    "endstream",
                    "endobj",
                    "obj stream",
                    "department of",
                    "university",
                    "doi:",
                    "correspondence",
                    "faculty of medicine",
                    "division of hematology",
                    "blood and marrow transplant program",
                    "perelman school of medicine",
                ]
            ):
                if self.debug:
                    print("[DEBUG] discard_reason: hard_metadata_noise")
                continue

            if any(lowered.startswith(prefix) for prefix in title_like_prefixes):
                if self.debug:
                    print("[DEBUG] discard_reason: title_like_prefix")
                continue

            if re.search(r"\b[A-Z]\.-[A-Z]\.", s):
                if self.debug:
                    print("[DEBUG] discard_reason: author_initial_pattern")
                continue

            if re.search(r"\b[A-ZÁÉÍÓÚÄËÏÖÜ][a-záéíóúäëïöüñ]+\d\b", s):
                if self.debug:
                    print("[DEBUG] discard_reason: token_name_number_pattern")
                continue

            if (
                len(s.split()) <= 18
                and sum(1 for token in s.split() if token[:1].isupper()) >= 6
            ):
                if self.debug:
                    print("[DEBUG] discard_reason: short_caps_heavy_sentence")
                continue

            cleaned_sentence = self._strip_editorial_phrases(s)
            cleaned_sentence = re.sub(r"\s+", " ", cleaned_sentence).strip(" ,;:-")

            if self.debug:
                print("[DEBUG] cleaned_sentence_candidate:", cleaned_sentence)

            if not cleaned_sentence:
                if self.debug:
                    print("[DEBUG] discard_reason: empty_after_cleaning")
                continue

            cleaned_lower = cleaned_sentence.lower()

            if any(cleaned_lower.startswith(prefix) for prefix in title_like_prefixes):
                if self.debug:
                    print("[DEBUG] discard_reason: cleaned_title_like_prefix")
                continue

            if cleaned_lower in {
                "reviews",
                "review",
                "update on etiopathogenesis and diagnosis",
            }:
                if self.debug:
                    print("[DEBUG] discard_reason: generic_title_fragment")
                continue

            if self._is_editorial_sentence(
                s
            ) and not self._is_clinically_useful_sentence(cleaned_sentence):
                if self.debug:
                    print("[DEBUG] discard_reason: editorial_not_clinically_useful")
                continue

            if not self._is_clinically_useful_sentence(cleaned_sentence):
                if self.debug:
                    print("[DEBUG] discard_reason: not_clinically_useful")
                continue

            # Importante: aplicar el filtro de referencia AQUÍ,
            # cuando la frase ya ha pasado limpieza y utilidad clínica.
            if self._looks_like_reference_sentence(cleaned_sentence):
                if self.debug:
                    print(
                        "[DEBUG] discard_reason: cleaned_sentence_looks_like_reference"
                    )
                continue

            selected.append(cleaned_sentence)

            if self.debug:
                print("[DEBUG] selected_sentence_final:", cleaned_sentence)

        deduped: List[str] = []
        seen = set()

        for s in selected:
            key = s.lower()
            if key not in seen:
                seen.add(key)
                deduped.append(s)

        high_value_sentences: List[str] = []
        normal_sentences: List[str] = []

        for s in deduped:
            lowered = s.lower()

            has_high_value_marker = any(
                marker in lowered for marker in high_value_markers
            )
            is_weak_context_only = any(
                pattern in lowered for pattern in weak_context_only_patterns
            )

            if has_high_value_marker and not is_weak_context_only:
                high_value_sentences.append(s)
            else:
                normal_sentences.append(s)

        prioritized = high_value_sentences + normal_sentences

        if self.debug:
            print("\n================ SENTENCE EXTRACTION FINAL DEBUG ================")
            print("[DEBUG] selected_count_before_dedupe:", len(selected))
            print("[DEBUG] deduped_count:", len(deduped))
            print("[DEBUG] high_value_count:", len(high_value_sentences))
            print("[DEBUG] normal_count:", len(normal_sentences))
            for i, sent in enumerate(prioritized[:6], 1):
                print(f"[DEBUG] prioritized_sentence[{i}]:", sent)

        return prioritized[:4]

    # _is_clinically_useful_sentence
    def _is_clinically_useful_sentence(self, sentence: str) -> bool:
        """
        Determina si una frase aporta contenido clínico real.
        """

        s = self._strip_editorial_phrases(sentence)
        s = re.sub(r"\s+", " ", s).strip()
        lowered = s.lower()

        if len(s.split()) < 8:
            return False

        too_general_patterns = [
            "is an important complication",
            "incidence of",
            "prophylaxis is essential",
            "initial therapy",
            "prognosis is poor",
            "patients undergoing allo-hsct",
            "update on etiopathogenesis and diagnosis",
            "pathobiology",
            "etiopathogenesis",
        ]
        if any(pattern in lowered for pattern in too_general_patterns):
            return False

        title_like_patterns = [
            "reviews liver graft versus host disease",
            "graft versus host disease after allogeneic peripheral stem cell transplantation",
            "review",
            "reviews",
        ]
        if any(pattern in lowered for pattern in title_like_patterns):
            return False

        mechanistic_patterns = [
            "cytokines",
            "cytokine receptors",
            "complement activation",
            "humoral",
            "donor-derived",
            "mesenchymal",
            "immune response",
            "pathogenesis",
            "etiopathogenesis",
            "development of gvhd",
            "gvhd development",
            "activated cd4",
            "t cells",
            "cellular mechanisms",
            "molecular mechanisms",
        ]
        if any(pattern in lowered for pattern in mechanistic_patterns):
            return False

        diagnostic_non_manifestation_patterns = [
            "correct and early recognition",
            "recognition of gvhd",
            "differentiation from",
            "differentiation of gvhd",
            "other liver diseases",
            "differential diagnosis",
            "diagnostic approach",
            "diagnosis should be",
            "diagnosis is based on",
            "establish the diagnosis",
        ]
        if any(pattern in lowered for pattern in diagnostic_non_manifestation_patterns):
            return False

        strong_clinical_markers = [
            "clinical manifestations",
            "manifestations",
            "clinical features",
            "clinical findings",
            "symptoms",
            "signs",
            "skin",
            "rash",
            "erythema",
            "liver",
            "bilirubin",
            "jaundice",
            "gastrointestinal",
            "diarrhea",
            "abdominal pain",
            "nausea",
            "vomiting",
            "oral",
            "mouth",
            "ocular",
            "dry eye",
            "pulmonary",
            "bronchiolitis obliterans",
            "alkaline phosphatase",
            "serum bilirubin",
            "hepatic dysfunction",
            "cholestatic",
            "sclerotic",
            "fibrotic",
            "skin involvement",
            "liver involvement",
            "gut involvement",
            "lung involvement",
        ]

        supportive_clinical_markers = [
            "diagnosis",
            "therapy",
            "treatment",
            "acute gvhd",
            "agvhd",
            "acute graft-versus-host disease",
            "acute graft versus host disease",
            "chronic gvhd",
            "cgvhd",
            "chronic graft-versus-host disease",
            "chronic graft versus host disease",
            "allogeneic hematopoietic stem cell transplantation",
            "hematopoietic stem cell transplantation",
            "hsct",
            "allo-hsct",
            "steroids",
        ]

        # Nueva vía: aceptar frases de abstract/objetivos/hallazgos clínicos
        # si están claramente ligadas a órgano/enfermedad y no son fisiopatología.
        abstract_clinical_markers = [
            "clinical characteristics",
            "clinical outcomes",
            "endoscopic",
            "histopathological",
            "histopathologic",
            "findings",
            "commonly affects",
            "affects the",
            "gi tract",
            "gastro-intestinal",
            "gastrointestinal tract",
        ]

        strong_hits = sum(1 for marker in strong_clinical_markers if marker in lowered)
        supportive_hits = sum(
            1 for marker in supportive_clinical_markers if marker in lowered
        )
        abstract_hits = sum(
            1 for marker in abstract_clinical_markers if marker in lowered
        )

        if strong_hits >= 1:
            return True

        if strong_hits == 0 and supportive_hits >= 2:
            return True

        # Regla nueva y controlada:
        # frases de abstract clínico con órgano/enfermedad explícitos.
        if abstract_hits >= 1 and (
            "gvhd" in lowered
            or "graft-versus-host disease" in lowered
            or "graft versus host disease" in lowered
        ):
            organ_or_case_markers = [
                "gastrointestinal",
                "gi",
                "gut",
                "liver",
                "hepatic",
                "skin",
                "cutaneous",
                "oral",
                "mouth",
                "ocular",
                "eye",
                "pulmonary",
                "lung",
                "children",
                "pediatric",
                "patients",
            ]
            if any(marker in lowered for marker in organ_or_case_markers):
                return True

        return False

    # _looks_like_reference_sentence
    def _looks_like_reference_sentence(self, sentence: str) -> bool:
        """
        Detecta frases que parecen referencias bibliográficas.
        Debe ser conservador: no descartar frases clínicas útiles
        solo por contener autores, años o journals mezclados con abstract.
        """

        s = sentence.strip()
        lowered = s.lower()

        # Salvaguarda: si la frase contiene señal clínica clara,
        # no debe descartarse como referencia.
        strong_clinical_markers = [
            "clinical characteristics",
            "clinical manifestations",
            "clinical features",
            "clinical findings",
            "symptoms",
            "signs",
            "gastrointestinal",
            "gi tract",
            "gut",
            "diarrhea",
            "abdominal pain",
            "nausea",
            "vomiting",
            "anorexia",
            "bleeding",
            "rash",
            "erythema",
            "dry eye",
            "keratoconjunctivitis",
            "oral lesions",
            "mouth lesions",
            "jaundice",
            "bilirubin",
            "bronchiolitis obliterans",
            "liver involvement",
            "skin involvement",
            "ocular involvement",
            "oral involvement",
            "pulmonary involvement",
            "objectives:",
            "methods:",
            "results:",
            "conclusions:",
            "abstract",
        ]

        if any(marker in lowered for marker in strong_clinical_markers):
            return False

        score = 0

        strong_terms = [
            "references",
            "bibliography",
            "et al.",
            "doi:",
            "pmid",
            "biol blood marrow transplant",
            "bone marrow transplant",
            "j clin oncol",
            "leukemia",
            "bbmt",
            "author manuscript",
            "available in pmc",
            "pubmed:",
        ]

        for term in strong_terms:
            if term in lowered:
                score += 1

        bracket_citations = len(re.findall(r"\[\d+\]", s))
        if bracket_citations >= 2:
            score += 2
        elif bracket_citations == 1:
            score += 1

        journal_pattern_hits = len(re.findall(r"\b\d{4};\d{1,3}:\d{1,5}-\d{1,5}\b", s))
        if journal_pattern_hits >= 1:
            score += 2

        author_like = len(re.findall(r"\b[A-Z][a-zA-Z\-']+\s+[A-Z]{1,3}\b", s))
        if author_like >= 10:
            score += 3
        elif author_like >= 7:
            score += 2
        elif author_like >= 5:
            score += 1

        years = len(re.findall(r"\b(?:19|20)\d{2}\b", s))
        if years >= 6:
            score += 3
        elif years >= 4:
            score += 2
        elif years >= 3:
            score += 1

        # Si la frase parece claramente una lista bibliográfica, descartar.
        if score >= 5:
            return True

        return False

    # _strip_editorial_phrases
    def _strip_editorial_phrases(self, sentence: str) -> str:
        """
        Elimina frases editoriales que no aportan contenido clínico útil.
        """
        cleaned = self._clean_editorial_prefix(sentence)

        editorial_patterns = [
            r"\bthe pathobiology, clinical findings, prophylaxis, and treatment of .*? will be summarized\b",
            r"\bwill be summarized\b",
            r"\bwill be discussed\b",
            r"\bwill be reviewed\b",
            r"\bthis review summarizes\b.*?(?=[\.\!\?]|$)",
            r"\bthis review describes\b.*?(?=[\.\!\?]|$)",
            r"\bwe review\b.*?(?=[\.\!\?]|$)",
            r"\bwe summarize\b.*?(?=[\.\!\?]|$)",
            r"\bwe describe\b.*?(?=[\.\!\?]|$)",
        ]

        for pattern in editorial_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;:-")
        return cleaned

    # _clean_editorial_prefix
    def _clean_editorial_prefix(self, sentence: str) -> str:
        """
        Elimina prefijos editoriales típicos de reviews.
        """
        cleaned = re.sub(
            r"^(?:in this article|this review|we review|we describe|we summarize|the purpose of this study|the purpose of this article|this study)\s*[\,\:\-]?\s*",
            "",
            sentence.strip(),
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    # _is_editorial_sentence
    def _is_editorial_sentence(self, sentence: str) -> bool:
        """
        Detecta frases de tono editorial/metadiscursivo.
        """
        lowered = sentence.strip().lower()

        editorial_patterns = [
            "in this article",
            "this review",
            "we review",
            "we summarize",
            "we describe",
            "the purpose of this study",
            "the purpose of this article",
            "will be summarized",
            "will be discussed",
            "will be reviewed",
        ]

        return any(pattern in lowered for pattern in editorial_patterns)

    def _looks_non_english_chunk(self, text: str) -> bool:
        """Heurística simple para descartar chunks claramente no ingleses."""
        lowered = text.lower()

        strong_non_english_patterns = [
            "diagnostischezeichen",
            "laboruntersuchungen",
            "bildgebung",
            "ausgeschlossenwerden",
            "hautanhangsgebilde",
            "betroﬀen",
            "präsentiert",
        ]
        if any(pattern in lowered for pattern in strong_non_english_patterns):
            return True

        compact_long_tokens = re.findall(r"\b[a-zA-Z]{18,}\b", text)
        if len(compact_long_tokens) >= 3:
            return True
        return False

    # =========================================================================
    # Fase 4: prompt y generación
    # =========================================================================

    def _build_llm_prompt(self, query: str, llm_context: str) -> str:
        """
        Construye un prompt clínico restringido para Opción B con contexto mixto.
        """
        return f"""You are a clinical evidence assistant.

        Answer the question using ONLY the provided context.

        The context may contain:
            1. PATIENT_STRUCTURED_CONTEXT: real structured clinical data of the patient
            2. LITERATURE_CONTEXT: filtered evidence from the local medical corpus

        Strict rules:
            - Use PATIENT_STRUCTURED_CONTEXT only to understand the clinical case.
            - Use LITERATURE_CONTEXT as the evidence base for the answer.
            - Focus strictly on clinical manifestations unless the question explicitly asks for something else.
            - Do NOT include pathophysiology, diagnosis methodology, or treatment unless explicitly requested.
            - Do NOT infer manifestations from a different organ than the one requested.
            - Do NOT infer acute manifestations from chronic context, or chronic manifestations from acute context.
            - If the literature context does not contain direct evidence for the requested organ or manifestation, say so clearly and briefly.
            - Do NOT guess, extrapolate, or generalize beyond the provided context.
            - If the evidence is partial, state that it is partial.
            - Be concise, clinically precise, and evidence-bounded.

        Question:
        {query}

        Context:
        {llm_context}

        Answer:
        """

    # _call_llm
    def _call_llm(self, prompt: str) -> str:
        """
        Llamada al LLM para Opción B.
        """
        backend = (self.llm_backend or "").lower()

        if backend != "ollama":
            raise RuntimeError(f"LLM no soportado: {self.llm_backend}")

        url = f"{self.llm_base_url}/api/generate"
        payload = {
            "model": self.llm_model_name,
            "prompt": prompt,
            "stream": False,
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib_request.Request(
            url=url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            if self.debug:
                print("\n================ OLLAMA HTTP DEBUG ================")
                print("[DEBUG] url:", url)
                print("[DEBUG] model:", self.llm_model_name)
                print("[DEBUG] timeout:", self.llm_timeout_seconds)

            with urllib_request.urlopen(
                req, timeout=self.llm_timeout_seconds
            ) as response:
                body = response.read().decode("utf-8")
                parsed = json.loads(body)

            if self.debug:
                print("[DEBUG] ollama_raw_response_preview:", body[:1000])

        except urllib_error.HTTPError as exc:
            raise RuntimeError(
                f"HTTPError llamando al LLM: {exc.code} {exc.reason}"
            ) from exc
        except urllib_error.URLError as exc:
            raise RuntimeError(f"URLError llamando al LLM: {exc.reason}") from exc
        except Exception as exc:
            raise RuntimeError(f"Error general llamando al LLM: {exc}") from exc

        answer = parsed.get("response", "")
        if not isinstance(answer, str):
            raise RuntimeError("Respuesta inválida del backend LLM")

        return answer.strip()

    # 8. Fase 5: salida
    def _build_sources(self, retrieved_entries) -> List[Dict[str, Any]]:
        """
        Construye la estructura de fuentes que devuelve la API.
        """
        sources: List[Dict[str, Any]] = []

        for entry in retrieved_entries:
            item = entry["item"]
            meta = entry["meta"]
            text = entry["text"]

            source_entry = {
                "file_name": meta.get("file_name"),
                "file_path": meta.get("file_path"),
                "source": meta.get("source", "bibliography"),
                "block": meta.get("block"),
                "diagnosis_type": meta.get("diagnosis_type"),
                "organ": meta.get("organ"),
                "doc_category": meta.get("doc_category"),
                "year": meta.get("year"),
                "score": round(float(getattr(item, "score", 0.0) or 0.0), 6),
                "heuristic_score": round(entry["heuristic_score"], 6),
                "cross_encoder_raw": round(entry["cross_encoder_raw"], 6),
                "cross_encoder_score": round(entry["cross_encoder_score"], 6),
                "final_score": round(entry["final_score"], 6),
                "score_breakdown": entry["score_breakdown"],
                "text_preview": self._clean_preview(text),
            }

            sources.append(source_entry)

        return sources

    # _fallback_answer
    def _fallback_answer(self, retrieved_entries, query: str) -> str:
        """
        Genera una respuesta usando primero frases clínicas útiles.
        """
        query_lower = query.lower()
        intent = self._detect_query_intent(query)

        organ_sentences: List[str] = []
        diagnosis_sentences: List[str] = []
        neutral_sentences: List[str] = []

        def _is_too_general_for_manifestations(sentence: str) -> bool:
            s = sentence.lower().strip()

            general_patterns = [
                "is an important complication",
                "incidence of",
                "prophylaxis is essential",
                "initial therapy",
                "prognosis is poor",
                "patients undergoing allo-hsct",
            ]

            return any(pattern in s for pattern in general_patterns)

        for entry in retrieved_entries:
            text = entry["text"].strip().replace("\n", " ")
            text = re.sub(r"\s+", " ", text)

            if not text:
                continue

            meta = entry.get("meta", {}) or {}
            block = (meta.get("block") or "").lower()
            chunk_sentences = self._extract_best_clinical_sentences_from_text(text)

            for sentence in chunk_sentences:
                lowered = sentence.lower()

                if wants_acute := ("acute" in query_lower):
                    if (
                        "chronic graft-versus-host disease" in lowered
                        or "chronic graft versus host disease" in lowered
                    ):
                        continue

                if wants_chronic := ("chronic" in query_lower):
                    if (
                        "acute graft-versus-host disease" in lowered
                        or "acute graft versus host disease" in lowered
                    ):
                        continue

                if intent == "manifestations" and _is_too_general_for_manifestations(
                    sentence
                ):
                    continue

                if block == "organ":
                    organ_sentences.append(sentence)
                elif block == "diagnosis":
                    diagnosis_sentences.append(sentence)
                else:
                    neutral_sentences.append(sentence)

        def _dedupe_keep_order(sentences: List[str]) -> List[str]:
            out: List[str] = []
            seen = set()
            for sentence in sentences:
                key = sentence.lower().strip()
                if key not in seen:
                    seen.add(key)
                    out.append(sentence)
            return out

        organ_sentences = _dedupe_keep_order(organ_sentences)
        diagnosis_sentences = _dedupe_keep_order(diagnosis_sentences)
        neutral_sentences = _dedupe_keep_order(neutral_sentences)

        if intent == "manifestations":
            final_sentences = (
                organ_sentences[:3] + diagnosis_sentences[:1] + neutral_sentences[:1]
            )
        else:
            final_sentences = (
                diagnosis_sentences[:2] + organ_sentences[:2] + neutral_sentences[:1]
            )

        final_sentences = _dedupe_keep_order(final_sentences)

        if final_sentences:
            return " ".join(final_sentences[:4])

        if retrieved_entries:
            best_text = retrieved_entries[0]["text"].strip().replace("\n", " ")
            best_text = re.sub(r"\s+", " ", best_text)

            cleaned_sentences = self._extract_best_clinical_sentences_from_text(
                best_text
            )
            cleaned_sentences = _dedupe_keep_order(cleaned_sentences)

            if intent == "manifestations":
                cleaned_sentences = [
                    s
                    for s in cleaned_sentences
                    if not _is_too_general_for_manifestations(s)
                ]

            if cleaned_sentences:
                return " ".join(cleaned_sentences[:3])

            best_text = self._clean_preview(best_text, max_len=500)
            best_text = self._strip_editorial_phrases(best_text)
            best_sentences = re.split(r"(?<=[\.\!\?])\s+", best_text)

            fallback_sentences = [
                s.strip()
                for s in best_sentences
                if s.strip()
                and not self._looks_like_reference_sentence(s)
                and not self._is_editorial_sentence(s)
                and len(s.strip().split()) >= 8
            ]

            if intent == "manifestations":
                fallback_sentences = [
                    s
                    for s in fallback_sentences
                    if not _is_too_general_for_manifestations(s)
                ]

            if fallback_sentences:
                return " ".join(fallback_sentences[:3])

            if best_text:
                return best_text

        return (
            "Relevant local literature was retrieved, but no clean clinical "
            "summary could be assembled."
        )

    def _clean_preview(self, text: str, max_len: int = 500) -> str:
        """
        Limpia un preview de texto para logs o salida compacta.
        """
        if not text:
            return ""

        text = text.replace("\n", " ")
        text = re.sub(r"\s+", " ", text).strip()

        text = re.sub(
            r"(Department of|University of|Faculty of Medicine|Perelman School of Medicine|Division of Hematology).*",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()

        return text[:max_len]

    def _debug_enabled(self) -> bool:
        return bool(self.debug) and self.debug_mode in {"academic", "full"}

    def _debug_full(self) -> bool:
        return bool(self.debug) and self.debug_mode == "full"

    def _debug_academic(self) -> bool:
        return bool(self.debug) and self.debug_mode == "academic"

    # Métodos nuevos debug
    def _debug_header(
        self,
        test_label: str,
        query: str,
        paciente_id: Optional[int],
        mode: str,
        requested_organs: List[str],
        intent: str,
        target_diag: str,
        is_specific_query: bool,
    ) -> None:
        if not self._debug_academic():
            return

        print("\n" + "=" * 60)
        print("RAG PIPELINE SUMMARY")
        print("=" * 60)
        print(f"Test: {test_label}")
        print(f"Mode: {mode}")
        print(f"Query: {query}")
        print(f"Paciente ID: {paciente_id}")
        print(f"Intent: {intent}")
        print(f"Target diagnosis: {target_diag or '-'}")
        print(f"Requested organs: {requested_organs or []}")
        print(f"Specific query: {is_specific_query}")

    def _debug_retrieval_summary(
        self,
        subqueries: List[str],
        raw_nodes,
    ) -> None:
        if not self._debug_academic():
            return

        print("\n" + "-" * 60)
        print("RETRIEVAL SUMMARY")
        print("-" * 60)
        print(f"Subqueries launched: {len(subqueries)}")
        for i, sq in enumerate(subqueries, 1):
            print(f"  {i}. {sq}")
        print(f"Raw nodes retrieved: {len(raw_nodes)}")

    def _debug_selection_summary(
        self,
        cleaned_entries: List[Dict[str, Any]],
    ) -> None:
        if not self._debug_academic():
            return

        print("\n" + "-" * 60)
        print("SELECTION SUMMARY")
        print("-" * 60)
        print(f"Cleaned entries selected: {len(cleaned_entries)}")

        for i, entry in enumerate(cleaned_entries[:5], 1):
            meta = entry.get("meta", {}) or {}
            print(
                f"{i}. {meta.get('file_name')} | "
                f"block={meta.get('block')} | "
                f"diag={meta.get('diagnosis_type') or '-'} | "
                f"organ={meta.get('organ') or '-'} | "
                f"final={round(entry.get('final_score', 0.0), 4)}"
            )

    def _debug_context_summary(
        self,
        sql_context: str,
        literature_context: str,
        combined_context: str,
        used_sql: bool,
    ) -> None:
        if not self._debug_academic():
            return

        print("\n" + "-" * 60)
        print("CONTEXT SUMMARY")
        print("-" * 60)
        print(f"SQL context used: {'yes' if used_sql else 'no'}")
        print(f"SQL context length: {len(sql_context or '')}")
        print(f"Literature context length: {len(literature_context or '')}")
        print(f"Combined context length: {len(combined_context or '')}")

    def _debug_llm_summary(
        self,
        llm_used: bool,
        prompt: str,
        llm_answer: str,
        fallback_reason: Optional[str] = None,
    ) -> None:
        if not self._debug_academic():
            return

        print("\n" + "-" * 60)
        print("LLM SUMMARY")
        print("-" * 60)
        print(f"LLM used: {'yes' if llm_used else 'no'}")
        print(f"Model: {self.llm_model_name}")
        print(f"Prompt length: {len(prompt or '')}")
        print(f"Answer length: {len(llm_answer or '')}")
        if fallback_reason:
            print(f"Fallback reason: {fallback_reason}")

    def _debug_outcome(
        self,
        answer: str,
        sources: List[Dict[str, Any]],
    ) -> None:
        if not self._debug_academic():
            return

        print("\n" + "-" * 60)
        print("FINAL OUTCOME")
        print("-" * 60)
        print("Answer:")
        print(answer.strip()[:1200] if answer else "(empty)")
        print("\nSources:")
        for src in sources[:5]:
            print(f"- {src.get('file_name')}")


if __name__ == "__main__":
    rag = RAGService(
        data_dir="data/core",
        chunk_size=512,
        chunk_overlap=80,
        similarity_top_k=12,
        use_hybrid=True,
        debug=True,
        embed_model_name="BAAI/bge-small-en-v1.5",
        use_cross_encoder=True,
        cross_encoder_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        llm_backend="ollama",
        llm_base_url="http://localhost:11434",
        llm_model_name="mistral",
        llm_timeout_seconds=90,
        llm_max_context_sentences=8,
    )

    test_queries = [
        # T01 aguda general
        "What are the clinical manifestations of acute graft versus host disease?",
        # T02 aguda por órgano
        "What are the skin manifestations of acute graft versus host disease?",
        "What are the gastrointestinal manifestations of acute graft versus host disease?",
        "What are the liver manifestations of acute graft versus host disease?",
        # T03 crónica general
        "What are the clinical manifestations of chronic graft versus host disease?",
        # T04 crónica por órgano
        "What are the ocular manifestations of chronic graft versus host disease?",
        "What are the oral manifestations of chronic graft versus host disease?",
        "What are the pulmonary manifestations of chronic graft versus host disease?",
        "What are the skin manifestations of chronic graft versus host disease?",
        # T05 combinación multiórgano crónica
        "What are the ocular and oral manifestations of chronic graft versus host disease?",
    ]

    for i, q in enumerate(test_queries, 1):
        print("\n" + "=" * 80)
        print(f"TEST {i}")
        print("=" * 80)
        print("QUERY:", q)

        result = rag.query(q, mode="option_b")

        print("\nANSWER:")
        print(result.get("answer"))

        print("\nSOURCES:")
        for s in result.get("sources", [])[:5]:
            print("-", s.get("file_name"))
