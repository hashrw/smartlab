from datetime import datetime, timezone

from flask import Flask, jsonify, request

from rag_service import RAGService
from schemas import (
    API_VERSION,
    build_evidence_response,
    build_error_response,
    validate_evidence_request,
)

ALIAS_QUERY_MAP = {
    "o1_diarrea_con_sangre": "bloody diarrhea",
    "o1_dolor_abdominal": "abdominal pain",
    "o1_nauseas": "nausea",
    "o1_vomitos": "vomiting",
    "o2_alt_elevada": "elevated ALT",
    "o2_fosfatasa_alcalina_elevada": "elevated alkaline phosphatase",
    "o2_hiperbilirrubinemia": "hyperbilirubinemia",
}

ORGAN_QUERY_MAP = {
    "Hígado": "liver",
    "Tracto gastrointestinal": "gastrointestinal tract",
    "Piel": "skin",
    "Ojos": "eyes",
    "Boca": "mouth",
    "Pulmones": "lungs",
}

app = Flask(__name__)

# Instancia única al arrancar la app, no en cada request.
# La salida activa del sistema queda fijada en Opción B,
# aunque Opción A sigue siendo la base interna del retrieval.
rag = RAGService(default_mode="option_b")


def iso_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def build_query(caso: dict) -> str:
    """
    Construye una query bibliográfica inicial a partir del caso clínico.

    Nota:
    - Esta query sigue siendo útil para el retrieval bibliográfico.
    - En la siguiente fase, el SQL connector añadirá contexto estructurado
      del paciente y esta función quedará como una pieza más del contexto global,
      no como única fuente de señal.
    """
    sintomas = caso.get("active_aliases_canonical", [])
    organos = caso.get("organo_score_nih_by_nombre", {})

    mapped_terms = [
        ALIAS_QUERY_MAP.get(alias, alias.replace("_", " ")) for alias in sintomas[:4]
    ]

    organ_terms = [ORGAN_QUERY_MAP.get(org, org) for org in list(organos.keys())[:2]]

    base_terms = mapped_terms + organ_terms + ["graft versus host disease", "GVHD"]

    return " ".join(base_terms).strip()


def build_citations_from_sources(sources: list[dict]) -> list[dict]:
    """
    Convierte las sources del RAG a una estructura de citas compatible
    con la respuesta del endpoint.
    """
    citations = []

    for source in sources:
        citations.append(
            {
                "title": source.get("file_name")
                or source.get("title")
                or "Unknown source",
                "source": source.get("source", "bibliography"),
                "year": source.get("year"),
                "block": source.get("block"),
                "diagnosis_type": source.get("diagnosis_type"),
                "organ": source.get("organ"),
                "doc_category": source.get("doc_category"),
            }
        )

    return citations


def build_evidence_map(citations: list[dict]) -> list[dict]:
    """
    Mapa simple de evidencia para mantener compatibilidad con el contrato actual.

    Nota:
    - Sigue siendo una capa ligera.
    - Se podrá enriquecer más adelante cuando entre el SQL connector
      y exista contexto mixto bibliográfico + clínico estructurado.
    """
    if not citations:
        return []

    titles = [c.get("title") for c in citations if c.get("title")]

    return [
        {
            "claim": "Retrieved local clinical evidence relevant to the current GVHD-related query.",
            "citation_titles": titles,
        }
    ]


@app.route("/evidence", methods=["POST"])
def evidence():
    data = request.get_json(silent=True) or {}

    try:
        validated = validate_evidence_request(data)

        caso = validated.get("caso_clinico", {}) if isinstance(validated, dict) else {}

        query = data.get("query") or build_query(caso) or "Analyze the clinical case"

        # Opción B queda explícita como salida activa del sistema.
        # Opción A sigue usándose internamente como base del retrieval.
        rag_result = rag.query(query, mode="option_b")

        sources = rag_result.get("sources", [])
        citations = build_citations_from_sources(sources)

        response = build_evidence_response(
            summary=rag_result.get("answer", ""),
            citations=citations,
            evidence_map=build_evidence_map(citations),
            warnings=[],
            query_summary={
                "generated_query": query,
                "mode": rag_result.get("mode", "option_b"),
                "llm_used": rag_result.get("llm_used", False),
                "timestamp": iso_now(),
            },
            generated_at=iso_now(),
        )

        return jsonify(response), 200

    except Exception as e:
        return (
            jsonify(
                build_error_response(
                    code="EVIDENCE_ERROR",
                    message=str(e),
                    api_version=API_VERSION,
                )
            ),
            500,
        )


@app.get("/health")
def health():
    return (
        jsonify(
            {
                "service": "evidence-service",
                "status": "ok",
                "api_version": API_VERSION,
                "rag_default_mode": "option_b",
            }
        ),
        200,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
