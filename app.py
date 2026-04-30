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

ORGAN_QUERY_MAP = {
    "Hígado": "liver",
    "Higado": "liver",
    "Tracto gastrointestinal": "gastrointestinal tract",
    "Piel": "skin",
    "Ojos": "ocular",
    "Boca": "oral",
    "Pulmón": "pulmonary",
    "Pulmon": "pulmonary",
    "Pulmones": "pulmonary",
}

app = Flask(__name__)

# Instancia única al arrancar la app.
# En integración Laravel se usa una query clínica ya generada desde el DSS.
# integration_mode=True evita el modo pesado de multiquery usado en pruebas T01-T05.
rag = RAGService(
    data_dir="data/core",
    default_mode="option_b",
    integration_mode=True,
    debug=False,
    similarity_top_k=8,
    use_hybrid=True,
    use_cross_encoder=True,
    llm_timeout_seconds=120,
    llm_max_context_sentences=3,
)


def iso_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def build_query(caso: dict) -> str:
    """
    Construye una query de respaldo a partir del caso clínico.

    En el flujo final Laravel envía una query dinámica completa.
    Esta función queda como error técnico si la petición no lleva query.
    """
    sintomas = caso.get("active_aliases_canonical", [])
    organos = caso.get("organo_score_nih_by_nombre", {})

    mapped_terms = [
        ALIAS_QUERY_MAP.get(alias, alias.replace("_", " ")) for alias in sintomas[:6]
    ]

    organ_terms = [ORGAN_QUERY_MAP.get(org, org) for org in list(organos.keys())[:3]]

    score_terms = []
    for org, score in list(organos.items())[:3]:
        organ_name = ORGAN_QUERY_MAP.get(org, org)
        score_terms.append(f"{organ_name} NIH score {score}")

    base_terms = (
        mapped_terms
        + organ_terms
        + score_terms
        + ["graft versus host disease", "GVHD", "clinical evidence"]
    )

    return " ".join(term for term in base_terms if term).strip()


def build_citations_from_sources(sources: list[dict]) -> list[dict]:
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
    if not citations:
        return []

    titles = [citation.get("title") for citation in citations if citation.get("title")]

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
        paciente_id = caso.get("paciente_id")
        resultado_inferencia = (
            validated.get("resultado_inferencia", {})
            if isinstance(validated, dict)
            else {}
        )

        query = (
            validated.get("query")
            or build_query(caso)
            or "Analyze the clinical case for graft versus host disease evidence"
        )

        rag_result = rag.generate_clinical_report(
            caso_clinico=caso,
            resultado_inferencia=resultado_inferencia,
            paciente_id=paciente_id,
        )

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
                "llm_model": rag_result.get("llm_model"),
                "paciente_id": paciente_id,
                "fallback_reason": rag_result.get("fallback_reason"),
                "timestamp": iso_now(),
            },
            generated_at=iso_now(),
        )

        return jsonify(response), 200

    except Exception as exc:
        return (
            jsonify(
                build_error_response(
                    code="EVIDENCE_ERROR",
                    message=str(exc),
                    api_version=API_VERSION,
                )
            ),
            500,
        )


@app.route("/clinical-report", methods=["POST"])
def clinical_report():
    data = request.get_json(silent=True) or {}

    try:
        validated = validate_evidence_request(data)

        caso = validated.get("caso_clinico", {}) if isinstance(validated, dict) else {}
        paciente_id = caso.get("paciente_id")

        rag_result = rag.generate_clinical_report(
            caso_clinico=caso,
            paciente_id=paciente_id,
        )

        response = {
            "api_version": API_VERSION,
            "status": "ok",
            "generated_at": iso_now(),
            "clinical_report": rag_result.get("clinical_report", {}),
            "traceability": {
                "sources": build_citations_from_sources(rag_result.get("sources", [])),
                "warnings": rag_result.get("warnings", []),
                "llm_used": rag_result.get("llm_used", False),
                "llm_model": rag_result.get("llm_model"),
                "fallback_reason": rag_result.get("fallback_reason"),
                "paciente_id": paciente_id,
            },
        }

        return jsonify(response), 200

    except Exception as exc:
        return (
            jsonify(
                build_error_response(
                    code="CLINICAL_REPORT_ERROR",
                    message=str(exc),
                    api_version=API_VERSION,
                )
            ),
            500,
        )


# endpoint para comprobar que el servicio esté levantado
@app.get("/health")
def health():
    return (
        jsonify(
            {
                "service": "evidence-service",
                "status": "ok",
                "api_version": API_VERSION,
                "rag_default_mode": "option_b",
                "integration_mode": True,
            }
        ),
        200,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
