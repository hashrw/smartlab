from typing import Any, Dict, List

API_VERSION = "v1"
REQUIRED_TOP_LEVEL_KEYS = {"caso_clinico", "resultado_inferencia"}


def validate_evidence_request(data: Any) -> Dict[str, Any]:
    """
    Valida el cuerpo de la request y devuelve el payload si es correcto.

    Lanza ValueError si el contrato no se cumple.
    """
    errors: List[str] = []

    if not isinstance(data, dict):
        raise ValueError("Request body must be a JSON object.")

    missing = REQUIRED_TOP_LEVEL_KEYS - set(data.keys())
    if missing:
        errors.append(f"Missing top-level keys: {', '.join(sorted(missing))}")

    caso = data.get("caso_clinico")
    resultado = data.get("resultado_inferencia")

    if caso is not None and not isinstance(caso, dict):
        errors.append("'caso_clinico' must be an object.")

    if resultado is not None and not isinstance(resultado, dict):
        errors.append("'resultado_inferencia' must be an object.")

    if isinstance(caso, dict):
        if "paciente_id" not in caso:
            errors.append("'caso_clinico.paciente_id' is required.")
        if "active_aliases_canonical" not in caso:
            errors.append("'caso_clinico.active_aliases_canonical' is required.")
        if "organo_score_nih_by_nombre" not in caso:
            errors.append("'caso_clinico.organo_score_nih_by_nombre' is required.")

        if "active_aliases_canonical" in caso and not isinstance(
            caso["active_aliases_canonical"], list
        ):
            errors.append("'caso_clinico.active_aliases_canonical' must be an array.")

        if "organo_score_nih_by_nombre" in caso and not isinstance(
            caso["organo_score_nih_by_nombre"], dict
        ):
            errors.append(
                "'caso_clinico.organo_score_nih_by_nombre' must be an object."
            )

    if isinstance(resultado, dict):
        if "status" not in resultado:
            errors.append("'resultado_inferencia.status' is required.")

        if resultado.get("status") == "match" and "diagnostico_id" not in resultado:
            errors.append(
                "'resultado_inferencia.diagnostico_id' is required when status=match."
            )

    if errors:
        raise ValueError(" | ".join(errors))

    return data


def build_evidence_response(
    *,
    summary: str,
    query_summary: Dict[str, Any],
    citations: List[Dict[str, Any]] | None = None,
    evidence_map: List[Dict[str, Any]] | None = None,
    warnings: List[str] | None = None,
    generated_at: str | None = None,
) -> Dict[str, Any]:
    """
    Construye la respuesta estándar del endpoint /evidence.
    """
    return {
        "api_version": API_VERSION,
        "status": "ok",
        "generated_at": generated_at,
        "summary": summary,
        "query_summary": query_summary,
        "citations": citations or [],
        "evidence_map": evidence_map or [],
        "warnings": warnings or [],
    }


def build_error_response(
    *,
    code: str,
    message: str,
    api_version: str | None = None,
    details: List[str] | None = None,
) -> Dict[str, Any]:
    """
    Construye la respuesta de error.
    """
    return {
        "api_version": api_version or API_VERSION,
        "status": "error",
        "error": {
            "code": code,
            "message": message,
            "details": details or [],
        },
    }
