from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import pymysql
from pymysql.cursors import DictCursor


class SQLService:
    """
    Servicio de acceso de solo lectura a la BD Laravel.

    Objetivo:
    - Extraer contexto clínico estructurado del paciente
    - Mantener este acceso separado de rag_service.py
    - Facilitar la futura construcción de contexto mixto:
      1. corpus bibliográfico
      2. datos estructurados reales del paciente

    Nota:
    - Esta implementación usa conexión simple a MySQL, defendible
      para el alcance del TFG y suficiente para validar la arquitectura.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        connect_timeout: int = 5,
        read_timeout: int = 10,
        write_timeout: int = 10,
        charset: str = "utf8mb4",
        debug: bool = True,
    ) -> None:
        self.host = host or os.getenv("DB_HOST", "localhost")
        self.port = int(port or os.getenv("DB_PORT", 3306))
        self.database = database or os.getenv("DB_DATABASE", "cgis")
        self.user = user or os.getenv("DB_USERNAME", "sail")
        self.password = password or os.getenv("DB_PASSWORD", "password")
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.write_timeout = write_timeout
        self.charset = charset
        self.debug = debug

    # =========================================================================
    # Conexión
    # =========================================================================

    def _get_connection(self):
        """
        Crea una conexión MySQL de solo lectura lógico-funcional.

        Nota:
        - No forzamos una sesión read-only a nivel servidor porque depende
          de permisos/configuración del entorno.
        - A nivel de código, este servicio solo expone operaciones SELECT.
        """
        return pymysql.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database,
            charset=self.charset,
            cursorclass=DictCursor,
            connect_timeout=self.connect_timeout,
            read_timeout=self.read_timeout,
            write_timeout=self.write_timeout,
            autocommit=True,
        )

    def test_connection(self) -> Dict[str, Any]:
        """
        Prueba conexión básica y devuelve metadatos mínimos.
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT DATABASE() AS db_name, NOW() AS server_time")
                    row = cursor.fetchone()

            return {
                "ok": True,
                "database": row.get("db_name") if row else self.database,
                "server_time": str(row.get("server_time")) if row else None,
            }

        except Exception as exc:
            return {
                "ok": False,
                "error": str(exc),
            }

    # =========================================================================
    # Queries base del contexto clínico
    # =========================================================================

    def get_patient_basic_info(self, paciente_id: int) -> Optional[Dict[str, Any]]:
        """
        Devuelve información base del paciente combinando:

        - users: identidad y datos base del usuario
        - pacientes: datos clínicos básicos del paciente

        Relación real:
        - users.paciente_id -> pacientes.id
        - users.tipo_usuario_id = 2 identifica usuarios paciente
        """

        sql = """
            SELECT
            users.id AS usuario_id,
            users.name,
            users.apellidos,
            users.email,
            users.telefono,
            users.tipo_usuario_id,
            users.created_at AS user_created_at,
            users.updated_at AS user_updated_at,
            pacientes.id AS paciente_id,
            pacientes.nuhsa,
            pacientes.peso,
            pacientes.fecha_nacimiento,
            pacientes.altura,
            pacientes.sexo,
            pacientes.created_at AS paciente_created_at,
            pacientes.updated_at AS paciente_updated_at,
            pacientes.deleted_at AS paciente_deleted_at
            FROM users
            INNER JOIN pacientes
                ON pacientes.id = users.paciente_id
            WHERE pacientes.id = %s
              AND users.tipo_usuario_id = 2
            LIMIT 1
        """

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql, (paciente_id,))
                    return cursor.fetchone()
        except Exception as exc:
            if self.debug:
                print(f"[SQL] get_patient_basic_info error: {exc}")
            return None

    def get_active_symptoms(self, paciente_id: int) -> List[Dict[str, Any]]:
        """
        Recupera síntomas activos del paciente desde paciente_sintoma.
        """
        sql = """
            SELECT
                sintomas.id,
                sintomas.sintoma,
                sintomas.manif_clinica,
                paciente_sintoma.fecha_observacion,
                paciente_sintoma.activo,
                paciente_sintoma.fuente
            FROM paciente_sintoma
            INNER JOIN sintomas
                ON sintomas.id = paciente_sintoma.sintoma_id
            WHERE paciente_sintoma.paciente_id = %s
            AND paciente_sintoma.activo = 1
            ORDER BY sintomas.id ASC
        """

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql, (paciente_id,))
                    rows = cursor.fetchall()
                    return rows or []
        except Exception as exc:
            if self.debug:
                print(f"[SQL] get_active_symptoms error: {exc}")
            return []

    def get_organ_scores(self, paciente_id: int) -> List[Dict[str, Any]]:
        """
        Recupera órganos evaluados y score NIH del paciente.
        """
        sql = """
            SELECT
                organos.id,
                organos.nombre,
                organo_paciente.score_nih,
                organo_paciente.fecha_evaluacion,
                organo_paciente.sintomas_asociados,
                organo_paciente.comentario
            FROM organo_paciente
            INNER JOIN organos
                ON organos.id = organo_paciente.organo_id
            WHERE organo_paciente.paciente_id = %s
            ORDER BY organos.id ASC
        """

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql, (paciente_id,))
                    rows = cursor.fetchall()
                    return rows or []
        except Exception as exc:
            if self.debug:
                print(f"[SQL] get_organ_scores error: {exc}")
            return []

    def get_recent_diagnoses(
        self, paciente_id: int, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Recupera diagnósticos recientes asociados al paciente.
        """
        sql = """
            SELECT
                diagnosticos.id,
                diagnosticos.fecha_diagnostico,
                diagnosticos.tipo_enfermedad,
                diagnosticos.estado_injerto,
                diagnosticos.observaciones,
                diagnosticos.grado_eich,
                diagnosticos.escala_karnofsky,
                diagnosticos.regla_decision_id,
                diagnosticos.created_at
            FROM diagnostico_paciente
            INNER JOIN diagnosticos
                ON diagnosticos.id = diagnostico_paciente.diagnostico_id
            WHERE diagnostico_paciente.paciente_id = %s
            ORDER BY
            diagnosticos.fecha_diagnostico DESC,
            diagnosticos.created_at DESC
            LIMIT %s
        """

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql, (paciente_id, limit))
                    rows = cursor.fetchall()
                return rows or []
        except Exception as exc:
            if self.debug:
                print(f"[SQL] get_recent_diagnoses error: {exc}")
            return []

    # =========================================================================
    # Construcción de contexto estructurado
    # =========================================================================

    def get_patient_context(self, paciente_id: int) -> Dict[str, Any]:
        """
        Recupera el contexto clínico estructurado completo del paciente.

        Este es el método principal que luego consumirá rag_service.py.
        """
        patient = self.get_patient_basic_info(paciente_id)
        active_symptoms = self.get_active_symptoms(paciente_id)
        organ_scores = self.get_organ_scores(paciente_id)
        recent_diagnoses = self.get_recent_diagnoses(paciente_id)

        return {
            "paciente_id": paciente_id,
            "patient": patient,
            "active_symptoms": active_symptoms,
            "organ_scores": organ_scores,
            "recent_diagnoses": recent_diagnoses,
        }

    def build_patient_context_text(self, paciente_id: int) -> str:
        """
        Convierte el contexto estructurado del paciente a texto compacto.
        """
        context = self.get_patient_context(paciente_id)

        lines: List[str] = [f"Patient ID: {paciente_id}"]

        patient = context.get("patient")
        if patient:
            sexo = patient.get("sexo")
            fecha_nacimiento = patient.get("fecha_nacimiento")
            nuhsa = patient.get("nuhsa")

            if nuhsa:
                lines.append(f"NUHSA: {nuhsa}")
            if sexo:
                lines.append(f"Sex: {sexo}")
            if fecha_nacimiento:
                lines.append(f"Birth date: {fecha_nacimiento}")

        symptoms = context.get("active_symptoms", [])
        if symptoms:
            symptom_names = [s.get("sintoma") for s in symptoms if s.get("sintoma")]
            if symptom_names:
                lines.append("Active symptoms: " + ", ".join(symptom_names))

        organ_scores = context.get("organ_scores", [])
        if organ_scores:
            organ_fragments = []
            for row in organ_scores:
                organ_name = row.get("nombre")
                score_nih = row.get("score_nih")
                if organ_name is not None and score_nih is not None:
                    organ_fragments.append(f"{organ_name} (NIH score: {score_nih})")
            if organ_fragments:
                lines.append("Organ scores: " + ", ".join(organ_fragments))

        diagnoses = context.get("recent_diagnoses", [])
        if diagnoses:
            diagnosis_fragments = []
            for d in diagnoses:
                tipo = d.get("tipo_enfermedad")
                fecha = d.get("fecha_diagnostico")
            if tipo and fecha:
                diagnosis_fragments.append(f"{tipo} ({fecha})")
            elif tipo:
                diagnosis_fragments.append(str(tipo))
            if diagnosis_fragments:
                lines.append("Recent diagnoses: " + ", ".join(diagnosis_fragments))

        return "\n".join(lines)

    # =========================================================================
    # Utilidades de inspección
    # =========================================================================

    def list_tables(self) -> List[str]:
        """
        Lista tablas disponibles en la BD actual.
        Muy útil para validar nombres reales antes de ajustar queries.
        """
        sql = "SHOW TABLES"

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql)
                    rows = cursor.fetchall()

            if not rows:
                return []

            table_names: List[str] = []
            for row in rows:
                table_names.extend(list(row.values()))

            return table_names

        except Exception as exc:
            if self.debug:
                print(f"[SQL] list_tables error: {exc}")
            return []

    def describe_table(self, table_name: str) -> List[Dict[str, Any]]:
        """
        Describe una tabla concreta para inspección rápida.
        """
        sql = f"DESCRIBE `{table_name}`"

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql)
                    rows = cursor.fetchall()
                    return rows or []
        except Exception as exc:
            if self.debug:
                print(f"[SQL] describe_table error ({table_name}): {exc}")
            return []


if __name__ == "__main__":
    sql_service = SQLService(debug=True)

    print("=== TEST CONNECTION ===")
    print(sql_service.test_connection())

    print("\n=== TABLES ===")
    print(sql_service.list_tables()[:20])

    # Ajusta el ID si quieres probar uno real.
    test_patient_id = 1

    print("\n=== PATIENT CONTEXT ===")
    print(sql_service.get_patient_context(test_patient_id))

    print("\n=== PATIENT CONTEXT TEXT ===")
    print(sql_service.build_patient_context_text(test_patient_id))
