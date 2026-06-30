# Evidence Service

## Descripción

Evidence Service es un microservicio desarrollado en Python cuyo objetivo es generar informes clínicos fundamentados en literatura científica para apoyar el diagnóstico de la Enfermedad Injerto Contra Huésped (EICH).

El servicio implementa una arquitectura **Retrieval-Augmented Generation (RAG)** que combina recuperación semántica de documentos científicos con modelos de lenguaje (LLM) ejecutados localmente.

Su finalidad no es realizar el diagnóstico clínico —responsabilidad del sistema experto— sino proporcionar una explicación basada en evidencia científica que respalde el resultado obtenido por el motor de inferencia.

El servicio se comunica con el Sistema de Gestión Clínica mediante una API REST.

---

# Arquitectura

El servicio se divide conceptualmente en dos etapas independientes:

```
Solicitud REST
        │
        ▼
 Construcción del contexto clínico
        │
        ▼
──────────────────────────────────────
        RETRIEVAL
──────────────────────────────────────
        │
        ▼
 Recuperación documental
        │
        ▼
 Selección de evidencia científica
        │
        ▼
──────────────────────────────────────
     GENERACIÓN LLM
──────────────────────────────────────
        │
        ▼
 Construcción del Prompt
        │
        ▼
 Generación del informe
        │
        ▼
Respuesta JSON
```

---

# Filosofía del sistema

Uno de los principales problemas al utilizar modelos de lenguaje en entornos clínicos consiste en que el modelo puede generar respuestas basadas únicamente en su conocimiento preentrenado.

En un dominio altamente especializado como la Enfermedad Injerto Contra Huésped, este comportamiento puede producir respuestas desactualizadas, incompletas o sin respaldo bibliográfico.

Para minimizar este problema, el sistema utiliza una estrategia Retrieval-Augmented Generation (RAG), donde el modelo genera la respuesta únicamente después de recuperar información científica relevante del corpus documental.

De esta forma el modelo no responde "por memoria", sino utilizando evidencia recuperada dinámicamente.

---

# Fase 1. Recuperación documental (Retrieval)

La primera etapa consiste en localizar los documentos científicos más relevantes para el caso clínico recibido.

El servicio recibe información estructurada procedente del Sistema de Gestión Clínica, incluyendo entre otros:

- diagnóstico inferido,
- síntomas activos,
- órganos afectados,
- puntuaciones NIH.

A partir de este contexto se construyen consultas específicas orientadas a recuperar la literatura científica más relevante.

La recuperación combina diferentes estrategias con el objetivo de mejorar tanto la precisión como la cobertura del contexto recuperado.

Entre las técnicas empleadas destacan:

- indexación vectorial mediante embeddings;
- búsqueda híbrida combinando similitud semántica y búsqueda léxica;
- recuperación específica por órgano afectado;
- filtrado y limpieza del contexto recuperado;
- selección diversa de fragmentos para evitar redundancia documental.

El resultado de esta etapa es un conjunto reducido de fragmentos científicos que representan la evidencia utilizada durante la generación del informe.

---

# Fase 2. Generación mediante LLM

Una vez recuperada la evidencia científica, el sistema construye un prompt estructurado que incorpora:

- contexto clínico del paciente;
- diagnóstico inferido por el sistema experto;
- literatura científica recuperada;
- instrucciones para la generación del informe.

Este prompt se envía a un modelo de lenguaje ejecutado localmente.

El modelo genera un informe clínico estructurado cuyo objetivo es justificar el diagnóstico mediante evidencia científica y proporcionar una explicación comprensible para el profesional sanitario.

La generación siempre depende del contexto recuperado durante la fase anterior, evitando respuestas desvinculadas de la literatura científica disponible.

---

# Corpus científico

El servicio trabaja sobre un corpus documental compuesto por publicaciones científicas especializadas en EICH.

Los documentos se organizan por áreas clínicas y órganos afectados, permitiendo realizar recuperaciones dirigidas según las características del caso clínico.

Durante la fase de inicialización el corpus se procesa automáticamente para construir los índices necesarios para la recuperación semántica.

---

# Organización del proyecto

```
Evidence-Service
│
├── app.py                 # API REST
├── rag_service.py         # Coordinador principal del pipeline
├── rag/
│   ├── retrieval.py
│   ├── prompts.py
│   ├── retrievers.py
│   ├── rerank.py
│   ├── formatter.py
│   └── ...
├── data/
│   ├── core/
│   └── ...
├── catalog.csv
├── schemas.py
└── requirements.txt
```

La mayor parte de la lógica del sistema se concentra en el módulo `rag`, donde se implementan las distintas fases del pipeline de recuperación y generación.

---

# Integración con el Sistema de Gestión Clínica

El servicio actúa como un sistema desacoplado del sistema principal.

La comunicación se realiza mediante una API REST.

Flujo de trabajo:

1. El Sistema de Gestión Clínica obtiene un diagnóstico mediante el sistema experto.
2. Se envía una petición HTTP con el contexto clínico.
3. Evidence Service recupera la literatura científica relevante.
4. Se genera el informe mediante el modelo de lenguaje.
5. El informe se devuelve en formato JSON.
6. El Sistema de Gestión Clínica almacena el resultado y lo presenta al médico.

---

# Tecnologías utilizadas

- Python
- Flask
- LlamaIndex
- Ollama
- Sentence Transformers
- Cross Encoder
- BM25
- FAISS / Vector Store
- Retrieval-Augmented Generation (RAG)

---

# Ejecución

## Crear entorno virtual

```bash
python -m venv venv
```

## Activar entorno

Linux / macOS

```bash
source venv/bin/activate
```

Windows

```bash
venv\Scripts\activate
```

---

## Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## Ejecutar el servicio

```bash
python app.py
```

El servicio quedará disponible para recibir peticiones http REST desde el Sistema de Gestión Clínica.

---

# Consideraciones

Este proyecto implementa un enfoque híbrido donde el razonamiento clínico y la generación textual permanecen completamente desacoplados.

El diagnóstico es responsabilidad exclusiva del sistema experto desarrollado en el Sistema de Gestión Clínica.

Evidence Service actúa como un sistema de apoyo documental cuya finalidad es recuperar evidencia científica y generar informes clínicos explicativos basados en dicha evidencia.
