# SEEG Recrutement — Dashboard & Scoring

Plateforme Streamlit pour centraliser, évaluer et piloter les candidatures SEEG via Supabase, avec conformité documentaire fiabilisée, jauges de scores, graphiques et synthèse IA cadrée (sans hallucination documentaire).

---

## Problème à résoudre

- **[Fragmentation des données]** Candidatures, profils, postes, documents et scores dispersés (Supabase, fichiers, éventuelle base locale).
- **[Manque de visibilité]** Difficulté à avoir une vue opérationnelle du pipeline (KPI, tendances, multi‑postes, distribution des scores).
- **[Évaluation hétérogène]** Scores peu homogènes et parfois inexpliqués (fit, complétude, final, MTP).
- **[Conformité documentaire incertaine]** La vérification des pièces (CV, Diplôme, Lettre) est manuelle et source d’erreurs.
- **[Synthèses IA risquant d’halluciner]** Les résumés peuvent mentionner des pièces ou contenus inexistants.
- **[Vision stratégique peu intégrée]** Le texte Vision SEEG n’est pas systématiquement injecté dans l’analyse d’adéquation/adhérence.

---

## Solution (hyper détaillée)

### 1) Architecture et flux de données

- **Source de vérité:** Supabase (tables: `applications`, `scores` ou fallback SQLite, `users`/`candidates`, `candidate_profiles`, `positions`/`job_offers`, `application_documents`).
- **Chargement consolidé:** `load_data()` dans `streamlit_app/app.py` interroge Supabase (et SQLite optionnellement pour `scores`).
- **Enrichissement:** `build_enriched()` joint les tables, harmonise les clés (`candidate_id`/`user_id`, `position_id`/`job_offer_id`) et agrège les liens documentaires par `application_id` (`doc_links`).
- **Vision SEEG:** injectée depuis `VISION_TEXT`/`VISION_FILE` via `seeg_core/config.py` (fallback sur `scripts/vision_seeg.md`).

### 2) Fonctionnalités principales

- **Dashboard Accueil (`page_home()`)**
  - KPIs: nb. de candidats, nb. de candidatures, nb./% multi‑postes.
  - Graphiques: H/F, multi‑postes (Top 20), candidatures par poste, histogramme/boxplot des scores, tendance mensuelle.

- **Page Candidat (`page_candidate()`)**
  - Jauges alignées (5): Complétude, Adéquation, Score final, Adhérence MTP, Conformité.
  - Adhérence MTP: combinaison pondérée Fit/MTP/Documents, robuste à l’absence de MTP.
  - Conformité documentaire: vérification déterministe depuis la base des documents (voir § Méthodes), checklist + score.
  - Synthèse IA: générée à la demande, explicitement informée des pièces présentes/absentes, avec règles strictes anti‑hallucination.

- **RAG local (option)**
  - Index SQLite par application (`rag_index`), embeddings OpenAI si clé disponible ou fallback hashing vectoriel local, recherche cosine.

### 3) Méthodes utilisées

- **Consolidation résiliente**
  - Gestion de schémas alternatifs (noms de tables/colonnes), fallback propre (messages d’info au lieu d’erreurs bloquantes).
  - Cache `@st.cache_data` pour limiter les appels répétés et accélérer l’UI.

- **Conformité documentaire fiabilisée (DB‑first)**
  - Filtrage des documents par `application_id` directement depuis `documents`.
  - Détection du type via colonnes sémantiques si elles existent: `type`, `doc_type`, `category`, `kind`, `label`, `document_type`, `name`.
  - Fallback sur colonnes de liens/fichiers: `link`, `url`, `public_url`, `download_url`, `path`, `storage_path`, `file_path`, `filename`, `name`.
  - Normalisation des pièces attendues: `cv`, `diploma`, `lm` (lettre de motivation). Les anciens `details.flags` et heuristiques par liens restent comme secours.

- **Adhérence MTP (non persistée)**
  - Extraction de `details.mtp.scores.overall` si présent.
  - Calcul: Fit (incl. vision) + MTP + Documents avec pondérations adaptatives en cas d’absence d’un composant.

- **Synthèse IA cadrée**
  - Prompt système explicitant l’interdiction de citer des documents absents et d’inventer du contenu.
  - Injection de `documents_present = {cv, diploma, lm}` dans le prompt utilisateur; message spécifique si aucune pièce n’est disponible.
  - Température faible et structure attendue (5–8 puces, verdict, décomposition pondérée, forces, risques, prochaine étape).

- **Extraction de texte et OCR**
  - PDF: `pdfminer.six`, fallback OCR via `PyMuPDF` + `pytesseract` si peu de texte.
  - Images: `Pillow` + `pytesseract`.
  - DOCX: `python-docx`.
  - TXT/générique: décodage robuste multi‑encodages.
  - Résumés hiérarchiques non‑IA pour compacter localement.

- **RAG local (option)**
  - Schéma: `ensure_sqlite_rag_schema()`; indexation par `upsert_rag_for_application()`; requêtes via `rag_search()`.
  - Embeddings OpenAI (si `OPENAI_API_KEY`) sinon fallback hashing vectoriel déterministe; similarité cosinus.

### 4) Outils utilisés

- **Frontend/serveur**: Streamlit.
- **Data**: pandas, numpy.
- **Charts**: Plotly (express + graph_objects).
- **Stockage**: Supabase (URL/Key depuis env), SQLite local (`scores.db`) pour scores/RAG.
- **Docs/OCR**: pdfminer.six, PyMuPDF (fitz), pytesseract, Pillow, python‑docx, requests.
- **IA (option)**: OpenAI (chat + embeddings) — désactivé si `OPENAI_API_KEY` absent.
- **Config**: python‑dotenv pour `.env`.

---

## Résultat attendu (hyper détaillé)

- **Vue globale pilotable**: KPIs, tendances et distributions pour orienter les actions RH.
- **Fiche candidat actionnable**: 5 jauges sur une ligne, commentaires courts, conformité fiable par la DB, synthèse IA exploitable sans hallucination.
- **Conformité documentaire exacte**: la présence/absence de CV, Diplôme, Lettre provient des enregistrements réels de la table documents (plus d’estimation hasardeuse).
- **Robustesse aux schémas**: l’app gère les variations de champs et de tables les plus courantes sans planter.
- **Performances**: caches, extraction/ocr ciblés, graphes interactifs légers.
- **Sécurité**: clés via environnement; pas d’upload de documents vers des APIs externes par défaut.

---

## Installation

1) Créer un environnement virtuel
```bash
python -m venv .venv
source .venv/bin/activate
```

2) Installer les dépendances
```bash
pip install -r requirements.txt
```

3) Dépendances systèmes (si OCR souhaité)
- Installer Tesseract (Linux): `sudo apt-get install tesseract-ocr`
- Facultatif: `tesseract-ocr-fra` pour le français

---

## Configuration

Variables d’environnement (via `.env`, voir `seeg_core/config.py`):

Obligatoires:
- `SUPABASE_URL`
- `SUPABASE_KEY`

Optionnelles:
- `OPENAI_API_KEY` (active la synthèse IA et les embeddings OpenAI)
- `OPENAI_MODEL` (par défaut `gpt-4o-mini`)
- `VISION_TEXT` (texte Vision SEEG inline) ou `VISION_FILE` (chemin vers un fichier)

Exemple `.env`:
```env
SUPABASE_URL=...
SUPABASE_KEY=...
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4o-mini
VISION_FILE=./scripts/vision_seeg.md
```

---

## Exécution

Lancer le dashboard:
```bash
streamlit run streamlit_app/app.py
```

Accès: http://localhost:8501

---

## Déploiement Docker

- __Prérequis__
  - Docker 20+ et Compose v2

- **Build de l'image**
  ```bash
  docker build -t seeg-dashboard:latest .
  ```

- **Run (sans compose)**
  ```bash
  docker run --rm \
    -p 8501:8501 \
    --env-file .env \
    -v "$(pwd)/scores.db:/app/scores.db" \
    seeg-dashboard:latest
  ```
  - L'application sera accessible sur http://localhost:8501
  - Le fichier local `scores.db` est monté dans le conteneur pour persister les scores/RAG.

- **docker-compose**
  ```bash
  docker compose up -d --build
  docker compose logs -f
  ```
  - Configuration dans `docker-compose.yml` (port 8501, `env_file: .env`, volume `scores.db`).

- **Variables d'environnement**
  - Utilisez `.env` (mêmes clés que la section Configuration). Exemple: `SUPABASE_URL`, `SUPABASE_KEY`, `OPENAI_API_KEY`, `VISION_FILE`.
  - Par défaut, l'image pointe `VISION_FILE` vers `/app/scripts/vision_seeg.md` (surchargable via `.env`).

- **Taille d'image et dépendances**
  - L'image installe `tesseract-ocr` (+ `tesseract-ocr-fra`) et le modèle spaCy `fr_core_news_md`.
  - Les documents ne sont pas envoyés à des APIs externes par défaut.

---

## Structure du projet

- `streamlit_app/app.py` — application Streamlit (pages Accueil/Candidat, jauges, conformité DB, synthèse IA, OCR, RAG optionnel)
- `seeg_core/config.py` — chargement des variables d’environnement (Supabase, Vision, OpenAI)
- `seeg_core/db.py` — connexions Supabase/SQLite, schémas SQLite (scores, RAG)
- `seeg_core/features.py` — fonctions utilitaires/features (si présent)
- `seeg_core/mtp.py` — logique MTP (si présent)
- `scripts/score_batch.py` — batch de scoring/upsert (option)
- `scripts/mtp_questions.json` — référentiel questions MTP
- `scripts/vision_seeg.md` — texte Vision SEEG
- `scores.db` — base SQLite locale (optionnelle)

---

## Spécifications techniques

- **Modèles NLP locaux**
  - spaCy `fr_core_news_md` — tokenisation FR et vecteurs (références: `seeg_core/nlp.py` `get_nlp()`, `text_vector()`, `cosine_sim()`)

- **Synthèse locale (sans IA)**
  - Résumés: `_summarize_text_simple()` et `_summarize_hierarchical()` (découpage par caractères, hiérarchique). Fichier: `streamlit_app/app.py`.

- **OCR / Extraction de texte**
  - PDF natif: `pdfminer.six` (`pdf_extract_text`)
  - PDF scanné: rendu PyMuPDF (`fitz`) + OCR `pytesseract` via `_ocr_pdf_with_fitz()`
  - Images: `Pillow` + `pytesseract` via `_ocr_image_bytes()`
  - DOC/DOCX: `python-docx` via `_extract_text_from_docx()`
  - TXT/générique: `_extract_text_from_txt()` (décodage robuste)

- **Embeddings & Similarité**
  - OpenAI (optionnel): `_embeddings_openai()` — modèle défaut `text-embedding-3-small`
  - Fallback local (sans réseau): `_embeddings_fallback(dim=256)` — hashing vectoriel déterministe basé sur `str.lower().split()`
  - Similarité: cosinus (`_cosine()`), vecteurs `float32`

- **RAG (index sémantique local SQLite)**
  - Schéma table: `rag_index(application_id TEXT, chunk_id INTEGER, text TEXT, embedding BLOB)` via `ensure_sqlite_rag_schema()`
  - Indexation: `upsert_rag_for_application()`
    - Découpage: `_chunk_text(max_chars=1200, overlap=200)`
    - Embeddings: OpenAI si `OPENAI_API_KEY` sinon fallback hashing `dim=256`
    - Stockage: `np.ndarray(dtype=float32).tobytes()`
  - Recherche: `rag_search(query, top_k=3)` — embedding de requête via OpenAI ou fallback, score cosinus, tri décroissant

- **Tokenisation**
  - spaCy: interne au modèle `fr_core_news_md` (utilisé pour `text_vector()`)
  - Fallback embeddings: séparation par espaces (`split`) en minuscules; pas de stopwords/stemming
  - Chunking textes: `_chunk_text(max_chars=1500, overlap=200)` pour résumés; 1200/200 pour RAG
  - OpenAI: tokenisation gérée côté API (non exposée dans le code)

- **Paramètres par défaut**
  - Résumés hiérarchiques: `chunk_size=1500`, `overlap=200`, `max_levels=3`
  - RAG: `chunk_size=1200`, `overlap=200`, `dim=256`, `top_k=3`
  - OCR PDF: `dpi=200`, `max_pages=20`, `lang=eng+fra`
  - Embeddings OpenAI: `text-embedding-3-small` (configurable)
  - Modèle chat (si IA activée): `OPENAI_MODEL=gpt-4o-mini`

- **Sécurité / confidentialité**
  - Extraction, OCR et résumés effectués localement; aucun envoi de documents aux APIs par défaut
  - Embeddings OpenAI uniquement si `OPENAI_API_KEY` est fourni

### Équations de scoring

- __Complétude (`compute_completeness`)__ — `seeg_core/scoring.py`
  - Flags attendus: `cv`, `lm`, `diploma`, `id`, `mtp`
  - Pondérations par défaut: `{cv: 0.30, lm: 0.20, diploma: 0.20, id: 0.10, mtp: 0.20}`
  - Formule (score 0–100):
    ```text
    completeness = 100 * (∑ w_k * 1{flag_k présent}) / (∑ w_k)
    ```

- __Adéquation/fit (`compute_fit`)__ — `seeg_core/scoring.py`
  - Vecteurs spaCy FR et cosinus entre `candidate_text`, `job_text`, `vision_text`
  - Poids: `alpha_job = 0.6`, `alpha_vision = 0.4`
  - Formule (score 0–100):
    ```text
    fit = 100 * clamp01(alpha_job * cos(cand, job) + alpha_vision * cos(cand, vision))
    ```

- __Score final batch (`scripts/score_batch.py`)__
  - Si score MTP disponible (`mtp_sub` de 0 à 100):
    ```text
    final = round(0.25 * completeness + 0.60 * fit + 0.15 * mtp_sub)
    ```
  - Sinon:
    ```text
    final = round(0.25 * completeness + 0.75 * fit)
    ```
  - Note: `seeg_core/scoring.py` expose aussi `compute_final(c, f, w_c=0.4, w_f=0.6)` (non utilisé dans le batch par défaut).

- __MTP (dimensionnel et global) (`seeg_core/mtp.py`)__
  - Par dimension `d ∈ {metier, talent, paradigme}` avec liste d’indicateurs `I_d`:
    ```text
    score_d = 100 * (# indicateurs trouvés dans la réponse / |I_d|), arrondi à l’entier
    ```
  - Détection: normalisation FR + correspondance sous-chaîne pour multi-mots, sinon intersection de mots (≥3 lettres)
  - Global:
    ```text
    overall = round( moyenne(score_d disponibles) )
    ```

- **Conformité documentaire (UI)** — `streamlit_app/app.py`
  - Pièces requises: `required = [cv, diploma, lm]`
  - Formule (0–100):
    ```text
    compliance = 100 * ( #pièces présentes / #pièces requises )
    ```

- __Seuils de recommandation (`recommend`)__ — `seeg_core/scoring.py`
  - `final ≥ 80` → « Fortement recommandé »
  - `60 ≤ final < 80` → « À considérer »
  - `final < 60` → « Non recommandé »

---

## Dépannage

- Avertissement `PeriodArray... timezone`: informatif (agrégation mensuelle), sans impact.
- Pas de documents détectés: vérifier `application_documents` et la présence des colonnes de type ou de lien (voir listes ci‑dessus).
- OCR manquant: installer `tesseract` et le paquet langue Fr si nécessaire.
- Synthèse IA inactive: définir `OPENAI_API_KEY` dans l’environnement.

---

## Évolutions possibles

- Colonnes de type dédiées (`doc_type`) côté DB pour supprimer les heuristiques.
- Règles de conformité par poste ou par pipeline.
- Export PDF de la fiche candidat et de la synthèse.
- Recherche sémantique (RAG) exposée dans l’UI.
