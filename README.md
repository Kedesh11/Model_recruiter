# SEEG Recrutement — Scoring & Dashboard

## Objectif
- Centraliser les candidatures via Supabase
- Calculer les scores: complétude, adéquation, score final, recommandation
- Visualiser via un dashboard Streamlit

## Installation
1. Créer un environnement virtuel
```
python -m venv .venv
source .venv/bin/activate
```
2. Installer les dépendances
```
pip install -r requirements.txt
python -m spacy download fr_core_news_md
```
3. Configurer l'environnement
- Créer un fichier `.env` (déjà présent) avec `SUPABASE_URL` et `SUPABASE_KEY` (service role pour batch). Ne pas committer ce fichier.

## Structure
- `seeg_core/` modules métier (DB, NLP, MTP parsing, scoring)
- `scripts/score_batch.py` batch pour calculer/upsert les scores
- `streamlit_app/app.py` dashboard Streamlit

## Lancer
- Batch scoring:
```
python scripts/score_batch.py --limit 100
```
- Dashboard:
```
streamlit run streamlit_app/app.py
```

## Notes
- Le pipeline évite scikit-learn pour limiter les soucis de compatibilité au début. On utilise des similarités sémantiques via spaCy. On pourra basculer vers `sentence-transformers` plus tard si besoin.
