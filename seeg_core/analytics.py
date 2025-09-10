from __future__ import annotations
import numpy as np
import pandas as pd


def derive_gender_series(users: pd.DataFrame, profiles: pd.DataFrame) -> pd.Series:
    """Retourne une Series libellée Homme/Femme/Inconnu.
    Préférence: candidate_profiles.gender si dispo, sinon users[gender|sexe|genre].
    """
    if profiles is not None and not profiles.empty and "gender" in profiles.columns and "user_id" in profiles.columns:
        s = profiles["gender"].fillna("Inconnu").astype(str)
        s = s.str.lower().map({
            "m": "Homme", "male": "Homme", "homme": "Homme",
            "f": "Femme", "female": "Femme", "femme": "Femme",
        }).fillna("Inconnu")
        return s
    if users is not None and not users.empty:
        for c in ["gender", "sexe", "genre"]:
            if c in users.columns:
                s = users[c].fillna("Inconnu").astype(str)
                s = s.str.lower().map({
                    "m": "Homme", "male": "Homme", "homme": "Homme",
                    "f": "Femme", "female": "Femme", "femme": "Femme",
                }).fillna("Inconnu")
                return s
        return pd.Series(["Inconnu"] * len(users))
    return pd.Series([], dtype=str)


def derive_age_series(users: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    """DataFrame avec colonnes: user_id, age, gender_label.
    - cherche une colonne date de naissance candidate
    - calcule l'âge en années glissantes
    - mappe le genre via derive_gender_series
    """
    df_users = users.copy() if users is not None and not users.empty else pd.DataFrame()
    if not df_users.empty and "id" in df_users.columns and "user_id" not in df_users.columns:
        df_users = df_users.rename(columns={"id": "user_id"})

    # base d'identifiants
    if not df_users.empty and "user_id" in df_users.columns:
        base = df_users[["user_id"]].drop_duplicates().copy()
    elif profiles is not None and not profiles.empty and "user_id" in profiles.columns:
        base = profiles[["user_id"]].drop_duplicates().copy()
    else:
        return pd.DataFrame(columns=["user_id", "age", "gender_label"])  # rien à joindre

    # cherche colonne DOB
    date_cols = ["birth_date", "date_naissance", "birthdate", "dob", "date_of_birth"]
    dob = None
    if profiles is not None and not profiles.empty:
        for c in date_cols:
            if c in profiles.columns:
                dob = profiles[["user_id", c]].rename(columns={c: "dob"})
                break
    if dob is None and not df_users.empty:
        for c in date_cols:
            if c in df_users.columns:
                src = df_users[["user_id", c]] if "user_id" in df_users.columns else pd.DataFrame()
                if not src.empty:
                    dob = src.rename(columns={c: "dob"})
                    break
    if dob is None:
        base["age"] = np.nan
    else:
        dob["dob"] = pd.to_datetime(dob["dob"], errors="coerce")
        today = pd.Timestamp.today().normalize()
        dob["age"] = ((today - dob["dob"]) / pd.Timedelta(days=365.25)).astype("float").apply(
            lambda x: np.floor(x) if pd.notna(x) else np.nan
        )
        base = base.merge(dob[["user_id", "age"]], on="user_id", how="left")

    g = derive_gender_series(df_users, profiles)
    gender_map = pd.DataFrame({"user_id": base["user_id"].values})
    if profiles is not None and not profiles.empty and "gender" in profiles.columns and "user_id" in profiles.columns:
        gg = profiles[["user_id", "gender"]].copy()
        gg["gender"] = gg["gender"].fillna("Inconnu").astype(str).str.lower().map({
            "m": "Homme", "male": "Homme", "homme": "Homme",
            "f": "Femme", "female": "Femme", "femme": "Femme",
        }).fillna("Inconnu")
        gender_map = gender_map.merge(gg, on="user_id", how="left").rename(columns={"gender": "gender_label"})
    elif not df_users.empty and "gender" in df_users.columns:
        gg = df_users[["user_id", "gender"]].copy()
        gg["gender"] = gg["gender"].fillna("Inconnu").astype(str).str.lower().map({
            "m": "Homme", "male": "Homme", "homme": "Homme",
            "f": "Femme", "female": "Femme", "femme": "Femme",
        }).fillna("Inconnu")
        gender_map = gender_map.merge(gg, on="user_id", how="left").rename(columns={"gender": "gender_label"})
    else:
        gender_map["gender_label"] = "Inconnu"

    return base.merge(gender_map, on="user_id", how="left")
