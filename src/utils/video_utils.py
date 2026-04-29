"""
video_utils.py — Utilitaires de gestion vidéo
==============================================
Fournit concat_episode_videos() pour assembler les vidéos épisodiques
produites par gym.wrappers.RecordVideo en une seule vidéo par run.
"""

import glob
import os


def concat_episode_videos(video_folder: str, output_name: str = "full_run.mp4",
                          delete_sources: bool = True) -> str | None:
    """
    Concatène tous les .mp4 de `video_folder` en un seul fichier.

    Args:
        video_folder:    Dossier contenant les fichiers .mp4 individuels.
        output_name:     Nom du fichier de sortie (dans le même dossier).
        delete_sources:  Si True, supprime les fichiers individuels après concaténation.

    Returns:
        Chemin du fichier concaténé, ou None si aucune vidéo trouvée.
    """
    try:
        from moviepy import VideoFileClip, concatenate_videoclips
    except ImportError:
        print("[WARN] moviepy non disponible — concaténation vidéo ignorée.")
        return None

    # Lister et trier les épisodes dans l'ordre naturel
    clips_paths = sorted(glob.glob(os.path.join(video_folder, "*.mp4")))
    out_path = os.path.join(video_folder, output_name)

    # Exclure la sortie si elle existe déjà (relance)
    clips_paths = [p for p in clips_paths if os.path.basename(p) != output_name]

    if not clips_paths:
        return None

    print(f"  → Concaténation de {len(clips_paths)} vidéos dans {out_path}...")
    try:
        clips = [VideoFileClip(p) for p in clips_paths]
        final = concatenate_videoclips(clips, method="compose")
        final.write_videofile(out_path, logger=None, codec="libx264", audio=False)

        # Fermer proprement les clips
        for c in clips:
            c.close()
        final.close()

        # Supprimer les épisodes individuels
        if delete_sources:
            for p in clips_paths:
                os.remove(p)

        print(f"  ✓ Vidéo finale : {out_path}")
        return out_path

    except Exception as e:
        print(f"  [WARN] Concaténation échouée : {e}")
        return None
