import streamlit as st
from fastai.vision.all import *
from PIL import Image

# Re-déclaration de la fonction utilisée lors de l'entraînement
def label_function(x):
    """Retourne le nom de la classe à partir du nom complet du fichier."""
    label = x.parent.name.lower()  # Retourne le nom du dossier parent comme label
    return label

# Fonction pour charger le modèle
def load_model(model_path):
    """Charge un modèle sauvegardé."""
    try:
        learn = load_learner(model_path)
        return learn
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {e}")
        return None

# Fonction principale de l'application Streamlit
def main():
    st.title("Application de Diagnostic des Cacaoyers")
    st.subheader("Prédisez la santé des cabosses de cacao à partir d'une image")

    # Chemin du modèle exporté
    model_path = r'D:\Master2\DeepLearning\devoir1\data\train\mondele_r50.pkl'  # Chemin complet vers le fichier modèle

    # Charger le modèle
    model = load_model(model_path)
    
    if model:
        st.success(f"Modèle chargé avec succès depuis : {model_path}")

        # Interface pour télécharger une image
        st.subheader("Téléchargez une image pour faire une prédiction")
        uploaded_image = st.file_uploader("Choisir une image...", type=["jpg", "png", "jpeg"])

        if uploaded_image:
            img = Image.open(uploaded_image)  # Ouvre l'image téléchargée
            st.image(img, caption="Image téléchargée", use_column_width=True)

            # Effectuer la prédiction
            pred_class, pred_idx, outputs = model.predict(img)

            # Afficher la prédiction et un message descriptif
            st.write(f"Classe prédite : **{pred_class}**")

            # Messages pour chaque classe
            messages = {
                "Healthy cacao": "✅ Votre cacaoyer semble en bonne santé. Continuez les bonnes pratiques agricoles pour maintenir cette condition.",
                "Cacao black pod disease": "⚠️ Attention ! Il semble que votre cacaoyer soit atteint de la pourriture noire des cabosses. Veuillez envisager un traitement rapide pour éviter la propagation.",
                "witches_broom": "⚠️ Alerte : Votre cacaoyer semble présenter les symptômes de la maladie du balai de sorcière. Contactez un spécialiste pour des recommandations de traitement."
            }

            # Afficher le message correspondant
            if pred_class in messages:
                st.info(messages[pred_class])
            else:
                st.warning("Classe non reconnue. Assurez-vous que l'image est bien conforme au contexte de classification.")

if __name__ == "__main__":
    main()
