from flask import Blueprint, render_template, Response, jsonify
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# Création d'un Blueprint pour la page d'analyse
analyze_bp = Blueprint('analyze', __name__)

# Initialisation du détecteur de mains et du classificateur
detector = HandDetector(maxHands=1)

# Choix du modèle de classification

# Pour toute l'alphabet
# classifier = Classifier("Model/alphabet/keras_model.h5", "Model/alphabet/labels.txt")
# labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "Q", "R", "S", "V", "W", "Y"]

# Pour les mots statiques
classifier = Classifier("Model/words/keras_model.h5", "Model/words/labels.txt")
labels = ["Bonjour", "Non", "Oui", "Tu vas bien"]

# Alphabet de A - K
# classifier = Classifier("Model/A-K/keras_model.h5", "Model/A-K/labels.txt")
# labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K"]

# Alphabet de L - Y
# classifier = Classifier("Model/L-Y/keras_model.h5", "Model/L-Y/labels.txt")
# labels = ["L", "M", "N", "O", "Q", "R", "S", "V", "W", "Y"]


# Paramètres pour l'affichage de l'image et le traitement des mains
offset = 20
imgSize = 300
current_label = "[Aucune main détectée]"  # Label par défaut lorsque aucune main n'est détectée

# Fonction générant les frames pour le flux vidéo avec la classification
def generate_frames_classification():
    global current_label
    cap = cv2.VideoCapture(1)  # Ouverture de la caméra (id=1)
    
    while True:
        success, img = cap.read()  # Capture d'une image de la caméra
        if not success:
            break  # Si l'image n'a pas été capturée, on quitte la boucle
        else:
            imgOutput = img.copy()  # Création d'une copie de l'image pour l'affichage
            hands, img = detector.findHands(img)  # Détection des mains dans l'image
            
            if hands:
                hand = hands[0]  # Prendre la première main détectée
                x, y, w, h = hand['bbox']  # Coordonnées de la boîte englobante de la main
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255  # Image blanche pour normaliser la taille de la main

                try:
                    # Recadrage de la main détectée dans l'image
                    imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]
                    aspectRatio = h / w  # Calcul du ratio de l'aspect de la main

                    # Ajustement de la taille de la main pour correspondre à la taille de l'image cible (imgSize)
                    if aspectRatio > 1:
                        k = imgSize / h
                        wCal = math.ceil(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, imgSize))  # Redimensionnement pour une hauteur uniforme
                        wGap = math.ceil((imgSize - wCal) / 2)
                        imgWhite[:, wGap:wCal + wGap] = imgResize  # Remplir l'image blanche avec l'image redimensionnée
                    else:
                        k = imgSize / w
                        hCal = math.ceil(k * h)
                        imgResize = cv2.resize(imgCrop, (imgSize, hCal))  # Redimensionnement pour une largeur uniforme
                        hGap = math.ceil((imgSize - hCal) / 2)
                        imgWhite[hGap:hCal + hGap, :] = imgResize  # Remplir l'image blanche avec l'image redimensionnée
                    
                    # Prédiction de la classe de la main avec le classificateur
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)
                    current_label = labels[index]  # Récupérer l'étiquette correspondante à la prédiction

                    # Dessiner un rectangle autour de la main et afficher l'étiquette
                    cv2.rectangle(
                        imgOutput,
                        (x - offset, y - offset - 50),
                        (x - offset + 90, y - offset - 50 + 50),
                        (13, 110, 253),
                        cv2.FILLED,
                    )
                    cv2.putText(
                        imgOutput,
                        labels[index],  # Affichage de l'étiquette prédite
                        (x, y - 26),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1.7,
                        (255, 255, 255),
                        2,
                    )
                    # Dessiner un rectangle autour de la main
                    cv2.rectangle(
                        imgOutput,
                        (x - offset, y - offset),
                        (x + w + offset, y + h + offset),
                        (13, 110, 253),
                        4,
                    )
                except Exception as e:
                    # Si une erreur se produit lors du traitement, l'afficher
                    print(f"Erreur lors du traitement de l'image : {e}")
            else:
                # Si aucune main n'est détectée, réinitialiser le label
                current_label = "[Aucune main détectée]"

            # Convertir l'image traitée en un flux JPEG
            ret, buffer = cv2.imencode('.jpg', imgOutput)
            img = buffer.tobytes()

            # Renvoie l'image sous forme de flux de données pour l'affichage
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')

# Route pour la page principale de l'analyse
@analyze_bp.route('/')
def index():
    return render_template('analyze/analyze_index.html')

# Route pour récupérer le flux vidéo avec l'image classifiée
@analyze_bp.route('/video_feed')
def video_feed():
    return Response(generate_frames_classification(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route pour obtenir l'étiquette actuelle de la main détectée
@analyze_bp.route('/current_label')
def get_current_label():
    global current_label
    return jsonify({'label': current_label})  # Retourner l'étiquette sous forme de réponse JSON
