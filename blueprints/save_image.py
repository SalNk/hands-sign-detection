from flask import Blueprint, render_template, Response, jsonify, request
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

# Création d'un Blueprint pour la sauvegarde d'images
save_image_bp = Blueprint('save_image', __name__)

# Initialisation du détecteur de mains avec un maximum d'une main détectée
detector = HandDetector(maxHands=1)

# Paramètres utilisés pour le traitement de l'image
offset = 20        # Marge autour de la main pour le recadrage
imgSize = 300      # Taille de l'image recadrée (300x300 pixels)
counter = 0        # Compteur pour les images sauvegardées
imgWhite = None    # Variable globale pour stocker l'image à sauvegarder

@save_image_bp.route('/')
def index():
    """
    Route principale pour la page de sauvegarde d'images.
    Affiche le template 'save_image_index.html'.
    """
    return render_template('save_image/save_image_index.html')

def generate_frames():
    """
    Génère un flux de frames vidéo avec détection des mains.
    Si une main est détectée, l'image est recadrée et stockée dans 'imgWhite'.
    """
    global imgWhite
    cap = cv2.VideoCapture(1)  # Utilise la caméra avec l'ID 1
    while True:
        success, img = cap.read()
        if not success:
            break  # Arrête la boucle si la lecture de l'image échoue
        else:
            # Détection des mains dans l'image
            hands, img = detector.findHands(img)
            if hands:
                hand = hands[0]  # Prend la première main détectée
                x, y, w, h = hand['bbox']  # Coordonnées de la boîte englobante de la main
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255  # Crée une image blanche

                try:
                    # Recadrage de l'image autour de la main avec l'offset
                    imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]
                    aspectRatio = h / w  # Calcul du ratio hauteur/largeur

                    if aspectRatio > 1:
                        # Si la main est plus haute que large
                        k = imgSize / h
                        wCal = math.ceil(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, imgSize))  # Redimensionne l'image recadrée
                        wGap = math.ceil((imgSize - wCal) / 2)
                        imgWhite[:, wGap:wCal + wGap] = imgResize  # Centre l'image redimensionnée sur l'image blanche
                    else:
                        # Si la main est plus large que haute
                        k = imgSize / w
                        hCal = math.ceil(k * h)
                        imgResize = cv2.resize(imgCrop, (imgSize, hCal))  # Redimensionne l'image recadrée
                        hGap = math.ceil((imgSize - hCal) / 2)
                        imgWhite[hGap:hCal + hGap, :] = imgResize  # Centre l'image redimensionnée sur l'image blanche

                    # Optionnel : Affiche l'image traitée dans une fenêtre séparée
                    cv2.imshow("ImageWhite", imgWhite)

                except Exception as e:
                    # Affiche une erreur si le recadrage échoue
                    print(f"Erreur lors du recadrage : {e}")

            # Convertit l'image en format JPEG pour le flux vidéo
            ret, buffer = cv2.imencode('.jpg', img)
            img = buffer.tobytes()

            # Renvoie l'image encodée sous forme de flux
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')

@save_image_bp.route('/video_feed')
def video_feed():
    """
    Route pour le flux vidéo en direct.
    Renvoie le flux généré par la fonction 'generate_frames'.
    """
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@save_image_bp.route('/save_image', methods=['POST'])
def save_image():
    """
    Route pour sauvegarder l'image traitée dans un dossier spécifié par l'utilisateur.
    Si aucun dossier n'est spécifié, utilise le dossier par défaut 'test'.
    """
    global counter, imgWhite
    data = request.get_json()  # Récupère les données JSON envoyées par le client
    folder_name = data.get('folder', 'test')  # Dossier choisi par l'utilisateur, par défaut "test"

    # Définition du chemin complet du dossier de sauvegarde
    save_folder = f"static/images/{folder_name}"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)  # Crée le dossier s'il n'existe pas

    if imgWhite is not None:
        counter += 1  # Incrémente le compteur
        filename = f'{save_folder}/Image_{time.time()}.jpg'  # Nom unique pour l'image
        cv2.imwrite(filename, imgWhite)  # Sauvegarde l'image dans le dossier spécifié
        return jsonify({'message': 'Image sauvegardée avec succès', 'filename': filename})
    
    # Renvoie une erreur si aucune image n'est disponible à sauvegarder
    return jsonify({'message': 'Aucune image disponible à sauvegarder'}), 400
