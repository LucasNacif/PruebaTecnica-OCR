import os
import cv2
import re
import json
import numpy as np
import unicodedata
from datetime import datetime
from paddleocr import PaddleOCR

class DNIClassifierPaddle:
    
    # Inicializa la clase, configura PaddleOCR en español y define extensiones válidas
    def __init__(self):
        print("Inicializando PaddleOCR...")        
        self.ocr = PaddleOCR(lang='es', use_textline_orientation=True)
        print("PaddleOCR listo!")
        self.valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    
    # Aplica preprocesamiento a la imagen
    def preprocess_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return []
        
        h, w = img.shape[:2]
        if w < 1000:
            scale = 1000 / w
            img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convertir a escala de grises
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) # Mejora de contraste
        enhanced = clahe.apply(gray) # Aplicar CLAHE
        denoised = cv2.fastNlMeansDenoising(enhanced) # Reducción de ruido
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) 
        sharpened = cv2.filter2D(denoised, -1, kernel) # Aplicar sharpen
        # Binarización adaptativa
        thresh = cv2.adaptiveThreshold(sharpened, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        return [denoised, sharpened, thresh]
    
    # Ejecuta OCR sobre la imagen original y variantes preprocesadas, devolviendo textos detectados con su score
    def extract_text(self, image_path):
        detections = []
        #Preprocesa la imagen
        processed_imgs = self.preprocess_image(image_path)
        all_imgs = [image_path] + processed_imgs
        
        # Ejecuta OCR en cada variante 
        for img in all_imgs:
            try:
                results = self.ocr.predict(img) # Ejecuta OCR (PaddleOCR.predict admite path o array)
                if not results or not isinstance(results, list):
                    continue

                for ocr_result in results:
                    try:
                        if 'rec_texts' in ocr_result and 'rec_scores' in ocr_result:
                            texts = ocr_result['rec_texts']
                            scores = ocr_result['rec_scores']

                            for i, (text, score) in enumerate(zip(texts, scores)):
                                if text and isinstance(text, str) and text.strip():
                                    detections.append((text.strip(), score))
                    except Exception:
                        continue
            except Exception:
                continue

        return detections
    
    # Normaliza un texto eliminando acentos y convirtiéndolo a mayúsculas
    def normalize_text(self, text):
        return unicodedata.normalize('NFKD', text).encode('ASCII','ignore').decode('ASCII').upper()
    
    # Clasifica la legibilidad de la imagen y extrae campos (DNI, nombre, apellido, vencimiento) según reglas
    def classify_legibility(self, detections, filename):
        data = {
            'archivo': filename,
            'nombre': None,
            'apellido': None,
            'dni': None,
            'fecha_vencimiento': None,
            'legible': False,
            'detecciones': [{'texto': t, 'score': s} for t, s in detections] 
        }

        # Normalizar textos
        texts_norm = [(self.normalize_text(text), text) for text, score in detections]

        # --- DNI ---
        dni_regex = r'\b[A-Z]?\d{7,8}\b|\b\d{2}\.\d{3}\.\d{3}\b'
        for norm, original in texts_norm:
            m = re.search(dni_regex, original)
            if m:
                data['dni'] = m.group()
                break

        # --- Nombre y Apellido ---
        nombre_labels = ['NOMBRE', 'NOMBRE 1', 'NAME']
        apellido_labels = ['APELLIDO', 'SURNAME']

        for i, (norm, original) in enumerate(texts_norm):
            if any(label in norm for label in apellido_labels):
                for j in range(i+1, min(i+4, len(texts_norm))):
                    candidate = texts_norm[j][1].strip()
                    if re.match(r'^[A-ZÁÉÍÓÚÑ\s\-]+$', self.normalize_text(candidate)):
                        data['apellido'] = candidate
                        break
            if any(label in norm for label in nombre_labels):
                for j in range(i+1, min(i+4, len(texts_norm))):
                    candidate = texts_norm[j][1].strip()
                    if re.match(r'^[A-ZÁÉÍÓÚÑ\s\-]+$', self.normalize_text(candidate)):
                        data['nombre'] = candidate
                        break

        # --- Fecha de vencimiento ---
        fecha_labels = ['VENCIMIENTO', 'DATE OF EXPIRY', 'DATE OF EXPIRATION']
        fecha_regex = r'\b\d{1,2}[/\-\s]?(?:ENE|FEB|MAR|ABR|MAY|JUN|JUL|AGO|SEP|OCT|NOV|DIC|\d{2})[/\-\s]?\d{2,4}\b'
        for i, (norm, original) in enumerate(texts_norm):
            if any(label in norm for label in fecha_labels):
                for j in range(i+1, min(i+5, len(texts_norm))):
                    fecha_cand = self.normalize_text(texts_norm[j][1])
                    if re.search(fecha_regex, fecha_cand):
                        data['fecha_vencimiento'] = texts_norm[j][1].strip()
                        break
                if data['fecha_vencimiento']:
                    break

        # Legibilidad: DNI + al menos 2 de los 3 campos restantes
        otros_campos = [data['nombre'], data['apellido'], data['fecha_vencimiento']]
        campos_presentes = sum(1 for c in otros_campos if c)
        if data['dni'] and campos_presentes >= 2:
            data['legible'] = True

        return data

    # Procesa todas las imágenes de una carpeta, aplicando OCR, extracción y clasificación, devolviendo resultados
    def process_folder(self, folder_path="DNIs"):
        if not os.path.exists(folder_path):
            print(f"❌ La carpeta '{folder_path}' no existe")
            return []
        
        # Lista de archivos de imagen en la carpeta
        image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                       if f.lower().endswith(self.valid_extensions)]
        
        if not image_files:
            print(f"❌ No se encontraron imágenes en la carpeta '{folder_path}'")
            return []
            
        results = []
        #Bucle sobre cada imagen
        for img_path in sorted(image_files):
            filename = os.path.basename(img_path)
            print(f"\nProcesando: {filename}")
            #Extrae con OCR
            detections = self.extract_text(img_path)
            #Clasifica legibilidad y extrae campos
            data = self.classify_legibility(detections, filename)
            results.append(data)

            # Consola
            status = "✅ LEGIBLE" if data['legible'] else "❌ ILEGIBLE"
            print(f"  Estado: {status}")
            print(f"  DNI: {data['dni'] or 'No detectado'}")
            print(f"  Nombre: {data['nombre'] or 'No detectado'}")
            print(f"  Apellido: {data['apellido'] or 'No detectado'}")
            print(f"  Fecha de vencimiento: {data['fecha_vencimiento'] or 'No detectado'}")

        return results
    
    # Guarda los resultados en un archivo JSON con timestamp
    def save_results(self, results):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"clasificacion_dni_{ts}.json"
        with open(fname, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResultados guardados en {fname}")


def main():
    classifier = DNIClassifierPaddle()
    results = classifier.process_folder("DNIs")
    classifier.save_results(results)

    # Resumen final
    total = len(results)
    legibles = sum(1 for r in results if r['legible'])
    no_legibles = total - legibles

    print("\n--- RESUMEN FINAL ---")
    print(f"Total de imágenes analizadas: {total}")
    print(f"✅ Legibles: {legibles}")
    print(f"❌ Ilegibles: {no_legibles}")

if __name__ == "__main__":
    main()
