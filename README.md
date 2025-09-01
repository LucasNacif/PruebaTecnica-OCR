Clasificador de Legibilidad de DNIs con OCR

-Este script permite analizar imágenes del frente de un DNI utilizando PaddleOCR y determinar si son legibles según criterios predefinidos. Además, extrae datos clave como DNI, nombre, apellido y fecha de vencimiento, generando un archivo JSON con los resultados.

🖼️ Importante

-Colocar imágenes del frente del DNI en la carpeta DNIs.

-Formatos aceptados: .jpg, .jpeg, .png, .bmp, .tiff, .tif.

▶️ Ejecución

-Instalar dependencias:

    pip install -r requirements.txt


-Ejecutar el script:

    python script.py