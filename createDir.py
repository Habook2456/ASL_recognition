import os

os.makedirs("Data")
# Obtener la letra inicial y final del alfabeto inglés
letra_inicial = ord('D')
letra_final = ord('Z')

# Crear las carpetas
for letra_ascii in range(letra_inicial, letra_final + 1):
    letra = chr(letra_ascii)

    os.makedirs(f"Data/{letra}")
    print(f"Se ha creado la carpeta: {letra}")
