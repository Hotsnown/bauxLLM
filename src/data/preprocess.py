import re

paragraphes = {}

with open("./data/lease.xml", "r", encoding="utf-8") as lease_file:
    contenu = lease_file.read()  # Lit tout le contenu du fichier
    
    for match in re.finditer(r'<(\w+)>(.*?)</\1>', contenu, re.DOTALL):
        tag, paragraphe = match.groups()
        
        paragraphe = paragraphe.strip()
        
        if tag in paragraphes:
            paragraphes[tag].append(paragraphe)
        else:
            paragraphes[tag] = [paragraphe]

for header, text in paragraphes.items():

    with open(f"./data/sections/{header}"+".txt", "w", encoding="utf-8") as out_file:
        text = text[0]
        out_file.write(text)