import xml.etree.ElementTree as ET
import pandas as pd
import re

print('Loading file...')
tree = ET.parse('data/dawiki-latest-pages-articles-multistream.xml')
root = tree.getroot()

df = pd.DataFrame()
counter = 1

for index, mother in enumerate(root):
    if mother.tag == 'page':
        print('Step {}'.format(counter), end='\r')

        for child in mother:
            linked_entities = []    # ... In a String seperated by a ; every time we add to it? 

            if child.tag == 'title':
                title = child.text
                title = title.replace(' ', '_')

            elif child.tag == 'revision':

                for grandchild in child:
                    if grandchild.tag == 'text':
                        rawtext = grandchild.text
                        regex = r"(^\s*{{[\w\s|=\[\],<>/.\(\):-]*}}$)|(&lt;ref&gt;.*&lt;\/ref&gt;)|(''')|(^\s*==.*)|(<ref[\w\s=\"]*/>)"
                        cleantext = re.sub(regex, '', rawtext, flags=re.MULTILINE|re.DOTALL).strip()

                        # Save the first line 
                        abstract = cleantext.splitlines()[0]

                        # Handle linked entities in that line
                        regex = r"\[\[(.*?)\]\]"
                        links = re.findall(regex, abstract)

                        # Store them as linked_entities 
                        for l in links:
                            l = re.sub(r"(?<!\\)\|.*", '', l)
                            l = l.replace(' ', '_').strip()
                            linked_entities.append(l)

                        # Clean the abstract
                        for l in links:
                            clean_l = re.sub(r".*(?<!\\)\|", '', l)
                            abstract = abstract.replace('[['+l+']]', clean_l)

                        df = df.append({'Title': title, 'Abstract': abstract, 'Linked Entities': linked_entities}, ignore_index=True)
                        counter +=1
                        print(title)
                        print(abstract)
                        print(linked_entities)

    if counter == 6:
        break


df.to_json('out/testfile.jsonl', orient='records', lines=True)
