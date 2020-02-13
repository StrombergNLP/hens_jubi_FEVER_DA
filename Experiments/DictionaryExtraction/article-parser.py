import xml.etree.ElementTree as ET
import pandas as pd
import re

tree = ET.parse('data/dawiki-test-file.xml')
root = tree.getroot()

# titles,abstracts, linked_entities
df = pd.DataFrame()

for index, mother in enumerate(root):
    # print('Step {}'.format(index+1), end='\r')

    if mother.tag == 'page':

        for child in mother:
            title = ''
            abstract = ''
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

            df = df.append({'Title': title, 'Abstract': '', 'Linked Entities': linked_entities}, ignore_index=True)

print(df)
