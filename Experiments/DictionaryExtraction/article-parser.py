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
            linked_entities = []    # Since there can be more than one, how do we store them in the DF? In a list or...
            linked_entities = ''    # ... In a String seperated by a ; every time we add to it? 

            if child.tag == 'title':
                title = child.text
                title = title.replace(' ', '_')

            elif child.tag == 'revision':

                for grandchild in child:
                    if grandchild.tag == 'text':

                        rawtext = grandchild.text
                        regex = r"(^{{(.*)}}$)|(&lt;ref&gt;.*&lt;\/ref&gt;)|(''')|(^==.*)" 

                        
                        # cleantext = re.sub(pattern, '', rawtext)
                        print(cleantext)

                        # Remove the {} groups 

                        # Save the first line 

                        # Handle linked entities in that line

                        # Store them as linked_entities 

            # df = df.append({'Title': title, 'Abstract': '', 'Linked Entities': linked_entities}, lines=True)