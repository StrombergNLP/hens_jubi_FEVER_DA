from datetime import datetime
import xml.etree.ElementTree as ET
import pandas as pd
import re

print('Loading file...')
tree = ET.parse('data/dawiki-latest-pages-articles-multistream.xml')
# tree = ET.parse('data/dawiki-latest-pages-articles-multistream-sample.xml')
root = tree.getroot()

print('Loading dataframe...')
df = pd.DataFrame()
counter = 1
faults = 0

for index, mother in enumerate(root):
    if mother.tag == 'page':
        print('Step {}'.format(counter), end='\r')

        for child in mother:
            linked_entities = []

            if child.tag == 'title':
                title = child.text
                if title.startswith('Hjælp:') or title.startswith('Wikipedia:'):
                    break
                title = title.replace(' ', '_')

            elif child.tag == 'revision':

                for grandchild in child:
                    if grandchild.tag == 'text':
                        # Handle entire article
                        rawtext = grandchild.text
                        regex = r"(^[\s:]*{{[^}]*}}$)|(&lt;.*?&gt;)|(^\s*==.*)|<\s*[^>]*>(.*?)<\s*/\s*\w+>" # |(''')
                        rawtext = re.sub(regex, '', rawtext, flags=re.MULTILINE|re.DOTALL).strip()
                        cleantext = rawtext.splitlines()
                        abstract = ''

                        for line in cleantext: 
                            if '=' or '}' not in line and "'''" in line: 
                                abstract = line
                                abstract = abstract.replace("'''", '')
                                break
                        
                        if '#redirect' in abstract.lower():
                            break

                        if abstract == '':
                            if not len(cleantext):
                                abstract = ''
                            elif cleantext[0] != '':
                                abstract = cleantext[0]
                            else: 
                                abstract = cleantext[1]


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

                        if abstract != '':
                            df = df.append({'Title': title, 'Abstract': abstract, 'Linked Entities': linked_entities}, ignore_index=True)
                        else: 
                            faults +=1

                        counter +=1

    if counter == 10001:
        break

print('')
print('{} faults of {} pages'.format(faults, counter))
print('Finishing parsing...')
print('Saving to file...')
df.to_json('out/{}.jsonl'.format(datetime.now().strftime("%d-%m-%Y-%H-%M-%S")), orient='records', lines=True)
print('Saved {} articles to file.'.format(len(df['Title'])))