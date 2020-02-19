from datetime import datetime
import xml.etree.ElementTree as ET
import pandas as pd
import re


print('Loading file...')
tree = ET.parse('data/dawiki-latest-pages-articles-multistream.xml')
# tree = ET.parse('data/dawiki-test-file.xml')
root = tree.getroot()

df = pd.DataFrame()
counter = 1
faults = 0
title_fault = 0
redirect = 0

start_time = datetime.now()

for index, mother in enumerate(root):
    if mother.tag == 'page':
        print('Step {}'.format(counter), end='\r')

        title_tag = mother.find('title')

        linked_entities = []

        title = title_tag.text
        title = title.replace(' ', '_')

        if 'Hjælp:' in title or 'Wikipedia:' in title: 
            title_fault += 1
            continue

        revision_tag = mother.find('revision')

        text_tag = revision_tag.find('text')
        # Handle entire article
        rawtext = text_tag.text
        rawtext = re.sub(r"(\[\[File?:.*)|({{.*}})", '',rawtext)

        regex = r"(^\|.*)|({{[^}]*}*)|(&lt;.*?&/?gt;)|('''?)|(^:.*)"
        cleantext = re.sub(regex, '', rawtext).strip()
        regex = r"(^\s*==.*)|<\s*[^>]*>(.*?)<\s*/\s*\w+>"
        cleantext = re.sub(regex, '', cleantext, flags=re.MULTILINE|re.DOTALL).strip()

        if '#redirect' in cleantext.lower():
            redirect += 1
            continue
        
        abstract = ''
        cleantext = cleantext.splitlines()
        for line in cleantext: 
            if len(line) > 10:
                abstract = line
                break
        
        if abstract == '':
            faults += 1
            continue

        # Handle linked entities in that line
        regex = r"\[\[(.*?)\]\]"
        links = re.findall(regex, abstract)

        # Store them as linked_entities 
        for l in links:
            l = re.sub(r"(?<!\\)\|.*", '', l)
            l = l.replace(' ', '_').strip()
            linked_entities.append(l)
            # Clean the abstract
            clean_l = re.sub(r".*(?<!\\)\|", '', l)
            abstract = abstract.replace('[['+l+']]', clean_l)

        df = df.append({'Title': title, 'Abstract': abstract, 'Linked Entities': linked_entities}, ignore_index=True)
            
        counter +=1

    if counter == 10001:
        break 

print('')
print("{} 'hjælp' or 'wikipedia' titles".format(title_fault))
print('{} redirect pages'.format(redirect))
print('{} empty abstracts of {}'.format(faults, counter))
df.to_json('out/{}.jsonl'.format(datetime.now().strftime("%d-%m-%Y-%H-%M-%S")), orient='records', lines=True)
print('Saved {} articles to file.'.format(len(df['Title'])))

print('Parsing took {}'.format(datetime.now()-start_time))