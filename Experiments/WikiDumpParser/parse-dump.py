import xml.etree.ElementTree as ET
import pandas as pd

tree = ET.parse('data/dawiki-latest-abstract.xml')
root = tree.getroot()

titles = []
urls = []
abstracts = []

for index, doc in enumerate(root):
    print('Step {}'.format(index+1), end='\r')
    for child in doc:
        if child.tag == 'title':
            title = child.text
            title = title.replace('Wikipedia: ', '').replace(' ', '_')
            titles.append(title)
        elif child.tag == 'url':
            urls.append(child.text)
        elif child.tag == 'abstract':
            if child.text:
                abstract = child.text
                abstract = abstract.replace(u'\xa0', u' ')
                abstracts.append(abstract)
            else:
                abstracts.append('')
print('')


df = pd.DataFrame()
df['title'] = titles
df['abstract'] = abstracts
df['url'] = urls

print('Size of data before cleaning: {}'.format(len(df['title'])))

df = df.loc[[not x for x in df['abstract'].str.contains('\||\[|\}|\{|\<|----')]] # Filter out rows that contain pipes or square brackets
df = df.reset_index(drop=True)

print('Size of data after cleaning: {}'.format(len(df['title'])))

df.to_csv('out/dawiki-latest-abstract.csv')
print('SUCCESS')