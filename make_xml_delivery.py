from lxml import etree
import sys

#Usage 
# python3 make_delivery.py it 
# first argument is language

#Open translated document
target = open('translation/test.'+sys.argv[1],'r').read().splitlines()
#Remove empty lines
target = [line for line in target if line]
#Parse original file format
parser = etree.XMLParser()
tree  = etree.parse('wp.test.ca.xml',parser)
root = tree.getroot()

#Replace each sentence with the translated sentences
index = 0
for paragraph in root.iter('p'):
    # for each paragraph, split its sentences and replace them by the target ones
    paragraph_text = ['']
    for sentence in paragraph.text.strip().split('\n'):
        paragraph_text.append(target[index])
        index += 1
    paragraph_text.append('')
    paragraph.text = '\n'.join(paragraph_text)

#Replace attribtutes
for doc in root.iter('doc'):
    doc.attrib['lang'] = sys.argv[1] #change language
    doc.attrib['date'] = '06072021' #change date
    doc.attrib['sysid'] = 'BSCPrimary' #specify new attribute for system id

#Write file
tree.write('BSCPrimary.romance.ca2'+sys.argv[1]+'.xml',encoding="UTF-8")
