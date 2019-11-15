from spacy import displacy
from pathlib import Path

from corpus import Corpus, Sentence
from config import Config

import numpy as np

visual_method = 'unk'
random_number = 100

config = Config('../../config.ini')
gold = Corpus.load(config.fdata)
origin = Corpus.load("{}/{}".format(config.result_path, 'raw_result.conllx'))
attacked = Corpus.load("{}/black_substitute_{}_0.1.conllx".format(config.result_path, visual_method))
attacked_index = []
for index, (origin_sen, attack_sen) in enumerate(zip(origin.sentences,attacked.sentences)):
    origin_sentence = [word.lower() for word in origin_sen.FORM]
    attack_sentence = [word.lower() for word in attack_sen.FORM]
    if origin_sentence != attack_sentence:
        attacked_index.append(index)
print(len(attacked_index))

random_index = np.random.choice(attacked_index, random_number, replace=False)
def to_svg(sent: Sentence):
    doc = {
        "words": [],
        "arcs": []
    }
    for i in range(len(sent.ID)):
        doc['words'].append({
            'text': sent.FORM[i],
            'tag': sent.POS[i]
        })
        if sent.HEAD[i] != '0':
            if int(sent.ID[i]) < int(sent.HEAD[i]):
                doc['arcs'].append({
                    'start': int(sent.ID[i]) - 1,
                    'end': int(sent.HEAD[i]) - 1,
                    'label': sent.DEPREL[i],
                    'dir': 'left'
                })
            else:
                doc['arcs'].append({
                    'start': int(sent.HEAD[i]) - 1,
                    'end': int(sent.ID[i]) - 1,
                    'label': sent.DEPREL[i],
                    'dir': 'right'
                })
    ret = displacy.render(doc, style='dep', options={'compact': True, 'distance': 80}, jupyter=False, manual=True)
    return ret


x = Path("{}/{}/{}_bert.html".format(config.workspace,"visualize",visual_method)).open('w', encoding='utf8')
x.write('<html><body>')

for i in random_index:
    x.write('SENTENCE {}'.format(i))
    x.write('<br>')

    x.write('<b>ORIGIN: </b>')
    x.write(' '.join(origin[i].FORM).lower())
    x.write('<br>')

    x.write('<b>ATTACK: </b>')
    x.write(' '.join(attacked[i].FORM).lower())
    x.write('<br>')

    revised_index = [index for index,(org_head,att_head) in enumerate(zip(origin[i].HEAD,attacked[i].HEAD)) if org_head!=att_head]
    x.write('<b>Result Changed: </b>')
    if len(revised_index)==0:
        x.write('None')
    else:
        x.write('{}. Index: {}'.format(len(revised_index),' '.join(str(x) for x in revised_index)))
    x.write('<br>')


    x.write('<b>GOLD  : </b>')
    x.write(to_svg(gold[i]))
    x.write('<br>')

    x.write('<b>ORIGIN: </b>')
    x.write(to_svg(origin[i]))
    x.write('<br>')

    x.write('<b>ATTACK: </b>')
    x.write(to_svg(attacked[i]))
    x.write('<br>')

    x.write('<br>')

x.write('</body></html>')


