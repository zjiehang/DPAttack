class CONSTANT(object):
    FALSE_TOKEN = -1

    AUXILIARY_VERB = ["do", "did", "does", "have", "has", "had"]

    VERB_TAG = 'VB'
    NOUN_TAG = 'NN'
    ADJ_TAG = 'JJ'
    ADV_TAG = 'RB'

    JJ_REL_MODIFIER = 'amod'
    RB_REL_MODIFIER = 'advmod'


    ADJ_NN_TAGS = [ADJ_TAG] + [NOUN_TAG]
    REAL_WORD_TAGS = ['JJ','JJR','JJS',
                      'NN','NNS','NNP','NNPS',
                      'VB','VBD','VBG','VBN','VBP','VBZ',
                      'RB','RBR','RBS']

    COMMA = ','
    PUNCT = 'punct'
