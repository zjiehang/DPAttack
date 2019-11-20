from dpattack.utils.corpus import Corpus, init_sentence

raw_corpus = Corpus.load("/disks/sdb/zjiehang/zhou_data/ptb/ptb_test_3.3.0.sd")
att_corpus = Corpus([])

log_file = open(
    '/disks/sdb/zjiehang/zhou_data/hackwhole-word_tag-njvr.Nov20_05-22-58.bak')

lines = log_file.readlines()
current_line = 0
for index, sentence in enumerate(raw_corpus.sentences):
    try:
        while True:
            line = lines[current_line].strip()
            current_line += 1
            if line.find("******") != -1:
                break
        print("Found Sentence {}: {}".format(index, line))
        exist_in_log = False
        while True:
            current_line += 1
            line = lines[current_line].strip()
            if line.find("------------") != -1:
                exist_in_log = True
                break
            if line.find("******") != -1:
                exist_in_log = False
                break
        if exist_in_log:
            current_line += 2
            seqs = []
            line = lines[current_line]
            while line.find("------------") == -1:
                split = line.split()
                if split[2] == '*':
                    seqs.append(split[1])
                else:
                    seqs.append(split[2][1:])
                current_line += 1
                line = lines[current_line]

            print("Sentence {} exists in log!".format(index))
        else:
            print("Sentence {} do not exists in log!".format(index))
            seqs = sentence.FORM
        print(' '.join(sentence.FORM))
        print(' '.join(seqs))
        att_corpus.append(init_sentence(
            tuple(seqs), sentence.CPOS, sentence.HEAD, sentence.DEPREL))
    except:
        print('Exception occured ')
        break
att_corpus.save('test.conllx')
