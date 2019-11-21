from dpattack.utils.corpus import Corpus, init_sentence
import re


log_file_path = '/disks/sdb/zjiehang/zhou_data/hackwhole-logit-8-1.Nov19_05-04-53.txt'
raw_corpus = Corpus.load("/disks/sdb/zjiehang/zhou_data/ptb/ptb_test_3.3.0.sd")
log_file = open(log_file_path)
out_file = open(log_file_path + '.conll', 'w')

lines = log_file.readlines()

found_sents = []
line_no = -1

while True:
    # Loop until finding the next sentence
    while True:
        line_no += 1
        if line_no >= len(lines):
            exit('EOF. processed {}'.format(len(found_sents)))
        found = re.search('[\*]{6}\s+(\d+):', lines[line_no].strip())
        if found is not None:
            found_idx = int(found.group(1))
            break
    print('Found sentence {}'.format(found_idx))
    found_sents.append(found_idx)

    exists_log = None
    while True:
        line_no += 1
        if line_no >= len(lines):
            exit('EOF. processed {}'.format(len(found_sents)))
        line = lines[line_no].strip()
        if '-----' in line:
            exists_log = True
            break
        if 'Aggregated result' in line:
            exists_log = False
            break
    print('Log existence', exists_log)

    if exists_log:
        line_no += 1   # SKIP <ROOT>
        words = []
        while True:
            line_no += 1
            if line_no >= len(lines):
                exit('EOF. processed {}'.format(len(found_sents)))
            line = lines[line_no].strip()
            if "-----" in line:
                break
            split = line.split()
            if split[2] == '*':
                words.append(split[1])
            else:
                words.append(split[2][1:])

        # print(words)

    sentence = raw_corpus[found_idx]
    for i in range(len(sentence.ID)):
        out_file.write(
            f'{sentence.ID[i]}\t{words[i] if exists_log else sentence.FORM[i]}\t_\t{sentence.POS[i]}\t{sentence.POS[i]}\t'
            f'_\t{sentence.HEAD[i]}\t{sentence.DEPREL[i]}\t_\t_\t\n')
    out_file.write('\n')




exit()

# current_line = 0
# for index, sentence in enumerate(raw_corpus.sentences):
#     try:
#         while True:
#             line = lines[current_line].strip()
#             current_line += 1
#             if line.find("******") != -1:
#                 break
#         print("Found Sentence {}: {}".format(index, line))
#         exist_in_log = False
#         while True:
#             current_line += 1
#             line = lines[current_line].strip()
#             if line.find("------------") != -1:
#                 exist_in_log = True
#                 break
#             if line.find("******") != -1:
#                 exist_in_log = False
#                 break
#         if exist_in_log:
#             current_line += 2
#             seqs = []
#             line = lines[current_line]
#             while line.find("------------") == -1:
#                 split = line.split()
#                 if split[2] == '*':
#                     seqs.append(split[1])
#                 else:
#                     seqs.append(split[2][1:])
#                 current_line += 1
#                 line = lines[current_line]

#             print("Sentence {} exists in log!".format(index))
#         else:
#             print("Sentence {} do not exists in log!".format(index))
#             seqs = sentence.FORM
#         print(' '.join(sentence.FORM))
#         print(' '.join(seqs))
#         att_corpus.append(init_sentence(
#             tuple(seqs), sentence.CPOS, sentence.HEAD, sentence.DEPREL))
#     except:
#         print('Exception occured ')
#         break
# att_corpus.save('test.conllx')
