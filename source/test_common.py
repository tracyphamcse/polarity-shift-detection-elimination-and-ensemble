# from eliminator import Eliminator
#
# test = "do n't particular your money . not only doe this book kill . a tedious repetition of fact and story already well known to even a casual u2 fan , it ca n't read protestant those fact continue . which bono took his name is misspelled , and perhaps most galling of all , the author ca n't read protestant the really name of larry mullen , jr. continue . i h. ave to agree that this book wa simply thrown together to make a quick buck and therefore not really really ."
# info = {
#     'd_negation': [test]
# }
# detection_info = [info]
# eliminator = Eliminator(detection_info, 'books')
# print eliminator.eliminate()[0]['d_negation'][0]

from common_algorithm import ClauseSplitter

print ClauseSplitter.split_sentence_by_negation_indicator('i dislike that movie, it is not funny, but i think it easy')
