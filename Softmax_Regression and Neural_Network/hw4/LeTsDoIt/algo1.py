import math

from sympy import true


def locate_card(cards,query):
    # brute force
    position = 0
    while true:
        if cards[position] == query:
            return position
        position += 1
        if position == len(cards):
            return -1
         


# cards = [13,11,10,9,7,4,3,2,1]
# query = 7
# output = 4

# result = output

# test = {
#     'input':{
#         'cards':[13,11,10,7,4,3,1,0],
#         'query':7
#     },
#     'output': 3
#     }

# locate_card(**tests['input']) == test['output']
# edge case
# query is the first element
# query is the last element
# just one input
# no query in the input
# list of cards is empty
# cards containing repeated numbers
# number query occurs more than one position in cards

tests = []

tests.append({'input':{
        'cards':[13,11,10,7,4,3,1,0],
        'query':7
    },
    'output': 3})

result = locate_card(tests['input']['cards'],tests['input']['query'])
result

if result == tests['output']:
    print (True )

