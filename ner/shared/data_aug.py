from neraug.augmentator import DictionaryReplacement,LabelWiseTokenReplacement
from neraug.scheme import IOBES

ne_dic = {'vertex': 'ANA',
          'cells': 'ANA',
          'big': 'CON'
          }
# ne_dic = {'Tokyo Big Sight': 'LOC'}
augmentator = DictionaryReplacement(ne_dic, str.split, IOBES)
x = ['the', 'ventricles', 'and', 'sulci', 'are', 'prominent']
y = ['O', 'S-ANA', 'O', 'S-ANA','O','S-ANA']

# x = ['I', 'went', 'to', 'Tokyo']
# y = ['O', 'O', 'O', 'S-LOC']

for i in range(10):
    x_augs, y_augs = augmentator.augment(x, y, n=1)
    print(x_augs, y_augs)   

