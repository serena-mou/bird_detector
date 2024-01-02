import glob

base = '/home/serena/Data/Birds/Serena/cut_partially_labelled'

ims = glob.glob(base+'/*.jpg')
print(len(ims))

txts = glob.glob(base+'/test_results/*.txt')
print(len(txts))
