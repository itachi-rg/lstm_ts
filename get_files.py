from os import listdir
from os.path import isfile, join

def getbin(value, width):
    return format(value, "0"+str(width)+"b")

def getFileVector(absolute_path):
    cpu_arch = {'4096':0, '64':1}
    block_size = {'1m':0, '100k':1}
    data_inst =  {'inst':0, 'both':1, 'data':2}
    benchmark = {'refequake': 0, 'refmcf': 1, 'refbzip23': 2, 'refapsi': 3, 'refcc15': 4, 'defaultJ9': 5, 'refparser': 6, 'refgalgel': 7, 'refapplu': 8, 'refperlbmk2': 9, 'tinyJikesRVM': 10, 'refbzip22': 11, 'defaultJikesRVM': 12, 'refcrafty': 13, 'defaultHotSpot': 14, 'refvpr2': 15, 'smallJikesRVM': 16, 'refcc12': 17, 'refperlbmk1': 18, 'smallHotSpot': 19, 'smallJ9': 20, 'small6trace': 21, 'refammp': 22, 'refvpr1': 23, 'refart': 24, 'small1trace': 25, 'small0trace': 26, 'tinyHotSpot': 27, 'refvortex': 28, 'refperlbmk5': 29, 'refvortex3': 30, 'refmesa': 31, 'refbzip21': 32, 'refperlbmk6': 33, 'refperlbmk3': 34, 'small3trace': 35}
    prog = {'parser': 0, 'luindex': 1, 'lusearch': 2, 'fop': 3, 'batik': 4, 'gcc': 5, 'equake': 6, 'galgel': 7, 'mesa': 8, 'pmd': 9, 'vortex': 10, 'apsi': 11, 'tomcat': 12, 'crafty': 13, 'vpr': 14, 'h2': 15, 'xalan': 16, 'avrora': 17, 'applu': 18, 'art': 19, 'sunflow': 20, 'bzip2': 21, 'ammp': 22, 'perlbmk': 23, 'jython': 24, 'eclipse': 25, 'mcf': 26}
    filename_w_extn = absolute_path.split('/')[-1]
    filename = filename_w_extn.split('.')[0]
    splitnames = filename.split('-')
    vector_string = ""
    cpu_arch_vector = splitnames[-1]
    block_size_vector = splitnames[-2]
    data_inst_vector = splitnames[-3]
    benchmark_vector = ''.join(splitnames[1:-3])
    prog_vector = splitnames[0]
    print(filename)
    print(prog_vector,benchmark_vector,data_inst_vector,block_size_vector,cpu_arch_vector)
    vector_string += getbin(prog[prog_vector], 5)
    vector_string += getbin(benchmark[benchmark_vector], 6)
    vector_string += getbin(data_inst[data_inst_vector],2)
    vector_string += getbin(block_size[block_size_vector],1)
    vector_string += getbin(cpu_arch[cpu_arch_vector],1)
    print(vector_string)
    vector = [int(bit) for bit in list(vector_string)]
    return vector


filedir = '/home/rgangaraju/chaos/selectedFiles_rishikesh/files'
files = [join(filedir,f) for f in listdir(filedir) if isfile(join(filedir, f))]

cpu_arch = []
block_size = []
data_inst = []
bench = []
prog = []
for f in files:
	print("Filename ", f, " -- ", getFileVector(f))

	'''
	name = f.split('.')[0]
	splitnames = name.split('-')
	cpu_arch.append(splitnames[-1])
	block_size.append(splitnames[-2])
	data_inst.append(splitnames[-3])
	bench.append(''.join(splitnames[1:-3]))
	prog.append(splitnames[0])

#print(splitnames)
print(set(cpu_arch))
print(set(block_size))
print(set(data_inst))
print(set(bench))
print(set(prog))
#print(bench)


'''